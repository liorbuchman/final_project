# uav_vision/optical_processor.py
import cv2
import config
import logging
import time
import math
import numpy as np
import queue
import threading
from ultralytics import YOLO
from uav_vision.camera_AC import setup_camera, move_camera, set_light_raw, stop_camera

logger = logging.getLogger("OpticalSystem")

class OpticalDetector:
    def __init__(self):
        self.model_path = config.YOLO_MODEL_PATH
        self.model = None
        self.cap = None  # cv2.VideoCapture handle; opened by main_system.initialize_system()
                          # before any CUDA model load - see optical_master_loop()
        self.ptz, self.move_req, self.token, self.ptz_url = [None]*4
        self.high_conf_achieved = False
        self.visual_lock = False
        self.lock_hold_counter = 0

        # ByteTrack track-id continuity: once a strong detection acquires the
        # lock we follow *that* track id, not "whichever box is highest-conf
        # this frame". Stops the lock hopping between distractors and stops a
        # single 0.5 frame latching high_conf_achieved forever on junk.
        self.locked_track_id = None
        self.id_missing_frames = 0     # frames the locked id has been unmatched (grace = YOLO_ID_RELEASE_FRAMES)
        self.lowconf_frames = 0        # consecutive frames with no detection >= YOLO_HIGH_CONF_THRESHOLD (ceiling = YOLO_LOCK_MAX_LOWCONF_FRAMES)
        self.last_error_update = 0.0   # time.time() of the last box-derived error write (staleness guard in execute_visual_closed_loop)

        # Tracking telemetry offsets
        self.error_x = 0.0
        self.error_y = 0.0
        #error history for derivative term
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.error_lock = threading.Lock()
    
        self.ptz_queue = queue.Queue(maxsize=1) 
        self.ptz_worker_running = True
        self.ptz_thread = threading.Thread(target=self._ptz_worker_loop, daemon=True, name="PTZ_Worker")
        self.ptz_thread.start()
        
        self.current_camera_pan = 0.0
        self.current_tilt = config.DEFAULT_ELEVATION_ANGLE
        # Speed last commanded by execute_visual_closed_loop; used to dead-reckon
        # current_camera_pan/current_tilt while closed-loop visual tracking is
        # driving the motors (see _integrate_ptz_motion).
        self.prev_pan_speed = 0.0
        self.prev_tilt_speed = 0.0

        # --- Non-blocking acoustic-search state machine -------------------
        # Replaces the old blocking handle_acoustic_search(): the FSM thread
        # now advances this ONE phase transition per 10Hz tick instead of
        # blocking for up to ~25s (which froze DOA intake, the lost-timer and
        # the watchdog). Phases: idle -> pan -> tilt_move -> tilt_stare -> done.
        self._search_phase = "idle"
        self._search_result = None            # None | "acquired" | "exhausted"
        self._search_target_az = 0.0
        self._search_phase_started = 0.0
        self._search_phase_deadline = 0.0
        self._search_pan_dir = "None"
        self._search_pan_target = 0.0
        self._search_pan_from = 0.0
        self._search_pan_degs = 0.0
        self._search_tilt_idx = 0
        self._search_tilt_target = 0.0
        self._search_tilt_from = 0.0
        self._search_tilt_degs = 0.0
        self._search_last_az = None
        self._search_last_finished = 0.0

        # --- Non-blocking "return tilt to default elevation" mini-move ----
        self._tilt_home_phase = "idle"        # idle | moving
        self._tilt_home_deadline = 0.0
        self._tilt_home_target = 0.0

    def auto_calibrate_and_home(self, calibrate_tilt=True):
        """
        Executes physical homing sequence using deterministic open-loop control.
        Assumes PAN_TIME_END_TO_END and TILT_TIME_END_TO_END in config.py
        are perfectly measured via a manual stopwatch test.

        calibrate_tilt=False re-homes PAN only and leaves TILT where it is -
        used by the periodic re-home when PERIODIC_REHOME_TILT_ENABLED is off.
        Runs on its own thread (calibration_routine / _periodic_rehome), never
        on the FSM thread, so its blocking sleeps are fine.
        """
        if self.ptz is None:
            logger.error("[Calibration] PTZ connection missing!")
            return False

        logger.info("[Calibration] STARTING DETERMINISTIC HOMING...")

        # --- PAN CALIBRATION ---
        # 1. Drive to absolute Right limit
        logger.info("[Calibration] Finding Pan Right Limit...")
        self.track_target(-config.PAN_MOVE_SPEED, 0.0)
        time.sleep(config.PAN_TIME_END_TO_END + 2.0)

        # 2. Drive to Center
        logger.info("[Calibration] Moving Pan to absolute Center...")
        self.track_target(config.PAN_MOVE_SPEED, 0.0)
        time.sleep(config.PAN_TIME_END_TO_END / 2.0)
        
        self.track_target(0.0, 0.0) 
        self.current_camera_pan = 0.0
        time.sleep(0.5) 

        # --- TILT CALIBRATION ---
        if calibrate_tilt:
            # 1. Drive to absolute Bottom limit
            logger.info("[Calibration] Finding Tilt Bottom Limit...")
            self.track_target(0.0, -config.TILT_MOVE_SPEED * config.TILT_DIRECTION_INVERSION)
            time.sleep(config.TILT_TIME_END_TO_END + 2.0)

            # 2. Drive up to Default Elevation
            logger.info(f"[Calibration] Moving Tilt to {config.DEFAULT_ELEVATION_ANGLE}°...")
            degrees_up = config.DEFAULT_ELEVATION_ANGLE - config.MIN_TILT
            time_up = degrees_up * config.TIME_PER_DEGREE_TILT

            self.track_target(0.0, config.TILT_MOVE_SPEED * config.TILT_DIRECTION_INVERSION)
            time.sleep(time_up)

            self.track_target(0.0, 0.0)
            self.current_tilt = config.DEFAULT_ELEVATION_ANGLE
        else:
            logger.info("[Calibration] TILT homing skipped (calibrate_tilt=False) - PAN only.")

        logger.info("[Calibration] HOMING COMPLETE! Camera centered.")
        return True
    
    def initialize_hardware(self):
        print("[Optical] Connecting to PTZ hardware controller via ONVIF...")
        self.ptz, self.move_req, self.token, self.ptz_url = setup_camera()
        
        print("[Optical] Initializing YOLOv8 tensor weights...")
        self.model = YOLO(self.model_path)
        if str(self.model_path).endswith('.pt'):
            # .to(device) is only valid for native PyTorch (.pt) models.
            # Exported formats (TensorRT .engine, ONNX, ...) are bound to a
            # device/precision at export time; device is passed per-call to
            # track()/predict() instead (see run_inference and the warmup below).
            self.model.to(config.DEVICE)

        # ultralytics defers building the real inference backend (AutoBackend)
        # and fusing conv+bn layers - which needs its own CUDA/cuBLAS
        # allocation - until the *first* .track()/.predict() call, not
        # .to(device). Under GPU memory pressure that first call can fail
        # with CUBLAS_STATUS_ALLOC_FAILED; forcing it here, during
        # calibration, means that failure (if it happens) is visible and
        # controlled instead of silently killing optical_master_loop
        # mid-mission on the first real detection. Mirrors run_inference's
        # own .track() call exactly, on a blank frame, so it's the same code
        # path that would otherwise fail live.
        print("[Optical] Warming up YOLO inference engine...")
        warmup_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        use_half = (config.DEVICE.type == 'cuda')
        self.model.track(warmup_frame,
                          stream=False,
                          persist=True,
                          tracker="bytetrack.yaml",
                          half=use_half,
                          conf=config.YOLO_LOW_CONF_THRESHOLD,
                          device=config.DEVICE,
                          verbose=False)
        print(f"[Optical] Vision pipeline hot on native execution target: {config.DEVICE}")

    def _clear_visual_lock(self):
        """Full reset of the visual-tracking state - back to 'need a strong
        detection to (re)acquire'."""
        self.visual_lock = False
        self.high_conf_achieved = False
        self.lock_hold_counter = 0
        self.lowconf_frames = 0
        self.locked_track_id = None
        self.id_missing_frames = 0

    def run_inference(self, frame):
        """Runs YOLO+ByteTrack and maintains visual_lock with three hysteresis
        mechanisms working together:
          1. track-id continuity  - follow the id we locked onto, not the
             highest-conf box of the frame (stops lock-hopping between drones);
          2. id-miss grace        - if that id is briefly unmatched, HOLD
             position for YOLO_ID_RELEASE_FRAMES rather than snapping to a
             distractor or dropping the lock;
          3. low-confidence ceiling - drop the lock after
             YOLO_LOCK_MAX_LOWCONF_FRAMES frames with nothing >= HIGH_CONF,
             so a single 0.5 frame can't keep us chasing 0.3 noise forever.
        """
        if config.SAVE_LAST_FRAME_FLAG:
            cv2.imwrite("what_yolo_actually_sees.jpg", frame)

        use_half = (config.DEVICE.type == 'cuda')
        results = self.model.track(frame,
                                   stream=False,
                                   persist=True,
                                   tracker="bytetrack.yaml",
                                   half=use_half,
                                   conf=config.YOLO_LOW_CONF_THRESHOLD,
                                   device=config.DEVICE,
                                   verbose=False)

        hold_frames     = getattr(config, 'YOLO_LOCK_HOLD_FRAMES', 15)
        id_release      = getattr(config, 'YOLO_ID_RELEASE_FRAMES', 12)
        lowconf_ceiling = getattr(config, 'YOLO_LOCK_MAX_LOWCONF_FRAMES', 25)

        boxes = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = list(r.boxes)
            break

        def _bid(b):
            try:
                return int(b.id[0]) if b.id is not None else None
            except Exception:
                return None

        chosen = None
        chosen_conf = 0.0

        # 1. Follow the locked track id if it's on screen.
        if self.locked_track_id is not None:
            for b in boxes:
                if _bid(b) == self.locked_track_id:
                    chosen = b
                    chosen_conf = float(b.conf[0])
                    break
            if chosen is not None:
                self.id_missing_frames = 0
            else:
                self.id_missing_frames += 1
                if self.id_missing_frames < id_release:
                    # 2. Grace: keep the lock, hold last-known error, do NOT
                    # grab a different box this frame. execute_visual_closed_loop's
                    # VISUAL_ERROR_STALE_SECS guard stops the motors so we don't
                    # slew blind on a stale error while bridging.
                    self.visual_lock = True
                    if self.lock_hold_counter > 0:
                        self.lock_hold_counter -= 1
                    return frame
                logger.info(f"[YOLO] Track id {self.locked_track_id} unmatched for "
                            f"{self.id_missing_frames} frames - releasing lock.")
                self.locked_track_id = None
                self.id_missing_frames = 0

        # 3. No locked id (or just released): (re)acquire on a strong detection,
        #    or bridge on a weak one only if we're already inside a lock session.
        if chosen is None and boxes:
            best = max(boxes, key=lambda b: float(b.conf[0]))
            best_conf = float(best.conf[0])
            if best_conf >= config.YOLO_HIGH_CONF_THRESHOLD:
                chosen = best
                chosen_conf = best_conf
                new_id = _bid(best)
                if new_id is not None and new_id != self.locked_track_id:
                    logger.info(f"[YOLO] Visual lock acquired on track id {new_id} "
                                f"(conf {best_conf:.2f}).")
                self.locked_track_id = new_id
                self.id_missing_frames = 0
            elif self.high_conf_achieved or self.lock_hold_counter > 0:
                chosen = best
                chosen_conf = best_conf
                if self.locked_track_id is None and _bid(best) is not None:
                    self.locked_track_id = _bid(best)

        detected_this_frame = chosen is not None and (self.high_conf_achieved or self.lock_hold_counter > 0
                                                      or chosen_conf >= config.YOLO_HIGH_CONF_THRESHOLD)

        if detected_this_frame:
            if chosen_conf >= config.YOLO_HIGH_CONF_THRESHOLD:
                self.high_conf_achieved = True
                self.lock_hold_counter = hold_frames
                self.lowconf_frames = 0
            else:
                self.lowconf_frames += 1

            if self.lowconf_frames >= lowconf_ceiling:
                logger.info(f"[YOLO] No detection >= {config.YOLO_HIGH_CONF_THRESHOLD:.2f} "
                            f"for {self.lowconf_frames} frames - dropping visual lock (target degraded to noise).")
                self._clear_visual_lock()
                return frame

            self.visual_lock = True
            yolo_conf = chosen_conf
            logger.info(f"[YOLO] Drone visually detected! Visual Confidence: {yolo_conf:.2f}")
            x1, y1, x2, y2 = map(int, chosen.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            with self.error_lock:
                self.error_x = center_x - (config.FRAME_WIDTH // 2)
                self.error_y = center_y - (config.FRAME_HEIGHT // 2)
            self.last_error_update = time.time()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Drone: {int(yolo_conf * 100)}%",
                        (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            if self.lock_hold_counter > 0:
                self.lock_hold_counter -= 1
                self.visual_lock = True
            else:
                self._clear_visual_lock()

        return frame

    def _ptz_worker_loop(self):
        """
        Dedicated background thread to handle PTZ motor commands asynchronously.
        This prevents blocking the main thread during HTTP/SOAP requests to the camera.

        Also resends the last commanded velocity on every empty-queue tick
        (~10x/sec) whenever it's non-zero. This is what actually drives the
        keep-alive: a long open-loop pan phase in the acoustic search commands
        a speed once (in start_acoustic_search) and does not re-push it for up
        to ~20s - step_acoustic_search only re-commands at phase boundaries -
        so without a periodic resend here the camera (many budget ONVIF PTZ
        units auto-stop ContinuousMove a few seconds after the last command)
        can silently stop moving mid-slew. move_camera's own
        PTZ_KEEPALIVE_INTERVAL throttles this down to one real HTTP call per
        second, so calling it this often is cheap.
        """
        last_pan, last_tilt = 0.0, 0.0
        while self.ptz_worker_running:
            try:
                last_pan, last_tilt = self.ptz_queue.get(timeout=0.1)
                move_camera(self.ptz, self.move_req, last_pan, last_tilt)
            except queue.Empty:
                if last_pan != 0.0 or last_tilt != 0.0:
                    move_camera(self.ptz, self.move_req, last_pan, last_tilt)
            except Exception as e:
                logger.error(f"[PTZ Worker] Error moving camera: {e}")

    def track_target(self, pan_speed, tilt_speed, wait=False):
        #move_camera(self.ptz, self.move_req, pan_speed, tilt_speed) # old version
        if self.ptz_queue.full():
            try:
                self.ptz_queue.get_nowait()
            except queue.Empty:
                pass
        self.ptz_queue.put((pan_speed, tilt_speed))

        if wait:
            time.sleep(0.15)

    def _integrate_ptz_motion(self, dt):
        """
        Dead-reckons current_camera_pan/current_tilt forward by the speed that
        was actually commanded during the last dt seconds. Mirrors the same
        PAN_MOVE_SPEED/TILT_MOVE_SPEED-referenced calibration (TIME_PER_DEGREE_PAN/TILT)
        that the acoustic search uses for its open-loop moves, just
        generalized to the continuously-variable speed the PD controller in
        execute_visual_closed_loop produces - so current_camera_pan/current_tilt
        stay accurate even while closed-loop visual tracking is driving the
        motors (previously only the open-loop scan updated them, so ENGAGED
        tracking silently desynced the software's belief from the physical
        camera position).
        """
        pan_delta = (self.prev_pan_speed / config.PAN_MOVE_SPEED) * dt / config.TIME_PER_DEGREE_PAN
        tilt_delta = ((self.prev_tilt_speed / config.TILT_MOVE_SPEED) * dt / config.TIME_PER_DEGREE_TILT
                       * config.TILT_DIRECTION_INVERSION)

        self.current_camera_pan = max(config.MIN_ANGLE, min(config.MAX_ANGLE,
                                       self.current_camera_pan + pan_delta))
        self.current_tilt = max(config.MIN_TILT, min(config.MAX_TILT,
                                 self.current_tilt + tilt_delta))

    def execute_visual_closed_loop(self):
        """Phase 2: Visual Tracking - PD Controller with Time Delta (dt)"""
        if self.ptz is None:
            return

        current_time = time.time()
        dt = current_time - getattr(self, 'prev_time', current_time - 0.1)
        if dt <= 0: dt = 0.001
        self.prev_time = current_time

        # Account for the motion the previously-commanded speed produced over
        # this tick before deciding on a new one.
        self._integrate_ptz_motion(dt)

        # Bound blind slewing: if the visual error hasn't been refreshed by a
        # real box within VISUAL_ERROR_STALE_SECS (e.g. during a run_inference
        # id-miss grace bridge, or the split second between losing the lock and
        # the FSM leaving ENGAGED) hold position instead of driving on a stale
        # target offset.
        error_is_stale = (current_time - self.last_error_update) > getattr(config, 'VISUAL_ERROR_STALE_SECS', 0.5)
        if not self.visual_lock or error_is_stale:
            self.track_target(0.0, 0.0)
            self.prev_error_x = 0.0
            self.prev_error_y = 0.0
            self.prev_pan_speed = 0.0
            self.prev_tilt_speed = 0.0
            return

        with self.error_lock:
            current_error_x = self.error_x
            current_error_y = self.error_y

        deadzone_x = config.FRAME_WIDTH * 0.10
        deadzone_y = config.FRAME_HEIGHT * 0.10
        
        pan_speed = 0.0
        tilt_speed = 0.0
        
        # --- X-Axis (Pan) Logic - PD ---
        abs_error_x = abs(current_error_x)
        if abs_error_x > deadzone_x:
           
            delta_x = (current_error_x - self.prev_error_x) / dt 
            
            p_term_x = current_error_x * config.KP_PAN
            d_term_x = delta_x * config.KD_PAN
            
            raw_pan = -1.0 * (p_term_x + d_term_x)
            pan_speed = max(-1.0, min(1.0, raw_pan))

        # --- Y-Axis (Tilt) Logic - PD ---
        abs_error_y = abs(current_error_y)
        if abs_error_y > deadzone_y:
            delta_y = (current_error_y - self.prev_error_y) / dt 
            
            p_term_y = current_error_y * config.KP_TILT
            d_term_y = delta_y * config.KD_TILT
            
            raw_tilt = (p_term_y + d_term_y) * config.TILT_DIRECTION_INVERSION
            tilt_speed = max(-1.0, min(1.0, raw_tilt))

        self.prev_error_x = current_error_x
        self.prev_error_y = current_error_y
        self.prev_pan_speed = pan_speed
        self.prev_tilt_speed = tilt_speed

        self.track_target(pan_speed, tilt_speed)

   
    def calculate_pan_movement(self, target_doa):
        """Calculates direction and degrees needed to reach the acoustic DOA."""
        target_angle = target_doa
        
        # Convert 0-360 range to -180 to 180 range
        if target_angle > 180:
            target_angle = target_angle - 360
            
        # Clamp target to physical mechanical limits
        safe_target = max(config.MIN_ANGLE, min(target_angle, config.MAX_ANGLE))
        
        # Calculate delta
        diff = safe_target - self.current_camera_pan
        direction = "Left" if diff > 0 else "Right" if diff < 0 else "None"
        degrees_to_move = abs(diff)
        
        return direction, safe_target, degrees_to_move
    
    # ------------------------------------------------------------------
    # Non-blocking acoustic search (replaces the blocking handle_acoustic_search)
    # ------------------------------------------------------------------
    # The FSM thread advances this ONE phase transition per 10Hz tick instead
    # of blocking for up to ~25s inside a full pan slew + vertical macro-scan
    # (which froze fresh-DOA intake, the TARGET_LOST_TIMEOUT timer and the
    # watchdog). Phases: idle -> pan -> tilt_move -> tilt_stare -> done.

    def search_active(self):
        return self._search_phase not in ("idle", "done")

    def search_phase_name(self):
        return self._search_phase

    def search_recently_ran(self, azimuth):
        """True if a full macro-scan aimed at ~this azimuth finished within the
        last SEARCH_RESCAN_MIN_INTERVAL seconds - the FSM uses this to avoid a
        tight re-sweep loop on a stationary acoustic target it can't see."""
        interval = getattr(config, 'SEARCH_RESCAN_MIN_INTERVAL', 2.0)
        if self._search_last_az is None:
            return False
        if time.time() - self._search_last_finished > interval:
            return False
        delta = abs((azimuth - self._search_last_az + 180.0) % 360.0 - 180.0)
        return delta < config.DOA_MAX_JUMP_DEG

    def start_acoustic_search(self, target_azimuth):
        """Begin a non-blocking acoustic search. Advanced one phase per FSM
        tick by step_acoustic_search()."""
        if self.ptz is None:
            self._search_phase = "done"
            self._search_result = "exhausted"
            return

        now = time.time()
        self._search_target_az = target_azimuth
        self._search_last_az = target_azimuth
        self._search_result = None
        self._search_tilt_idx = 0

        if self.visual_lock:
            self._search_phase = "done"
            self._search_result = "acquired"
            self._search_last_finished = now
            return

        direction, safe_target, degrees_to_move = self.calculate_pan_movement(target_azimuth)
        if abs(degrees_to_move) >= 2.0 and direction != "None":
            logger.info(f"[Optical] Slew {direction} by {degrees_to_move:.1f} degrees "
                        f"(Target: {safe_target:.1f})...")
            self._search_pan_dir = direction
            self._search_pan_target = safe_target
            self._search_pan_from = self.current_camera_pan
            self._search_pan_degs = degrees_to_move
            self._search_phase = "pan"
            self._search_phase_started = now
            self._search_phase_deadline = now + degrees_to_move * config.TIME_PER_DEGREE_PAN
            x_speed = -config.PAN_MOVE_SPEED if direction == "Right" else config.PAN_MOVE_SPEED
            self.track_target(x_speed, 0.0)
        else:
            logger.info("[Optical] Initiating Optimized Vertical Macro-Scan...")
            self._begin_tilt_phase(now)

    def _begin_tilt_phase(self, now):
        """Advance to the next TILT_CHECKPOINTS entry that needs a move (or a
        stare if we're already there); finish the search once they're used up."""
        checkpoints = config.TILT_CHECKPOINTS
        while self._search_tilt_idx < len(checkpoints):
            target_tilt = float(checkpoints[self._search_tilt_idx])
            tilt_diff = target_tilt - self.current_tilt
            if abs(tilt_diff) >= 2.0:
                logger.info(f"[Optical] Macro-Slewing TILT to {target_tilt:.1f}°...")
                y_speed = ((config.TILT_MOVE_SPEED if tilt_diff > 0 else -config.TILT_MOVE_SPEED)
                           * config.TILT_DIRECTION_INVERSION)
                self._search_tilt_target = target_tilt
                self._search_tilt_from = self.current_tilt
                self._search_tilt_degs = abs(tilt_diff)
                self._search_phase = "tilt_move"
                self._search_phase_started = now
                self._search_phase_deadline = now + abs(tilt_diff) * config.TIME_PER_DEGREE_TILT
                self.track_target(0.0, y_speed)
                return
            logger.info(f"[Optical] Camera stationary at {target_tilt:.1f}°. Flushing buffer and staring...")
            self._search_tilt_target = target_tilt
            self._search_phase = "tilt_stare"
            self._search_phase_started = now
            self._search_phase_deadline = now + getattr(config, 'SEARCH_TILT_STARE_SECS', 0.8)
            self.track_target(0.0, 0.0)
            return
        self._search_phase = "done"
        self._search_result = "exhausted"
        self._search_last_finished = time.time()
        self.track_target(0.0, 0.0)

    def step_acoustic_search(self, fresh_azimuth):
        """Advance the search by at most one phase transition. Returns
        'running' | 'acquired' | 'exhausted'. Never blocks."""
        phase = self._search_phase
        if phase in ("idle", "done"):
            return self._search_result or "exhausted"

        now = time.time()

        if self.visual_lock:
            self._finalize_partial_motion(now)
            self.track_target(0.0, 0.0)
            self._search_phase = "done"
            self._search_result = "acquired"
            self._search_last_finished = now
            logger.info(f"*** TARGET VISUALLY ACQUIRED during {phase} at "
                        f"~pan {self.current_camera_pan:.1f}° / tilt {self.current_tilt:.1f}° ***")
            return "acquired"

        if phase == "pan":
            replan = getattr(config, 'SEARCH_PAN_REPLAN_DEG', 0.0)
            if replan > 0.0:
                drift = abs((fresh_azimuth - self._search_target_az + 180.0) % 360.0 - 180.0)
                if drift > replan:
                    self._finalize_partial_motion(now)
                    logger.info(f"[Optical] DOA shifted {drift:.0f}° mid-slew - re-planning pan.")
                    self.start_acoustic_search(fresh_azimuth)
                    return "running"
            if now >= self._search_phase_deadline:
                self.track_target(0.0, 0.0)
                self.current_camera_pan = max(config.MIN_ANGLE,
                                              min(config.MAX_ANGLE, self._search_pan_target))
                logger.info("[Optical] Initiating Optimized Vertical Macro-Scan...")
                self._begin_tilt_phase(now)
            return "running"

        if phase == "tilt_move":
            if now >= self._search_phase_deadline:
                self.track_target(0.0, 0.0)
                self.current_tilt = max(config.MIN_TILT, min(config.MAX_TILT, self._search_tilt_target))
                logger.info(f"[Optical] Camera stationary at {self.current_tilt:.1f}°. "
                            f"Flushing buffer and staring...")
                self._search_phase = "tilt_stare"
                self._search_phase_started = now
                self._search_phase_deadline = now + getattr(config, 'SEARCH_TILT_STARE_SECS', 0.8)
            return "running"

        if phase == "tilt_stare":
            if now >= self._search_phase_deadline:
                self._search_tilt_idx += 1
                self._begin_tilt_phase(now)
            return "running"

        return "running"

    def _finalize_partial_motion(self, now):
        """A moving phase was cut short (visual lock / DOA re-plan / abort):
        dead-reckon how far the axis actually travelled before it is stopped."""
        phase = self._search_phase
        if phase == "pan":
            span = max(1e-6, self._search_phase_deadline - self._search_phase_started)
            frac = min(1.0, max(0.0, (now - self._search_phase_started) / span))
            moved = self._search_pan_degs * frac
            pan = (self._search_pan_from - moved if self._search_pan_dir == "Right"
                   else self._search_pan_from + moved)
            self.current_camera_pan = max(config.MIN_ANGLE, min(config.MAX_ANGLE, pan))
        elif phase == "tilt_move":
            span = max(1e-6, self._search_phase_deadline - self._search_phase_started)
            frac = min(1.0, max(0.0, (now - self._search_phase_started) / span))
            moved = self._search_tilt_degs * frac
            tilt = (self._search_tilt_from + moved if self._search_tilt_target >= self._search_tilt_from
                    else self._search_tilt_from - moved)
            self.current_tilt = max(config.MIN_TILT, min(config.MAX_TILT, tilt))

    def abort_acoustic_search(self):
        """Stop any in-progress search and halt the motors. Called by the FSM
        on every exit from TRACKING (lock acquired / timeout / grace-hold)."""
        if self.search_active():
            self._finalize_partial_motion(time.time())
        self._search_phase = "idle"
        self._search_result = None
        self.track_target(0.0, 0.0)

    # ------------------------------------------------------------------
    # Non-blocking tilt-home (replaces the blocking return_to_default_elevation)
    # ------------------------------------------------------------------
    def start_tilt_home(self):
        """Begin returning TILT to DEFAULT_ELEVATION_ANGLE. Stepped by
        step_tilt_home() from the FSM SCANNING state."""
        if self.ptz is None:
            self._tilt_home_phase = "idle"
            return
        target = config.DEFAULT_ELEVATION_ANGLE
        diff = target - self.current_tilt
        if abs(diff) < 2.0:
            self._tilt_home_phase = "idle"
            return
        logger.info(f"[Optical] Returning TILT to default elevation ({target}°)...")
        y_speed = ((config.TILT_MOVE_SPEED if diff > 0 else -config.TILT_MOVE_SPEED)
                   * config.TILT_DIRECTION_INVERSION)
        self._tilt_home_target = target
        self._tilt_home_deadline = time.time() + abs(diff) * config.TIME_PER_DEGREE_TILT
        self._tilt_home_phase = "moving"
        self.track_target(0.0, y_speed)

    def step_tilt_home(self):
        """Returns True while still moving, False when done/idle."""
        if self._tilt_home_phase != "moving":
            return False
        if self.visual_lock or time.time() >= self._tilt_home_deadline:
            self.track_target(0.0, 0.0)
            if self.visual_lock:
                logger.info("[Optical] Target reacquired while returning to default elevation.")
            else:
                self.current_tilt = self._tilt_home_target
            self._tilt_home_phase = "idle"
            return False
        return True

    def tilt_home_active(self):
        return self._tilt_home_phase == "moving"
