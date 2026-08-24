# uav_vision/optical_processor.py
import cv2
import config
import logging
import time     
import math
import queue
import threading
from ultralytics import YOLO
from uav_vision.camera_AC import setup_camera, move_camera, set_light_raw, stop_camera

logger = logging.getLogger("OpticalSystem")

class OpticalDetector:
    def __init__(self):
        self.model_path = config.YOLO_MODEL_PATH
        self.model = None
        self.ptz, self.move_req, self.token, self.ptz_url = [None]*4
        self.high_conf_achieved = False
        self.visual_lock = False
        self.lock_hold_counter = 0
        
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

    def auto_calibrate_and_home(self):
        """
        Executes physical homing sequence using deterministic open-loop control.
        Assumes PAN_TIME_END_TO_END and TILT_TIME_END_TO_END in config.py 
        are perfectly measured via a manual stopwatch test.
        """
        if self.ptz is None:
            logger.error("[Calibration] PTZ connection missing!")
            return False

        logger.info("[Calibration] STARTING DETERMINISTIC HOMING...")
        
        # --- PAN CALIBRATION ---
        # 1. Drive to absolute Right limit
        logger.info("[Calibration] Finding Pan Right Limit...")
        self.track_target(-config.MOVE_SPEED, 0.0)
        time.sleep(config.PAN_TIME_END_TO_END + 2.0)
        
        # 2. Drive to Center
        logger.info("[Calibration] Moving Pan to absolute Center...")
        self.track_target(config.MOVE_SPEED, 0.0)
        time.sleep(config.PAN_TIME_END_TO_END / 2.0)
        
        self.track_target(0.0, 0.0) 
        self.current_camera_pan = 0.0
        time.sleep(0.5) 

        # --- TILT CALIBRATION ---
        # 1. Drive to absolute Bottom limit
        logger.info("[Calibration] Finding Tilt Bottom Limit...")
        self.track_target(0.0, -config.MOVE_SPEED * config.TILT_DIRECTION_INVERSION)
        time.sleep(config.TILT_TIME_END_TO_END + 2.0)
        
        # 2. Drive up to Default Elevation
        logger.info(f"[Calibration] Moving Tilt to {config.DEFAULT_ELEVATION_ANGLE}°...")
        degrees_up = config.DEFAULT_ELEVATION_ANGLE - config.MIN_TILT
        time_up = degrees_up * config.TIME_PER_DEGREE_TILT
        
        self.track_target(0.0, config.MOVE_SPEED * config.TILT_DIRECTION_INVERSION)
        time.sleep(time_up)
        
        self.track_target(0.0, 0.0) 
        self.current_tilt = config.DEFAULT_ELEVATION_ANGLE

        logger.info("[Calibration] HOMING COMPLETE! Camera centered.")
        return True
    
    def initialize_hardware(self):
        print("[Optical] Connecting to PTZ hardware controller via ONVIF...")
        self.ptz, self.move_req, self.token, self.ptz_url = setup_camera()
        
        print("[Optical] Initializing YOLOv8 tensor weights...")
        self.model = YOLO(self.model_path)
        self.model.to(config.DEVICE)
        print(f"[Optical] Vision pipeline hot on native execution target: {config.DEVICE}")

    def run_inference(self, frame):
        """Processes 640x480 frame matrices on GPU and extracts spatial target center errors."""
        if config.SAVE_LAST_FRAME_FLAG:
            #logger.info(f"[DEBUG] Frame shape before YOLO: {frame.shape}")
            cv2.imwrite("what_yolo_actually_sees.jpg", frame)
        use_half = (config.DEVICE.type == 'cuda')
        results = self.model.track(frame, 
                                   stream=False, 
                                   persist=True, 
                                   tracker="bytetrack.yaml", 
                                   half=use_half, 
                                   conf=config.YOLO_LOW_CONF_THRESHOLD, 
                                   verbose=False)
        detected_this_frame = False
        
        for r in results:
            if len(r.boxes) > 0:
                sorted_boxes = sorted(r.boxes, key=lambda b: float(b.conf[0]), reverse=True)
                box =sorted_boxes[0]
                yolo_conf = float(box.conf[0])
                

                if yolo_conf >= config.YOLO_HIGH_CONF_THRESHOLD:
                    self.high_conf_achieved = True
                    self.lock_hold_counter = 15
            
                if self.high_conf_achieved or self.lock_hold_counter > 0:
                    self.visual_lock = True
                    detected_this_frame = True
                    yolo_conf_percent = int(yolo_conf * 100)
                    logger.info(f"[YOLO] Drone visually detected! Visual Confidence: {yolo_conf:.2f}")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    with self.error_lock:
                        self.error_x = center_x - (config.FRAME_WIDTH // 2)
                        self.error_y = center_y - (config.FRAME_HEIGHT // 2)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    label = f"Drone: {yolo_conf_percent}%"
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_position = (x1, max(y1 - 10, 10)) 
                    
                    cv2.putText(frame, label, text_position, font, 0.6, (255, 255, 255), 2)
                    # Lock onto the first (highest confidence) target
                    break 

        if not detected_this_frame:
            if self.lock_hold_counter > 0:
                self.lock_hold_counter -= 1
                self.visual_lock = True 
            else:
                self.visual_lock = False
                self.high_conf_achieved = False
        
        return frame
    def _ptz_worker_loop(self):
        """
        Dedicated background thread to handle PTZ motor commands asynchronously.
        This prevents blocking the main thread during HTTP/SOAP requests to the camera.
        """
        while self.ptz_worker_running:
            try:
                pan_speed, tilt_speed = self.ptz_queue.get(timeout=0.1)
                move_camera(self.ptz, self.move_req, pan_speed, tilt_speed)
            except queue.Empty:
                pass
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

    def execute_visual_closed_loop(self):
        """Phase 2: Visual Tracking - PD Controller with Time Delta (dt)"""
        if self.ptz is None: 
            return

        if not self.visual_lock:
            self.track_target(0.0, 0.0)
            self.prev_error_x = 0.0 
            self.prev_error_y = 0.0
            self.prev_time = time.time() 
            return

        with self.error_lock:
            current_error_x = self.error_x
            current_error_y = self.error_y
        
        
        current_time = time.time()
        dt = current_time - getattr(self, 'prev_time', current_time - 0.1)
        if dt <= 0: dt = 0.001 
        self.prev_time = current_time

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
    
    def _sleep_and_check_lock(self, duration):
        """
        Sleeps for the specified duration in tiny increments.
        This allows the system to instantly interrupt the movement if YOLO finds the drone.
        Returns True if a visual lock was acquired during the sleep, False otherwise.
        """
        steps = int(duration / 0.1)
        for _ in range(steps):
            if self.visual_lock:
                return True
            time.sleep(0.1)
            
        # Sleep remaining fractional time
        time.sleep(duration % 0.1)
        return self.visual_lock

    def handle_acoustic_search(self, target_azimuth):
        """
        Unified Phase 1 (Time-Based Open Loop): 
        1. Slew horizontally (Pan) to the acoustic vector.
        2. If target is not found during Pan, begin vertical sweep (Tilt).
        Stops automatically the exact moment a visual lock is acquired.
        """
        if self.ptz is None:
            return

        # ==========================================
        # STEP 1: HORIZONTAL PAN TO TARGET DOA
        # ==========================================
        direction, safe_target, degrees_to_move = self.calculate_pan_movement(target_azimuth)

        if abs(degrees_to_move) >= 2.0 and direction != "None":
            logger.info(f"[Optical] Slew {direction} by {degrees_to_move:.1f} degrees (Target: {safe_target})...")
            
            # Invert X axis based on hardware specifics (Right is negative)
            x_speed = -config.MOVE_SPEED if direction == "Right" else config.MOVE_SPEED
            # Start pan motor
            self.track_target(x_speed, 0.0)
            sleep_time = degrees_to_move * config.TIME_PER_DEGREE_PAN

            # Wait for movement to finish, BUT abort if YOLO sees the drone
            start_time = time.time()
            target_found_during_pan = self._sleep_and_check_lock(sleep_time)
            elapsed_time = time.time() - start_time
            
            # Stop motor and update software position state
            self.track_target(0.0, 0.0)
            if target_found_during_pan and sleep_time > 0:
                fraction = min(1.0, elapsed_time / sleep_time)
                actual_moved = degrees_to_move * fraction

                if direction == "Right":
                    self.current_camera_pan -= actual_moved
                else:
                    self.current_camera_pan += actual_moved
            else:
                self.current_camera_pan = safe_target            

            if target_found_during_pan:
                logger.info(f"[Optical] Target detected during PAN! Stopped at ~{self.current_camera_pan:.1f}°")
                return

        # If we already see the target, no need to tilt scan
        if self.visual_lock:
            return

        # ==========================================
        # STEP 2: VERTICAL MACRO-SCAN
        # ==========================================
        # Assuming an effective vertical FOV of ~45 degrees.
        # We only need 3 stops to cover from 0 to 90 degrees.
        
        logger.info("[Optical] Initiating Optimized Vertical Macro-Scan...")

        for target_tilt in config.TILT_CHECKPOINTS:
            if self.visual_lock:
                break
                
            current_tilt_val = getattr(self, 'current_tilt', config.MIN_TILT)
            tilt_diff = target_tilt - current_tilt_val
            
            if abs(tilt_diff) >= 2.0:
                logger.info(f"[Optical] Macro-Slewing TILT to {target_tilt}°...")
                y_speed = (config.MOVE_SPEED if tilt_diff > 0 else -config.MOVE_SPEED) * config.TILT_DIRECTION_INVERSION
                time_to_tilt = abs(tilt_diff) * config.TIME_PER_DEGREE_TILT
                
                # --- 1. MOVE (Fast Slew) ---
                self.track_target(0.0, y_speed)
                
                # Sleep mostly blind, but keep a tiny chance to catch it mid-flight
                start_time = time.time()
                target_found_during_move = self._sleep_and_check_lock(time_to_tilt)
                elapsed_time = time.time() - start_time
                self.track_target(0.0, 0.0)

                if target_found_during_move and time_to_tilt > 0:
                    fraction = min(1.0, elapsed_time / time_to_tilt)
                    actual_moved_tilt = abs(tilt_diff) * fraction
                    if tilt_diff > 0:
                        self.current_tilt += actual_moved_tilt
                    else:
                        self.current_tilt -= actual_moved_tilt
                else:
                    self.current_tilt = target_tilt
                
                if target_found_during_move:
                    logger.info(f"[Optical] Target acquired during macro-slew to {self.current_tilt:.1f}°")
                    break

            # --- 2. STARE (Settle and Detect) ---
            logger.info(f"[Optical] Camera stationary at {target_tilt}°. Flushing buffer and staring...")
            # 0.8 seconds is usually enough to flush the RTSP buffer and stabilize Autofocus.
            # During this time, we constantly ask YOLO "do you see it now?".
            target_found = self._sleep_and_check_lock(0.8) 
            
            if target_found:
                logger.info(f"*** TARGET VISUALLY ACQUIRED at ~{self.current_tilt}°! ***")
                break

        # old version of tracking the drone after YOLO detection 
        # def execute_visual_closed_loop(self): 
        #     """
        #     Phase 2: Visual Tracking - Zoned Control with Deadband for ONVIF Optimization.
        #     This approach drastically reduces network spam by sending discrete speed commands
        #     only when the target crosses specific bounding zones.
        #     """
        #     if self.ptz is None: 
        #         return
    
        #     if not self.visual_lock:
        #         self.track_target(0.0, 0.0)  # Stop motors if no visual lock
        #         return
            
        #     # 1. Define the Deadzone - The target is centered enough, do not move the motors.
        #     # This prevents micro-jitters when the drone is hovering in the center.
        #     deadzone_x = config.FRAME_WIDTH * 0.15  # ~96 pixels from the center horizontally
        #     deadzone_y = config.FRAME_HEIGHT * 0.15 # ~72 pixels from the center vertically
            
        #     # 2. Define tered motor speeds (Gears)
        #     fast_speed = 0.8 # Maximum speed for when the target is escaping the frame edge
        #     smooth_speed = 0.4 # Slower, smooth speed for normal tracking (can be tuned)
            
        #     pan_speed = 0.0
        #     tilt_speed = 0.0
            
        #     # --- X-Axis (Pan) Logic ---
        #     abs_error_x = abs(self.error_x)
        #     if abs_error_x > deadzone_x:
        #         # If the target is near the extreme edges of the screen -> go fast. Otherwise -> smooth.
        #         if abs_error_x > (config.FRAME_WIDTH * 0.35):
        #             base_pan = fast_speed
        #         else:
        #             base_pan = smooth_speed
                
        #         # Apply direction based on your specific camera hardware calibration.
        #         # If error_x > 0, the target is on the right side of the screen.
        #         # Note: Right movement is negative (-) based on previous physical tests.
        #         pan_speed = -base_pan if self.error_x > 0 else base_pan
    
        #     # --- Y-Axis (Tilt) Logic ---
        #     abs_error_y = abs(self.error_y)
        #     if abs_error_y > deadzone_y:
        #         if abs_error_y > (config.FRAME_HEIGHT * 0.35):
        #             base_tilt = fast_speed
        #         else:
        #             base_tilt = smooth_speed
                
        #         # If error_y > 0, the target is in the lower part of the screen, so we move down.
        #         # TILT_DIRECTION_INVERSION from config handles upside-down ceiling installations.
        #         raw_tilt_speed = base_tilt if self.error_y > 0 else -base_tilt
        #         tilt_speed = raw_tilt_speed * config.TILT_DIRECTION_INVERSION
    
        #     # 3. Dispatch the command!
        #     # Thanks to the state-caching mechanism in camera_AC.py, an actual HTTP/SOAP request 
        #     # is ONLY sent over the network if the calculated speeds have physically changed.
        #     self.track_target(pan_speed, tilt_speed)

        # fix elevation
        # # ==========================================
        # # STEP 2: MOVE TILT TO FIXED MIDDLE ELEVATION
        # # ==========================================
        # # Target angle: DEFAULT_ELEVATION_ANGLE or midpoint between MIN_TILT and MAX_TILT
        # target_tilt = getattr(config, 'DEFAULT_ELEVATION_ANGLE', (config.MIN_TILT + config.MAX_TILT) / 2.0)
        # current_tilt_val = getattr(self, 'current_tilt', config.MIN_TILT)
        # tilt_diff = target_tilt - current_tilt_val

        # if abs(tilt_diff) >= 2.0:
        #     logger.info(f"[Optical] Moving TILT to fixed middle elevation ({target_tilt}°)...")
            
        #     y_speed = config.MOVE_SPEED if tilt_diff > 0 else -config.MOVE_SPEED
        #     time_to_tilt = abs(tilt_diff) * config.TIME_PER_DEGREE_TILT

        #     # Start Tilt motor directly to the center angle
        #     self.track_target(0.0, y_speed)
        #     target_found_during_tilt = self._sleep_and_check_lock(time_to_tilt)

        #     self.track_target(0.0, 0.0)
        #     self.current_tilt = target_tilt

        #     if target_found_during_tilt:
        #         logger.info("[Optical] Target detected while moving Tilt to center!")
        #         return

        # # ==========================================
        # # STEP 3: STATIONARY STARE (VERIFY YOLO DETECTION)
        # # ==========================================
        # logger.info(f"[Optical] Camera stationary at Pan: {safe_target}°, Tilt: {target_tilt}°. Waiting for YOLO lock...")
        # self.track_target(0.0, 0.0)

        # # Stand completely still for 1.5 seconds to let focus & GStreamer buffer settle, checking YOLO continuously
        # target_acquired = self._sleep_and_check_lock(1.5)

        # if target_acquired:
        #     logger.info("*** TARGET VISUALLY ACQUIRED BY YOLO! ***")
        # else:
        #     logger.info("[Optical] Stationary check finished - No visual lock acquired.")


        #old version works good but is slow and can miss the target if it flies away during the tilt scan
        # # ==========================================
        # # STEP 2: VERTICAL SCAN (TILT SWEEP)
        # # ==========================================
        # logger.info("[Optical] Initiating Vertical Scan...")
        # current_tilt = config.MIN_TILT
        # scan_step = 20.0
        # direction_tilt = "Up"
        # time_per_step = scan_step * config.TIME_PER_DEGREE_TILT

        # # 2A. Reset Tilt to the bottom limit before scanning
        # logger.info("[Optical] Resetting tilt to bottom limit...")
        # self.track_target(0.0, -config.MOVE_SPEED) # Send motor DOWN
        
        # if self._sleep_and_check_lock(config.TILT_TIME_END_TO_END):
        #     self.track_target(0.0, 0.0)
        #     logger.info("[Optical] Target detected while resetting tilt!")
        #     return
        # self.track_target(0.0, 0.0)

        # # 2B. Execute step-by-step vertical scan
        # # Limited to 20 sweeps to prevent infinite loop if drone flies away
        # max_scan_sweeps = 5
        # for _ in range(max_scan_sweeps):
        #     # Pre-check visual lock
        #     if self.visual_lock:
        #         break
                
        #     logger.info(f"[Optical] Scanning at {current_tilt} degrees...")
            
        #     # Determine motor direction for this step
        #     y_speed = config.MOVE_SPEED if direction_tilt == "Up" else -config.MOVE_SPEED
            
        #     # Start tilt motor
        #     self.track_target(0.0, y_speed)
            
        #     # Wait for step to finish, abort immediately if YOLO sees the drone
        #     target_found = self._sleep_and_check_lock(time_per_step)
        #     self.track_target(0.0, 0.0)
            
        #     if target_found:
        #         logger.info(f"*** TARGET VISUALLY ACQUIRED at ~{current_tilt} degrees! ***")
        #         break
                
        #     # Update math for next step
        #     if direction_tilt == "Up":
        #         current_tilt += scan_step
        #     else:
        #         current_tilt -= scan_step
                
        #     # Change direction at bounds (Bounce effect)
        #     if current_tilt >= config.MAX_TILT:
        #         direction_tilt = "Down"
        #         current_tilt = config.MAX_TILT
        #     elif current_tilt <= config.MIN_TILT:
        #         direction_tilt = "Up"
        #         current_tilt = config.MIN_TILT
                
        # # Ensure motor is stopped when scan finishes or aborts
        # # self.track_target(0.0, 0.0)