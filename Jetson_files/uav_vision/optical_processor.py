# uav_vision/optical_processor.py
import cv2
import config
import logging
import time     
import math
from ultralytics import YOLO
from uav_vision.camera_AC import setup_camera, move_camera, set_light_raw, stop_camera

class OpticalDetector:
    def __init__(self):
        self.model_path = config.YOLO_MODEL_PATH
        self.model = None
        self.ptz, self.move_req, self.token, self.ptz_url = [None]*4
        self.visual_lock = False
        
        # Tracking telemetry offsets
        self.error_x = 0.0
        self.error_y = 0.0
        
        # Phase 1: Acoustic Tracking State
        self.current_camera_pan = 0.0

    def initialize_hardware(self):
        print("[Optical] Connecting to PTZ hardware controller via ONVIF...")
        self.ptz, self.move_req, self.token, self.ptz_url = setup_camera()
        
        print("[Optical] Initializing YOLOv8 tensor weights...")
        self.model = YOLO(self.model_path)
        self.model.to(config.DEVICE)
        print(f"[Optical] Vision pipeline hot on native execution target: {config.DEVICE}")

    def run_inference(self, frame):
        """Processes 640x480 frame matrices on GPU and extracts spatial target center errors."""
        results = self.model(frame, stream=False, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
        self.visual_lock = False
        
        for r in results:
            if len(r.boxes) > 0:
                self.visual_lock = True
                box = r.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                self.error_x = center_x - (config.FRAME_WIDTH // 2)
                self.error_y = center_y - (config.FRAME_HEIGHT // 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Lock onto the first (highest confidence) target
                break 
        
        return frame

    def track_target(self, pan_speed, tilt_speed):
        move_camera(self.ptz, self.move_req, pan_speed, tilt_speed)
        
    def execute_visual_closed_loop(self):
        """
        Phase 2: Visual Tracking - Zoned Control with Deadband for ONVIF Optimization.
        This approach drastically reduces network spam by sending discrete speed commands
        only when the target crosses specific bounding zones.
        """
        if self.ptz is None: 
            return
        
        # 1. Define the Deadzone - The target is centered enough, do not move the motors.
        # This prevents micro-jitters when the drone is hovering in the center.
        deadzone_x = config.FRAME_WIDTH * 0.15  # ~96 pixels from the center horizontally
        deadzone_y = config.FRAME_HEIGHT * 0.15 # ~72 pixels from the center vertically
        
        # 2. Define tered motor speeds (Gears)
        fast_speed = 0.8 # Maximum speed for when the target is escaping the frame edge
        smooth_speed = 0.4 # Slower, smooth speed for normal tracking (can be tuned)
        
        pan_speed = 0.0
        tilt_speed = 0.0
        
        # --- X-Axis (Pan) Logic ---
        abs_error_x = abs(self.error_x)
        if abs_error_x > deadzone_x:
            # If the target is near the extreme edges of the screen -> go fast. Otherwise -> smooth.
            if abs_error_x > (config.FRAME_WIDTH * 0.35):
                base_pan = fast_speed
            else:
                base_pan = smooth_speed
            
            # Apply direction based on your specific camera hardware calibration.
            # If error_x > 0, the target is on the right side of the screen.
            # Note: Right movement is negative (-) based on previous physical tests.
            pan_speed = -base_pan if self.error_x > 0 else base_pan

        # --- Y-Axis (Tilt) Logic ---
        abs_error_y = abs(self.error_y)
        if abs_error_y > deadzone_y:
            if abs_error_y > (config.FRAME_HEIGHT * 0.35):
                base_tilt = fast_speed
            else:
                base_tilt = smooth_speed
            
            # If error_y > 0, the target is in the lower part of the screen, so we move down.
            # TILT_DIRECTION_INVERSION from config handles upside-down ceiling installations.
            raw_tilt_speed = base_tilt if self.error_y > 0 else -base_tilt
            tilt_speed = raw_tilt_speed * config.TILT_DIRECTION_INVERSION

        # 3. Dispatch the command!
        # Thanks to the state-caching mechanism in camera_AC.py, an actual HTTP/SOAP request 
        # is ONLY sent over the network if the calculated speeds have physically changed.
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
        direction = "Right" if diff > 0 else "Left" if diff < 0 else "None"
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
            logging.info(f"[Optical] Slew {direction} by {degrees_to_move:.1f} degrees (Target: {safe_target})...")
            
            # Invert X axis based on hardware specifics (Right is negative)
            x_speed = -config.MOVE_SPEED if direction == "Right" else config.MOVE_SPEED
            
            # Start pan motor
            self.track_target(x_speed, 0.0)
            sleep_time = degrees_to_move * config.TIME_PER_DEGREE
            
            # Wait for movement to finish, BUT abort if YOLO sees the drone
            target_found_during_pan = self._sleep_and_check_lock(sleep_time)
            
            # Stop motor and update software position state
            stop_camera(self.ptz, self.move_req)
            self.current_camera_pan = safe_target
            
            if target_found_during_pan:
                logging.info("[Optical] Target detected during PAN movement! Stopping search.")
                return

        # If we already see the target, no need to tilt scan
        if self.visual_lock:
            return

        # ==========================================
        # STEP 2: VERTICAL SCAN (TILT SWEEP)
        # ==========================================
        logging.info("[Optical] Initiating Vertical Scan...")
        current_tilt = config.MIN_TILT
        scan_step = 20.0
        direction_tilt = "Up"
        time_per_step = scan_step * config.TIME_PER_DEGREE_TILT

        # 2A. Reset Tilt to the bottom limit before scanning
        logging.info("[Optical] Resetting tilt to bottom limit...")
        self.track_target(0.0, -config.MOVE_SPEED) # Send motor DOWN
        
        if self._sleep_and_check_lock(config.TILT_TIME_END_TO_END):
            stop_camera(self.ptz, self.move_req)
            logging.info("[Optical] Target detected while resetting tilt!")
            return
        stop_camera(self.ptz, self.move_req)

        # 2B. Execute step-by-step vertical scan
        # Limited to 20 sweeps to prevent infinite loop if drone flies away
        max_scan_sweeps = 8
        for _ in range(max_scan_sweeps):
            # Pre-check visual lock
            if self.visual_lock:
                break
                
            logging.info(f"[Optical] Scanning at {current_tilt} degrees...")
            
            # Determine motor direction for this step
            y_speed = config.MOVE_SPEED if direction_tilt == "Up" else -config.MOVE_SPEED
            
            # Start tilt motor
            self.track_target(0.0, y_speed)
            
            # Wait for step to finish, abort immediately if YOLO sees the drone
            target_found = self._sleep_and_check_lock(time_per_step)
            stop_camera(self.ptz, self.move_req)
            
            if target_found:
                logging.info(f"*** TARGET VISUALLY ACQUIRED at ~{current_tilt} degrees! ***")
                break
                
            # Update math for next step
            if direction_tilt == "Up":
                current_tilt += scan_step
            else:
                current_tilt -= scan_step
                
            # Change direction at bounds (Bounce effect)
            if current_tilt >= config.MAX_TILT:
                direction_tilt = "Down"
                current_tilt = config.MAX_TILT
            elif current_tilt <= config.MIN_TILT:
                direction_tilt = "Up"
                current_tilt = config.MIN_TILT
                
        # Ensure motor is stopped when scan finishes or aborts
        stop_camera(self.ptz, self.move_req)