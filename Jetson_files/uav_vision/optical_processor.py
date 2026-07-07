# uav_vision/optical_processor.py
import cv2
import config
import logging
from ultralytics import YOLO
from uav_vision.camera_AC import setup_camera, move_camera, set_light_raw

class OpticalDetector:
    def __init__(self):
        self.model_path = config.YOLO_MODEL_PATH
        self.model = None
        self.ptz, self.move_req, self.token, self.ptz_url = [None]*4
        self.visual_lock = False
        
        # Tracking telemetry offsets
        self.error_x = 0.0
        self.error_y = 0.0
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
        results = self.model(frame, stream=True, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
        target_found = False
        
        # Default screen center for normalized 640x480 boundaries
        screen_cx, screen_cy = 320, 240
        
        for r in results:
            if len(r.boxes) > 0:
                target_found = True
                # Isolate the highest confidence bounding box
                box = r.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Calculate target spatial center coordinates
                target_cx = int((x1 + x2) / 2)
                target_cy = int((y1 + y2) / 2)
                
                # Compute pixel drift errors relative to frame midpoint
                self.error_x = float(target_cx - screen_cx)
                self.error_y = float(target_cy - screen_cy)
                
                # Draw tactical OSD lock graphics
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(frame, (target_cx, target_cy), 5, (0, 0, 255), -1)
                cv2.line(frame, (screen_cx, screen_cy), (target_cx, target_cy), (0, 255, 0), 2)
                
                # Print active offset tracking metadata on the display canvas
                cv2.putText(frame, f"ERR_X: {self.error_x:.1f} | ERR_Y: {self.error_y:.1f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                break # Process single target to lock computing pipeline overhead
                
        self.visual_lock = target_found
        
        # Draw central targeting crosshair
        cv2.drawMarker(frame, (screen_cx, screen_cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
        return frame

    def track_target(self, pan_speed, tilt_speed):
        """Sends directional raw speed vector adjustments (-1.0 to 1.0) to camera motors."""
        if self.ptz is not None:
            move_camera(self.ptz, self.move_req, pan_speed, tilt_speed)

    def execute_visual_closed_loop(self):
        """
        NEW: Visual Closed-Loop P-Controller.
        Translates real-time pixel drift offsets from screen center into smooth,
        proportional ONVIF velocities, keeping the drone centered in the frame.
        """
        if self.ptz is None or not self.visual_lock:
            return
            
        # Proportional tracking formula: Error * Coefficient Gain
        pan_velocity = self.error_x * config.KP_PAN
        tilt_velocity = self.error_y * config.KP_TILT
        
        # Bound velocities inside strict ONVIF limits [-1.0, 1.0]
        final_pan = max(-1.0, min(1.0, pan_velocity))
        final_tilt = max(-1.0, min(1.0, tilt_velocity))
        
        # Dispatch fine tracking vector to the motors
        self.track_target(final_pan, final_tilt)

    def slew_to_acoustic_azimuth(self, target_azimuth):
        """Queries camera telemetry and slews smoothly towards acoustic vector."""
        if self.ptz is None:
            return
        try:
            status = self.ptz.GetStatus({'ProfileToken': self.token})
            raw_pan = status.Position.PanTilt.x
            self.current_camera_pan = ((raw_pan + 1.0) / 2.0) * 360.0
        except Exception:
            pass

        angular_error = (target_azimuth - self.current_camera_pan + 180) % 360 - 180
        p_gain = 0.04
        computed_pan_speed = angular_error * p_gain
        final_pan_speed = max(-1.0, min(1.0, computed_pan_speed))
        
        if abs(angular_error) < 4.0:
            final_pan_speed = 0.0
            
        self.track_target(final_pan_speed, config.DEFAULT_ELEVATION_ANGLE if final_pan_speed != 0.0 else 0.0)

    def trigger_deterrent(self, state_on):
        """Controls tactical hardware deterrent devices using auxiliary profiles."""
        cmd = "LightOn" if state_on else "LightOff"
        if self.ptz_url and self.token:
            set_light_raw(self.ptz_url, self.token, cmd)
            
    def calculate_safe_velocity(self, target_azimuth):
        """
        Calculates optimal velocity vector while respecting physical axis boundaries.
        Prevents motor burnout at hardware limits.
        """
        # Fetch current status
        status = self.ptz.GetStatus({'ProfileToken': self.token})
        pan_pos = status.Position.PanTilt.x * 180.0 # Normalize to degrees
        
        # 1. Kinematic Logic: If target is out of Pan range, do not move.
        if target_azimuth < config.PAN_MIN or target_azimuth > config.PAN_MAX:
            logging.warning("Target outside physical Pan limit!")
            return 0.0, 0.0
            
        # 2. P-Controller Logic for Pan
        pan_error = target_azimuth - pan_pos
        pan_speed = max(-1.0, min(1.0, pan_error * config.KP_PAN))
        
        # 3. Handle Singularity: If Pan is near limit, prioritize returning to center
        # to unlock the ability to rotate further
        return pan_speed, config.DEFAULT_ELEVATION_ANGLE * config.TILT_DIRECTION_INVERSION