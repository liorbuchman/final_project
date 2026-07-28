# uav_vision/optical_processor.py
import cv2
import config
import logging
import time     
import math
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
        
        # Phase 1: Acoustic Tracking State
        self.current_camera_pan = 0.0
        self.last_status_poll_time = 0.0

    def initialize_hardware(self):
        print("[Optical] Connecting to PTZ hardware controller via ONVIF...")
        self.ptz, self.move_req, self.token, self.ptz_url = setup_camera()
        
        print("[Optical] Initializing YOLOv8 tensor weights...")
        self.model = YOLO(self.model_path)
        self.model.to(config.DEVICE)
        print(f"[Optical] Vision pipeline hot on native execution target: {config.DEVICE}")

    def run_inference(self, frame):
        """Processes 640x480 frame matrices on GPU and extracts spatial target center errors."""
        # OPTIMIZATION: stream=False prevents generator memory leaks for single frames
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
        """Phase 2: Visual Tracking P-Controller."""
        if self.ptz is None: return
        
        pan_speed = max(-1.0, min(1.0, self.error_x * config.KP_PAN))
        raw_tilt_speed = self.error_y * config.KP_TILT
        
        tilt_speed = max(-1.0, min(1.0, raw_tilt_speed * config.TILT_DIRECTION_INVERSION))
        
        self.track_target(pan_speed, tilt_speed)

    def handle_acoustic_search(self, target_azimuth):
        """
        Unified Phase 1: Slew to vector safely (respecting hard stops), then scan elevation.
        """
        if self.ptz is None:
            return

        # Map acoustic vector to PTZ hard-stop coordinate system
        raw_target_pan = (target_azimuth + config.CAMERA_MOUNT_OFFSET) % 360
        if raw_target_pan > 180:
            raw_target_pan -= 360
            
        target_pan_deg = max(-170.0, min(170.0, raw_target_pan))

        # OPTIMIZATION: Throttle ONVIF SOAP requests to 2Hz max to prevent blocking the FSM thread
        current_time = time.time()
        if current_time - self.last_status_poll_time > 0.5:
            try:
                status = self.ptz.GetStatus({'ProfileToken': self.token})
                self.current_camera_pan = status.Position.PanTilt.x * 170.0 
                self.last_status_poll_time = current_time
            except Exception:
                pass # Use last known position if failed

        pan_error = target_pan_deg - self.current_camera_pan
        
        if abs(pan_error) > 5.0:
            pan_speed = max(-1.0, min(1.0, pan_error * config.KP_PAN_ACOUSTIC))
            tilt_speed = 0.0 
            self.track_target(pan_speed, tilt_speed)
        else:
            pan_speed = 0.0 
            raw_tilt_speed = math.sin(current_time * config.SWEEP_FREQUENCY) * config.SWEEP_SPEED_MULTIPLIER
            tilt_speed = raw_tilt_speed * config.TILT_DIRECTION_INVERSION
            self.track_target(pan_speed, tilt_speed)