# optical_processor.py
import cv2
import config
from ultralytics import YOLO

from uav_vision.camera_AC import setup_camera, move_camera, set_light_raw

class OpticalDetector:
    def __init__(self):
        self.model_path = config.YOLO_MODEL_PATH
        self.model = None
        self.ptz, self.move_req, self.token, self.ptz_url = [None]*4
        self.visual_lock = False

    def initialize_hardware(self):
        print("[Optical] Connecting to PTZ hardware controller via ONVIF...")
        self.ptz, self.move_req, self.token, self.ptz_url = setup_camera()
        
        print("[Optical] Initializing YOLOv8 tensor weights...")
        self.model = YOLO(self.model_path)
        self.model.to(config.DEVICE)
        print(f"[Optical] Vision pipeline hot on native execution target: {config.DEVICE}")

    def run_inference(self, frame):
        """Processes raw frame matrices directly on the GPU to locate drone targets."""
        results = self.model(frame, stream=True, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
        target_found = False
        
        for r in results:
            if len(r.boxes) > 0:
                target_found = True
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
        self.visual_lock = target_found
        return frame

    def track_target(self, pan, tilt):
        """Sends directional speed vector adjustments to the camera motors."""
        move_camera(self.ptz, self.move_req, pan, tilt)

    def trigger_deterrent(self, state_on):
        """Controls tactical hardware deterrent devices (e.g., tracking spotlights)."""
        mode = 'LightOn' if state_on else 'LightOff'
        set_light_raw(self.ptz_url, self.token, mode)