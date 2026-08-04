#!/usr/bin/env python3
"""
Tactical Drone Field Recorder with Full PTZ Keyboard Control
Controls: WASD for Pan/Tilt, Space to Stop, R to Record, Z/X for Zoom, Q to Quit.
"""

import cv2
import time
import os
import sys
import datetime
from onvif import ONVIFCamera

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Camera Credentials & Network Setup
CAMERA_IP = "192.168.1.90"
ONVIF_PORT = 8899
RTSP_PORT = 554
USER = "admin"
PASS = "admin"

RTSP_URL = f"rtsp://{USER}:{PASS}@{CAMERA_IP}:{RTSP_PORT}/live/ch0"
SAVE_DIR = os.path.join(script_dir, "recorded_drones")

os.makedirs(SAVE_DIR, exist_ok=True)

class PTZRecorderController:
    def __init__(self):
        self.ptz = None
        self.move_request = None
        self.profile_token = None
        self.is_recording = False
        self.video_writer = None
        self.current_filename = ""
        self.ptz_speed = 0.4  # מהירות תנועה דיפולטיבית (מתוך טווח של 1.0- עד 1.0)

    def setup_onvif(self):
        """Initializes ONVIF connection for PTZ Control."""
        print(f" Connecting to ONVIF Camera at {CAMERA_IP}:{ONVIF_PORT}...")
        try:
            cam = ONVIFCamera(CAMERA_IP, ONVIF_PORT, USER, PASS)
            self.ptz = cam.create_ptz_service()
            media = cam.create_media_service()
            
            profile = media.GetProfiles()[0]
            raw_token = profile.token
            if isinstance(raw_token, list):
                raw_token = raw_token[0]
            self.profile_token = str(raw_token).strip()
            
            # Create Continuous Move Request for Pan/Tilt/Zoom
            self.move_request = self.ptz.create_type('ContinuousMove')
            self.move_request.ProfileToken = self.profile_token
            
            if self.move_request.Velocity is None:
                status = self.ptz.GetStatus({'ProfileToken': self.profile_token})
                self.move_request.Velocity = status.Position
                self.move_request.Velocity.PanTilt.space = None
                self.move_request.Velocity.Zoom.space = None
                
            print(" ONVIF PTZ Controller Connected Successfully!")
            return True
        except Exception as e:
            print(f" ONVIF Connection Warning: {e}")
            print("Video capture will continue, but PTZ keyboard controls will be disabled.")
            return False

    def move_ptz(self, pan_speed=0.0, tilt_speed=0.0, zoom_speed=0.0):
        """
        Sends Continuous Move vectors to the camera.
        pan_speed: -1.0 (Left) to 1.0 (Right)
        tilt_speed: -1.0 (Down) to 1.0 (Up)
        zoom_speed: -1.0 (Out) to 1.0 (In)
        """
        if not self.ptz or not self.move_request:
            print(" PTZ service is not active.")
            return
            
        try:
            self.move_request.Velocity.PanTilt.x = float(pan_speed)
            self.move_request.Velocity.PanTilt.y = float(tilt_speed)
            self.move_request.Velocity.Zoom.x = float(zoom_speed)
            
            self.ptz.ContinuousMove(self.move_request)
            print(f" PTZ Move -> Pan: {pan_speed}, Tilt: {tilt_speed}, Zoom: {zoom_speed}")
        except Exception as e:
            print(f" Failed to move PTZ: {e}")

    def stop_motion(self):
        """Stops all physical PTZ motion."""
        if self.ptz and self.profile_token:
            try:
                self.ptz.Stop({'ProfileToken': self.profile_token})
                print(" PTZ Motion Stopped.")
            except Exception as e:
                print(f" Stop command failed: {e}")

    def toggle_recording(self, frame_width=1280, frame_height=720, fps=10.0):
        """Toggles video recording for YOLO dataset creation."""
        if not self.is_recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_filename = os.path.join(SAVE_DIR, f"drone_field_rec_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.current_filename, fourcc, fps, (frame_width, frame_height)
            )
            self.is_recording = True
            print(f" RECORDING STARTED: {self.current_filename}")
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print(f" RECORDING STOPPED: {self.current_filename}")

    def run(self):
        self.setup_onvif()
        
        print(f" Opening RTSP Stream: {RTSP_URL}")
        cap = cv2.VideoCapture(RTSP_URL)
        
        if not cap.isOpened():
            print(" Error: Unable to open RTSP video stream.")
            return

        print("\n" + "="*50)
        print(" TACTICAL PTZ CAMERA CONTROLLER & FIELD RECORDER")
        print("="*50)
        print(" [S] - Tilt UP           [W] - Tilt DOWN")
        print(" [D] - Pan LEFT          [A] - Pan RIGHT")
        print(" [Space] - STOP PTZ Motion")
        print(" [R] - Toggle Record Video (MP4 for YOLO)")
        print(" [Z/X] - Zoom In / Zoom Out")
        print(" [Q] - Quit Application")
        print("="*50 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_corrected = cv2.flip(frame, -1)
            record_frame = cv2.resize(frame_corrected, (1280, 720))

            if self.is_recording and self.video_writer:
                self.video_writer.write(record_frame)

            display_frame = record_frame.copy()
            status_text = " RECORDING..." if self.is_recording else " STANDBY (Press 'R' to record)"
            status_color = (0, 0, 255) if self.is_recording else (255, 255, 255)
            
            cv2.rectangle(display_frame, (0, 0), (1280, 40), (0, 0, 0), -1)
            cv2.putText(display_frame, status_text, (20, 26), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if self.is_recording:
                cv2.circle(display_frame, (1240, 20), 10, (0, 0, 255), -1)

            cv2.imshow("Tactical Drone PTZ Recorder", display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('r'), ord('R')):
                self.toggle_recording(1280, 720)
            elif key in (ord('w'), ord('W')):
                self.move_ptz(tilt_speed=self.ptz_speed)     
            elif key in (ord('s'), ord('S')):
                self.move_ptz(tilt_speed=-self.ptz_speed)  
            elif key in (ord('a'), ord('A')):
                self.move_ptz(pan_speed=-self.ptz_speed)   
            elif key in (ord('d'), ord('D')):
                self.move_ptz(pan_speed=self.ptz_speed)     
            elif key == 32: 
                self.stop_motion()
            elif key in (ord('z'), ord('Z')):
                self.move_ptz(zoom_speed=self.ptz_speed)    
            elif key in (ord('x'), ord('X')):
                self.move_ptz(zoom_speed=-self.ptz_speed)    

        if self.is_recording and self.video_writer:
            self.video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print(" Session closed safely.")

if __name__ == "__main__":
    app = PTZRecorderController()
    app.run()