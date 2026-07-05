# uav_vision/test_optical_local.py
import cv2
import sys
import os
import time
import requests
import math
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera

print("==================================================")
print("     FULLY INTEGRATED HARDWARE TEST BENCH (H.265) ")
print("==================================================")

# Resolve workspace mapping
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from uav_vision.optical_processor import OpticalDetector

# --- Native ONVIF Dynamic Resolution Functions ---
def find_wsdl_path():
    import onvif
    package_dir = os.path.dirname(onvif.__file__)
    potential_paths = [
        os.path.join(package_dir, "wsdl"),
        os.path.join(package_dir, "..", "wsdl"),
        os.path.join(os.getcwd(), "wsdl")
    ]
    for path in potential_paths:
        full_path = os.path.abspath(path)
        if os.path.exists(os.path.join(full_path, "devicemgmt.wsdl")):
            return full_path
    return None

def setup_local_camera_services():
    # Dynamic IP Resolution from config
    camera_ip = getattr(config, 'CAMERA_IP', '192.168.1.90')
    print(f"Connecting to ONVIF Discovery Core at {camera_ip}:8899...")
    wsdl_path = find_wsdl_path()
    try:
        if wsdl_path:
            mycam = ONVIFCamera(camera_ip, 8899, "admin", "admin", wsdl_dir=wsdl_path)
        else:
            mycam = ONVIFCamera(camera_ip, 8899, "admin", "admin")
        
        ptz = mycam.create_ptz_service()
        media = mycam.create_media_service()
        
        media_profile = media.GetProfiles()[0]
        raw_token = media_profile.token
        if isinstance(raw_token, list): raw_token = raw_token[0]
        profile_token = str(raw_token).strip()
        
        ptz_url = ptz.location if hasattr(ptz, 'location') else f"http://{camera_ip}:8899/onvif/ptz_service"
        
        move_request = ptz.create_type('ContinuousMove')
        move_request.ProfileToken = profile_token
        if move_request.Velocity is None:
            status = ptz.GetStatus({'ProfileToken': profile_token})
            move_request.Velocity = status.Position
            move_request.Velocity.PanTilt.space = None
            move_request.Velocity.Zoom.space = None

        print("Dynamic ONVIF Endpoint Discovery Successful!")
        return ptz, move_request, profile_token, ptz_url
    except Exception as e:
        print(f"ONVIF Endpoint Discovery Failed: {e}")
        return None, None, None, None

def move_camera_native(ptz, request, x, y):
    if ptz is None: return
    try:
        request.Velocity.PanTilt.x = float(x)
        request.Velocity.PanTilt.y = float(y)
        ptz.ContinuousMove(request)
    except Exception:
        pass

def set_light_raw_dynamic(ptz_url, profile_token, command):
    if not ptz_url: return
    print(f"Pushing Auxiliary Command '{command}' to resolved endpoint: {ptz_url}")
    xml_payload = f"""<?xml version="1.0" encoding="utf-8"?>
    <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
      <s:Body>
        <tptz:SendAuxiliaryCommand xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
          <tptz:ProfileToken>{profile_token}</tptz:ProfileToken>
          <tptz:AuxiliaryCommand>{command}</tptz:AuxiliaryCommand>
        </tptz:SendAuxiliaryCommand>
      </s:Body>
    </s:Envelope>"""

    headers = {'Content-Type': 'application/soap+xml; charset=utf-8'}
    try:
        response = requests.post(
            ptz_url, 
            data=xml_payload, 
            headers=headers, 
            auth=HTTPDigestAuth("admin", "admin"),
            timeout=2
        )
        if response.status_code == 200:
            print("Hardware accepted command packet.")
        else:
            print(f"Hardware rejected command. Status Code: {response.status_code}")
    except Exception as e:
        print(f"Transmission Error: {e}")

# --- Core Execution Flow ---

# 1. Resolve Dynamic ONVIF parameters locally using global IP
ptz, move_req, token, ptz_url = setup_local_camera_services()

# 2. Initialize CUDA YOLO Core via modern class
detector = OpticalDetector()
detector.initialize_hardware()

# 3. Spin up accelerated Nvidia video pipelines
pipeline = config.get_gstreamer_pipeline()
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("[CRITICAL] Failed to open H.265 GStreamer video stream device link.")
    sys.exit(1)

# 4. Generate bounded viewport layout window
window_name = "Drone Detection System v3 - Jetson Core"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\nSYSTEM OPERATIONAL IN JETSON ACCELERATED ENVIRONMENT!")
print("   Controls: WASD=Move | SPACE=Stop | L/K=Light | I/O=IR | Q=Quit")

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame stream interrupted.")
        break
        
    # Standard spatial inversion matrix leveling
    frame = cv2.flip(frame, -1)
    
    # RESIZE FIX: Normalizing matrix to match OSD layouts and boost YOLO inference FPS
    frame = cv2.resize(frame, (854, 480))
    
    # Run tracking inference inside the CUDA context
    results = detector.model(frame, stream=True, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
    drone_detected = False
    target_count = 0
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            if conf >= config.YOLO_CONF_THRESHOLD:
                drone_detected = True
                target_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, f'DRONE {conf}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Frame Rate Overhead telemetry tracking
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
    prev_time = curr_time

    # Construct OSD Tactical Metadata overlays
    status_color = (0, 0, 255) if drone_detected else (0, 255, 0)
    status_text = f"CRITICAL LOCK: DETECTED ({target_count})" if drone_detected else "SCANNING..."
    
    # Render banner background metrics
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f} | Backend: H.265 CUDA | Status: {status_text}", 
                (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # This position (15, 455) is now perfectly aligned at the bottom of the 480p frame
    cv2.putText(frame, "WASD=Move | SPACE=Stop | L/K=Light | I/O=IR | Q=Quit", (15, 455), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Output tracking context to native window viewport
    cv2.imshow(window_name, frame)

    # Hardware keystroke event handler mappings
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'): move_camera_native(ptz, move_req, 0, -config.MOVE_SPEED)
    elif key == ord('s'): move_camera_native(ptz, move_req, 0, config.MOVE_SPEED)
    elif key == ord('a'): move_camera_native(ptz, move_req, config.MOVE_SPEED, 0)
    elif key == ord('d'): move_camera_native(ptz, move_req, -config.MOVE_SPEED, 0)
    elif key == 32:       move_camera_native(ptz, move_req, 0, 0) # Spacebar hard stop
    elif key == ord('l'): set_light_raw_dynamic(ptz_url, token, 'LightOn')
    elif key == ord('k'): set_light_raw_dynamic(ptz_url, token, 'LightOff')
    elif key == ord('i'): set_light_raw_dynamic(ptz_url, token, 'IrOn')
    elif key == ord('o'): set_light_raw_dynamic(ptz_url, token, 'IrOff')

# Resource clean environment teardown
if ptz: move_camera_native(ptz, move_req, 0, 0)
cap.release()
cv2.destroyAllWindows()
print("==================================================")