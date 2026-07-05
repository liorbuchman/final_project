#!/usr/bin/env python3
# uav_vision/test_optical_local.py

import cv2
import sys
import os
import time
import math
import logging
import datetime

print("==================================================")
print("     OPTICALLY STABILIZED HARDWARE TEST BENCH     ")
print("          WITH LOCK (PURPLE) & SUSPECT (YELLOW)   ")
print("==================================================")

# Resolve workspace mapping to locate global configuration module
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_dir)

import config
from uav_vision.optical_processor import OpticalDetector
from uav_vision.camera_AC import set_light_raw

# Setup dynamic telemetry logging directory
optical_logs_dir = os.path.join(root_dir, "logs", "optical")
os.makedirs(optical_logs_dir, exist_ok=True)

# Generate unique filename using timestamp to prevent data overwrites
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_name = f"optical_run_{current_time}.log"
log_file_path = os.path.join(optical_logs_dir, log_file_name)

# Initialize centralized logging context
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)],
    force=True # <-- CRITICAL: Forces Python to attach the FileHandler despite earlier library impor
)

# 1. Initialize localized vision context and ONVIF hardware link
detector = OpticalDetector()
detector.initialize_hardware()
logging.info("SUCCESS: Optical Subsystem initialized. Logger channel is hot.")

# 2. Instantiate video capture using optimized GStreamer pipeline
pipeline = config.get_gstreamer_pipeline()
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    logging.critical("Failed to open GStreamer video stream device link.")
    sys.exit(1)

# 3. Establish specialized display viewport
window_name = "Drone Detection System v3 - Jetson Core"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 854, 480) 

# Fetch target validation confidence threshold from configuration
LOCK_THRESHOLD = config.YOLO_CONF_THRESHOLD 
print(f"\n🎮 SYSTEM OPERATIONAL! LOCK Threshold: {LOCK_THRESHOLD}")

prev_time = time.time()
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logging.warning("Hardware digital video matrix stream interrupted.")
        break
        
    frame_counter += 1
    
    # a) Invert matrix orientation to compensate for inverted mechanical installation
    frame = cv2.flip(frame, -1)
    
    # b) CRITICAL FIX: Scale frame to target model aspect ratio (640x480) BEFORE inference
    # This aligns interpolation parameters with PC execution, eliminating the Domain Shift
    inference_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    
    # c) Run CNN inference with a low threshold floor (0.05) to capture tracking suspects
    results = detector.model(inference_frame, stream=True, conf=0.05, verbose=False)
    
    drone_detected = False
    target_count = 0
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = detector.model.names[cls_id] if cls_id in detector.model.names else f"Unknown"
            
            # --- Two-Tier Tactical HUD Classification ---
            
            # Tier 1: CRITICAL LOCK State (Solid Purple Badge) - Meets validation requirements
            if conf >= LOCK_THRESHOLD:
                drone_detected = True
                target_count += 1
                
                logging.info(f"[Frame {frame_counter}] CRITICAL LOCK: '{cls_name}' Conf: {conf:.2f} Bbox: [{x1}, {y1}, {x2}, {y2}]")
                
                cv2.rectangle(inference_frame, (x1, y1), (x2, y2), (255, 0, 255), 3) 
                label = f"LOCK: {cls_name.upper()} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_y = max(y1, h + 15)
                
                cv2.rectangle(inference_frame, (x1 - 2, text_y - h - 12), (x1 + w + 10, text_y), (255, 0, 255), -1)
                cv2.putText(inference_frame, label, (x1 + 5, text_y - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Tier 2: SUSPECT Tracking State (Solid Yellow Badge) - Low confidence tracking indicator
            elif conf >= 0.15:
                logging.info(f"[Frame {frame_counter}] SUSPECT TARGET: '{cls_name}' Conf: {conf:.2f} Bbox: [{x1}, {y1}, {x2}, {y2}]")
                cv2.rectangle(inference_frame, (x1, y1), (x2, y2), (0, 255, 255), 1) 
                
                label = f"SUSPECT: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_y = max(y1, h + 10)
                
                cv2.rectangle(inference_frame, (x1 - 1, text_y - h - 8), (x1 + w + 6, text_y), (0, 255, 255), -1)
                cv2.putText(inference_frame, label, (x1 + 3, text_y - 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Runtime operational matrix profiling calculations
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
    prev_time = curr_time

    # Construct visual instrumentation HUD overlay elements
    status_color = (0, 0, 255) if drone_detected else (0, 255, 0)
    status_text = f"CRITICAL LOCK: DETECTED ({target_count})" if drone_detected else "SCANNING..."
    
    # Upper metadata bar
    cv2.rectangle(inference_frame, (0, 0), (inference_frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(inference_frame, f"FPS: {fps:.1f} | Core: Orin Nano CUDA | Status: {status_text}", 
                (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    # Lower user input mapping bar
    cv2.rectangle(inference_frame, (0, inference_frame.shape[0] - 35), (inference_frame.shape[1], inference_frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(inference_frame, "WASD=Move | SPACE=Stop | L/K=Light | Q=Quit", 
                (15, inference_frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Rescale structural matrix viewport for window layout without compromising mathematical bounding inputs
    display_frame = cv2.resize(inference_frame, (854, 480))
    cv2.imshow(window_name, display_frame)

    # Core keystroke interrupt polling sequence mapping
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'): detector.track_target(0, -config.MOVE_SPEED)
    elif key == ord('s'): detector.track_target(0, config.MOVE_SPEED)
    elif key == ord('a'): detector.track_target(config.MOVE_SPEED, 0)
    elif key == ord('d'): detector.track_target(-config.MOVE_SPEED, 0)
    elif key == 32:       detector.track_target(0, 0) 
    elif key == ord('l'): detector.trigger_deterrent(True)
    elif key == ord('k'): detector.trigger_deterrent(False)
    elif key == ord('i'): set_light_raw(detector.ptz_url, detector.token, 'IrOn')
    elif key == ord('o'): set_light_raw(detector.ptz_url, detector.token, 'IrOff')

# Resource teardown context execution sequence
detector.track_target(0, 0)
cap.release()
cv2.destroyAllWindows()
logging.info("Hardware bench context destroyed safely.")