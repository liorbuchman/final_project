#!/usr/bin/env python3
import time
import sys
import config
from uav_vision.camera_AC import setup_camera, move_camera, stop_camera

def main():
    print("==================================================")
    print("   PTZ HARDWARE CALIBRATION & TIMING TOOL")
    print("==================================================")
    
    ptz, move_req, token, ptz_url = setup_camera()
    if ptz is None:
        print("Failed to connect to camera. Exiting.")
        sys.exit(1)

    speed = config.MOVE_SPEED
    print(f"\n[INFO] Connected. Using config.MOVE_SPEED = {speed}")
    
    # --- PAN CALIBRATION ---
    print("\n--- STEP 1: PAN CALIBRATION ---")
    input("Press ENTER to move camera to the extreme LEFT...")
    move_camera(ptz, move_req, -speed, 0.0)
    input("Camera moving LEFT... Press ENTER exactly when it hits the mechanical limit (stops moving)!")
    stop_camera(ptz, move_req)
    time.sleep(1)

    print("\nReady to measure PAN time.")
    input("Press ENTER to start moving RIGHT and begin the timer...")
    start_time = time.time()
    move_camera(ptz, move_req, speed, 0.0)
    
    input("Camera moving RIGHT... Press ENTER exactly when it hits the right mechanical limit!")
    pan_time = time.time() - start_time
    stop_camera(ptz, move_req)
    print(f">>> MEASURED PAN_TIME_END_TO_END: {pan_time:.3f} seconds")
    time.sleep(1)

    # --- TILT CALIBRATION ---
    print("\n--- STEP 2: TILT CALIBRATION ---")
    input("Press ENTER to move camera to the extreme BOTTOM...")
    move_camera(ptz, move_req, 0.0, -speed)
    input("Camera moving DOWN... Press ENTER exactly when it hits the bottom mechanical limit!")
    stop_camera(ptz, move_req)
    time.sleep(1)

    print("\nReady to measure TILT time.")
    input("Press ENTER to start moving UP and begin the timer...")
    start_time = time.time()
    move_camera(ptz, move_req, 0.0, speed)
    
    input("Camera moving UP... Press ENTER exactly when it hits the top mechanical limit!")
    tilt_time = time.time() - start_time
    stop_camera(ptz, move_req)
    print(f">>> MEASURED TILT_TIME_END_TO_END: {tilt_time:.3f} seconds")
    
    # --- SUMMARY & TESTING ---
    print("\n==================================================")
    print("   CALIBRATION SUMMARY (Copy to config.py)")
    print("==================================================")
    print(f"PAN_TIME_END_TO_END = {pan_time:.2f}")
    print(f"TILT_TIME_END_TO_END = {tilt_time:.2f}")
    
    print("\n--- STEP 3: TEST ELEVATION ANGLE ---")
    tilt_range = float(input("What is the total physical physical tilt range in degrees? (e.g., 90): "))
    time_per_deg_tilt = tilt_time / tilt_range
    print(f"[INFO] Calculated TIME_PER_DEGREE_TILT: {time_per_deg_tilt:.4f}s")
    
    while True:
        try:
            target_angle = float(input("\nEnter an elevation angle to test (or press Ctrl+C to quit): "))
            
            # Reset to bottom
            print("Resetting to bottom limit...")
            move_camera(ptz, move_req, 0.0, -speed)
            time.sleep(tilt_time + 1.0)
            stop_camera(ptz, move_req)
            time.sleep(0.5)
            
            # Move to angle
            print(f"Moving to {target_angle} degrees...")
            time_to_move = target_angle * time_per_deg_tilt
            move_camera(ptz, move_req, 0.0, speed)
            time.sleep(time_to_move)
            stop_camera(ptz, move_req)
            print("Done! Look at the camera. Is this a good default elevation?")
            
        except KeyboardInterrupt:
            print("\nExiting calibration tool.")
            break

if __name__ == "__main__":
    main()