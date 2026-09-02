
#!/usr/bin/env python3
import os
import time
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
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

    pan_speed = config.PAN_MOVE_SPEED
    tilt_speed = config.TILT_MOVE_SPEED
    print(f"\n[INFO] Connected. Using config.PAN_MOVE_SPEED = {pan_speed}, config.TILT_MOVE_SPEED = {tilt_speed}")

    # --- PAN CALIBRATION ---
    print("\n--- STEP 1: PAN CALIBRATION ---")
    input("Press ENTER to move camera to the extreme RIGHT...")
    move_camera(ptz, move_req, -pan_speed, 0.0)
    input("Camera moving RIGHT... Press ENTER exactly when it hits the mechanical limit (stops moving)!")
    stop_camera(ptz, move_req)
    time.sleep(1)

    print("\nReady to measure PAN time.")
    input("Press ENTER to start moving LEFT and begin the timer...")
    start_time = time.time()
    move_camera(ptz, move_req, pan_speed, 0.0)

    input("Camera moving LEFT... Press ENTER exactly when it hits the left mechanical limit!")
    pan_time = time.time() - start_time
    stop_camera(ptz, move_req)
    print(f">>> MEASURED PAN_TIME_END_TO_END: {pan_time:.3f} seconds (at PAN_MOVE_SPEED={pan_speed})")
    time.sleep(1)

    # --- TILT CALIBRATION ---
    print("\n--- STEP 2: TILT CALIBRATION ---")
    input("Press ENTER to move camera to the extreme BOTTOM (MIN_TILT)...")
    move_camera(ptz, move_req, 0.0, -tilt_speed * config.TILT_DIRECTION_INVERSION)
    input("Camera moving DOWN... Press ENTER exactly when it hits the bottom mechanical limit!")
    stop_camera(ptz, move_req)
    time.sleep(1)

    print("\nReady to measure TILT time.")
    input("Press ENTER to start moving UP and begin the timer...")
    start_time = time.time()
    move_camera(ptz, move_req, 0.0, tilt_speed * config.TILT_DIRECTION_INVERSION)

    input("Camera moving UP... Press ENTER exactly when it hits the top mechanical limit!")
    tilt_time = time.time() - start_time
    stop_camera(ptz, move_req)
    print(f">>> MEASURED TILT_TIME_END_TO_END: {tilt_time:.3f} seconds (at TILT_MOVE_SPEED={tilt_speed})")
    
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

    print("Resetting to bottom limit to establish baseline...")
    move_camera(ptz, move_req, 0.0, -tilt_speed * config.TILT_DIRECTION_INVERSION)
    time.sleep(tilt_time + 1.0)
    stop_camera(ptz, move_req)
    current_angle = 0.0

    while True:
        try:
            target_angle = float(input("\nEnter an elevation angle to test (or press Ctrl+C to quit): "))
            delta_angle = target_angle - current_angle
            time_to_move = abs(delta_angle) * time_per_deg_tilt

            if delta_angle == 0:
                print("Already at this angle.")
                continue

            base_tilt_speed = tilt_speed if delta_angle > 0 else -tilt_speed
            applied_tilt_speed = base_tilt_speed * config.TILT_DIRECTION_INVERSION
            
            print(f"Moving from {current_angle}° to {target_angle}° (Delta: {delta_angle}°)...")
            move_camera(ptz, move_req, 0.0, applied_tilt_speed)
            time.sleep(time_to_move) 
            stop_camera(ptz, move_req)
            
            current_angle = target_angle 
            print("Done! Look at the camera. Is this a good default elevation?")
        
            
        except KeyboardInterrupt:
            print("\nExiting calibration tool.")
            break

if __name__ == "__main__":
    main()