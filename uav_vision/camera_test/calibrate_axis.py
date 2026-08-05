import cv2
import time
import os
from onvif import ONVIFCamera
import config

def find_wsdl_path():
    import onvif
    package_dir = os.path.dirname(onvif.__file__)
    for path in [os.path.join(package_dir, "wsdl"), os.path.join(package_dir, "..", "wsdl"), os.path.join(os.getcwd(), "wsdl")]:
        full_path = os.path.abspath(path)
        if os.path.exists(os.path.join(full_path, "devicemgmt.wsdl")):
            return full_path
    return None

def setup_camera():
    wsdl_path = find_wsdl_path()
    try:
        mycam = ONVIFCamera(config.IP, config.ONVIF_PORT, config.USER, config.PASS, wsdl_dir=wsdl_path) if wsdl_path else ONVIFCamera(config.IP, config.ONVIF_PORT, config.USER, config.PASS)
        ptz = mycam.create_ptz_service()
        media = mycam.create_media_service()
        profile_token = str(media.GetProfiles()[0].token[0] if isinstance(media.GetProfiles()[0].token, list) else media.GetProfiles()[0].token).strip()
        
        move_req = ptz.create_type('ContinuousMove')
        move_req.ProfileToken = profile_token
        status = ptz.GetStatus({'ProfileToken': profile_token})
        move_req.Velocity = status.Position
        move_req.Velocity.PanTilt.space = None
        move_req.Velocity.Zoom.space = None
        return ptz, move_req
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def move_camera(ptz, request, x_speed):
    if not ptz: return
    try:
        request.Velocity.PanTilt.x = float(x_speed)
        request.Velocity.PanTilt.y = 0.0
        ptz.ContinuousMove(request)
    except Exception as e:
        pass

def main():
    ptz, move_req = setup_camera()
    cap = cv2.VideoCapture(config.RTSP_URL)
    
    print("\n--- AXIS CALIBRATION TOOL ---")
    print("[A] Hold to move LEFT (towards 0 point)")
    print("[D] Hold to move RIGHT (towards Max point)")
    print("[Space] STOP camera")
    print("[T] Start/Stop Timer")
    print("[Q] Quit")

    timer_running = False
    start_time = 0
    elapsed_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 480))
        
        # Display instructions and timer
        status_color = (0, 255, 0) if timer_running else (0, 0, 255)
        current_time = time.time() - start_time if timer_running else elapsed_time
        
        cv2.putText(frame, f"Timer: {current_time:.2f} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, "A=Left | D=Right | Space=Stop | T=Timer | Q=Quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Calibration', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            # Move Left (Positive speed due to inverted axis)
            move_camera(ptz, move_req, config.MOVE_SPEED)
        elif key == ord('d'):
            # Move Right (Negative speed due to inverted axis)
            move_camera(ptz, move_req, -config.MOVE_SPEED)
        elif key == 32: # Spacebar
            move_camera(ptz, move_req, 0.0)
        elif key == ord('t'):
            timer_running = not timer_running
            if timer_running:
                start_time = time.time()
                print("\nTimer STARTED!")
            else:
                elapsed_time = time.time() - start_time
                print(f"\nTimer STOPPED! Total time: {elapsed_time:.2f} seconds")

    move_camera(ptz, move_req, 0.0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()