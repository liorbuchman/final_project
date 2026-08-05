import cv2
import time
import requests
import os
from onvif import ONVIFCamera
from requests.auth import HTTPDigestAuth

# Import from our modular files
import config
import movement_math

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

def setup_camera():
    print(f"Connecting to Camera Control at {config.IP}:{config.ONVIF_PORT}...")
    wsdl_path = find_wsdl_path()
    
    try:
        if wsdl_path:
            mycam = ONVIFCamera(config.IP, config.ONVIF_PORT, config.USER, config.PASS, wsdl_dir=wsdl_path)
        else:
            mycam = ONVIFCamera(config.IP, config.ONVIF_PORT, config.USER, config.PASS)
        
        ptz = mycam.create_ptz_service()
        media = mycam.create_media_service()
        
        media_profile = media.GetProfiles()[0]
        raw_token = media_profile.token
        if isinstance(raw_token, list): raw_token = raw_token[0]
        profile_token = str(raw_token).strip()
        
        ptz_url = ptz.location if hasattr(ptz, 'location') else f"http://{config.IP}:{config.ONVIF_PORT}/onvif/ptz_service"
        
        move_request = ptz.create_type('ContinuousMove')
        move_request.ProfileToken = profile_token
        if move_request.Velocity is None:
            status = ptz.GetStatus({'ProfileToken': profile_token})
            move_request.Velocity = status.Position
            move_request.Velocity.PanTilt.space = None
            move_request.Velocity.Zoom.space = None

        print("ONVIF Control Connected Successfully!")
        return ptz, move_request, profile_token, ptz_url
        
    except Exception as e:
        print(f"ONVIF Connection Failed: {e}")
        return None, None, None, None

def move_camera(ptz, request, x, y):
    if ptz is None: return
    try:
        request.Velocity.PanTilt.x = float(x)
        request.Velocity.PanTilt.y = float(y)
        ptz.ContinuousMove(request)
    except Exception as e:
        print(f"Move error: {e}")

def stop_camera(ptz, request):
    """Sends a zero velocity command to stop the camera."""
    move_camera(ptz, request, 0.0, 0.0)

def scan_for_target(ptz, move_req):
    """
    מבצע סריקה אנכית מדורגת מ-0 עד 90 מעלות, וחוזר חלילה.
    הסריקה נעצרת ברגע שהמטרה מזוהה (מודמה על ידי לחיצה על האות 's').
    """
    print("\n[Target Locked on Pan] Initiating Vertical Scan...")
    print("Press 's' (Scan Stop) on the video window to simulate target visual detection.")
    
    current_tilt = config.MIN_TILT
    scan_step = 10.0 # מעלות לכל קפיצה
    direction = "Up"
    
    # חישוב הזמן לקפיצה אחת (נשתמש בניחוש של 8 שניות מ-0 ל-90 שהגדרנו)
    time_per_step = scan_step * (config.TILT_TIME_END_TO_END / config.TILT_RANGE_SOFTWARE)

    # המצלמה חייבת להתחיל מ-0
    move_camera(ptz, move_req, 0.0, -config.MOVE_SPEED) # פקודת ירידה עד הסוף
    time.sleep(config.TILT_TIME_END_TO_END)
    move_camera(ptz, move_req, 0.0, 0.0)

    while True:
        # בדיקה האם המשתמש עצר את הסריקה (סימולציה של זיהוי)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('s'):
            print(f"*** TARGET DETECTED at Tilt: {current_tilt} degrees! ***")
            break
        elif key == ord('q'):
            # יציאה מוחלטת מהתוכנית
            return "Quit"

        # ביצוע הקפיצה
        print(f"Scanning at {current_tilt} degrees...")
        y_speed = config.MOVE_SPEED if direction == "Up" else -config.MOVE_SPEED
        
        move_camera(ptz, move_req, 0.0, y_speed)
        time.sleep(time_per_step)
        move_camera(ptz, move_req, 0.0, 0.0)
        
        # עדכון הזווית הנוכחית
        if direction == "Up":
            current_tilt += scan_step
        else:
            current_tilt -= scan_step
            
        # שינוי כיוון אם הגענו לגבול
        if current_tilt >= config.MAX_TILT:
            direction = "Down"
            current_tilt = config.MAX_TILT
        elif current_tilt <= config.MIN_TILT:
            direction = "Up"
            current_tilt = config.MIN_TILT
            
    return current_tilt
def track_doa_target(ptz, move_req, current_angle, target_doa):
    """
    Executes the calculated movement on the physical hardware using the calibrated limits.
    """
    # 1. משיכת 3 המשתנים בסדר הנכון
    direction, safe_target, degrees_to_move = movement_math.calculate_movement(current_angle, target_doa)
    
    if abs(degrees_to_move) < 2.0 or direction == "None":
        print("Target is already in frame. No movement needed.")
        return current_angle

    print(f"Auto-Tracking: Moving {direction} by {abs(degrees_to_move):.1f} degrees (Target: {safe_target})...")
    
    # 2. תיקון ציר ה-X ההפוך של המצלמה שלכם
    if direction == "Right":
        x_speed = -config.MOVE_SPEED  # ימינה זה מינוס
    else:
        x_speed = config.MOVE_SPEED   # שמאלה זה פלוס
    
    # 3. תחילת תנועה (שימי לב אם פונקציית ה-move_camera שלך דורשת 3 או 4 משתנים, התאמתי למה ששלחת)
    try:
        move_camera(ptz, move_req, x_speed, 0.0)
    except TypeError:
        # במקרה והפונקציה שלך מקבלת רק 3 משתנים
        move_camera(ptz, move_req, x_speed)
    
    # 4. שימוש בזמן המדויק שחישבנו מהכיול של ה-26.61 שניות
    sleep_time = abs(degrees_to_move) * config.TIME_PER_DEGREE
    time.sleep(sleep_time)
    
    # 5. עצירת המנוע
    try:
        stop_camera(ptz, move_req)
    except NameError:
        # למקרה שאין לך stop_camera אלא רק move_camera עם מהירות 0
        try:
            move_camera(ptz, move_req, 0.0, 0.0)
        except TypeError:
            move_camera(ptz, move_req, 0.0)
    
    # 6. החזרת היעד הבטוח בלבד (בלי מודולו 360 שמקלקל את העוגנים)
    return safe_target

def main():
    ptz, move_req, token, ptz_url = setup_camera()
    
    print(f"Opening Stream: {config.RTSP_URL}")
    cap = cv2.VideoCapture(config.RTSP_URL)
    
    if not cap.isOpened():
        print("Critical Error: Cannot open RTSP stream.")
        return

    current_camera_angle = 0.0

    print("SYSTEM READY!")
    print("Press 't' to enter a target DOA manually.")
    print("Press 'q' to quit.")
        

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, -1)
        frame = cv2.resize(frame, (640, 480))
        cv2.putText(frame, f"Angle: {current_camera_angle} | T=Track | Q=Quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Jetson Camera Control', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('t'):
            # Pause stream slightly to get console input
            try:
                doa_input = float(input("\nEnter detected DOA target angle: "))
                current_camera_angle = track_doa_target(ptz, move_req, current_camera_angle, doa_input)
                result = scan_for_target(ptz, move_req)
                
                if result == "Quit":
                    break
                else:
                    print(f"System is holding position at Pan: {current_camera_angle}, Tilt: {result}")
                    
            except ValueError as e:
                print(f"Error! Detail: {e}")

    stop_camera(ptz, move_req)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()