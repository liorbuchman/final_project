import cv2
import os
import sys
import time
import requests
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera

# ==========================================
#              הגדרות משתמש
# ==========================================
IP = "192.168.1.10"
USER = "admin"
PASS = "admin"
ONVIF_PORT = 8899 #for asecam 
RTSP_PORT = "554"
STREAM_PATH = "live/ch0" 
RTSP_URL = f"rtsp://{USER}:{PASS}@{IP}:{RTSP_PORT}/{STREAM_PATH}"
MOVE_SPEED = 0.5

# ==========================================
#           פונקציות מערכת
# ==========================================

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
    print(f"🔌 Connecting to Camera Control at {IP}:{ONVIF_PORT}...")
    wsdl_path = find_wsdl_path()
    
    try:
        if wsdl_path:
            mycam = ONVIFCamera(IP, ONVIF_PORT, USER, PASS, wsdl_dir=wsdl_path)
        else:
            mycam = ONVIFCamera(IP, ONVIF_PORT, USER, PASS)
        
        ptz = mycam.create_ptz_service()
        media = mycam.create_media_service()
        
        media_profile = media.GetProfiles()[0]
        raw_token = media_profile.token
        if isinstance(raw_token, list): raw_token = raw_token[0]
        profile_token = str(raw_token).strip()
        
        # שמירת הכתובת המדויקת לשליחת פקודות PTZ (נצטרך אותה לפנסים)
        # לפעמים הכתובת נמצאת ב-location ולפעמים בתוך ה-binding
        ptz_url = ptz.location if hasattr(ptz, 'location') else f"http://{IP}:{ONVIF_PORT}/onvif/ptz_service"
        
        # תיקון וידוא מהירות
        move_request = ptz.create_type('ContinuousMove')
        move_request.ProfileToken = profile_token
        if move_request.Velocity is None:
            status = ptz.GetStatus({'ProfileToken': profile_token})
            move_request.Velocity = status.Position
            move_request.Velocity.PanTilt.space = None
            move_request.Velocity.Zoom.space = None

        print("✅ ONVIF Control Connected Successfully!")
        
        # אנחנו מחזירים גם את ה-URL של השירות כדי שנוכל לשלוח אליו XML ישירות
        return ptz, move_request, profile_token, ptz_url
        
    except Exception as e:
        print(f"❌ ONVIF Connection Failed: {e}")
        return None, None, None, None

def move_camera(ptz, request, x, y):
    if ptz is None: return
    try:
        request.Velocity.PanTilt.x = float(x)
        request.Velocity.PanTilt.y = float(y)
        ptz.ContinuousMove(request)
    except Exception as e:
        pass

def set_light_raw(ptz_url, profile_token, command):
    """
    פונקציית 'נשק יום הדין': עוקפת את הספרייה ושולחת XML ישירות למצלמה.
    זה פותר את כל בעיות הטיפוסים (ReferenceToken errors).
    """
    print(f"💡 Sending RAW Command: {command} ... ", end="")
    
    # בניית מעטפת XML ידנית - עוקף את כל הבדיקות
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
        # שימוש ב-requests פשוט עם אימות Digest
        response = requests.post(
            ptz_url, 
            data=xml_payload, 
            headers=headers, 
            auth=HTTPDigestAuth(USER, PASS),
            timeout=2
        )
        
        if response.status_code == 200:
            print("✅ Sent! (200 OK)")
        else:
            print(f"❌ Failed (Status {response.status_code})")
            # הדפסת השגיאה מהמצלמה אם יש
            # print(response.text) 
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")

# ==========================================
#           התוכנית הראשית
# ==========================================

def main():
    # קבלת ptz_url בנוסף לשאר המשתנים
    ptz, move_req, token, ptz_url = setup_camera()

    print(f"📷 Opening Stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("❌ Critical Error: Cannot open RTSP stream.")
        return

    print("\n🎮 SYSTEM READY!")
    print("   Move:  [W] Up, [S] Down, [A] Left, [D] Right")
    print("   Stop:  [Space]")
    print("   Light: [L] On, [K] Off")
    print("   IR:    [I] On, [O] Off")
    print("   Quit:  [Q]")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Video stream lost.")
            break
        frame = cv2.flip(frame, -1)
        frame = cv2.resize(frame, (640, 480))
        cv2.putText(frame, "WASD=Move | L/K=Light | Q=Quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Jetson Camera Control', frame)

        key = cv2.waitKey(1) & 0xFF

        if key != 255:
            if key == ord('q'):
                break
            elif key == ord('w'): move_camera(ptz, move_req, 0, -MOVE_SPEED)
            elif key == ord('s'): move_camera(ptz, move_req, 0, MOVE_SPEED)
            elif key == ord('a'): move_camera(ptz, move_req, MOVE_SPEED, 0)
            elif key == ord('d'): move_camera(ptz, move_req, -MOVE_SPEED, 0)
            elif key == 32:       move_camera(ptz, move_req, 0, 0) 
            
            # שימוש בפונקציה החדשה (Raw XML)
            elif key == ord('l'): set_light_raw(ptz_url, token, 'LightOn')
            elif key == ord('k'): set_light_raw(ptz_url, token, 'LightOff')
            elif key == ord('i'): set_light_raw(ptz_url, token, 'IrOn')
            elif key == ord('o'): set_light_raw(ptz_url, token, 'IrOff')

    if ptz: move_camera(ptz, move_req, 0, 0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()