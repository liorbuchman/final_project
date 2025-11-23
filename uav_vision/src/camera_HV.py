#non relevant code - camrera dosn't have ptz

# import cv2
# import os
# import sys
# import time
# from onvif import ONVIFCamera

# # ==========================================
# #           הגדרות HIKVISION
# # ==========================================
# CAMERA_IP = "192.168.1.65"
# RTSP_PORT = "554"
# ONVIF_PORT = 80             

# # משתמש לוידאו (ראשי)
# RTSP_USER = "admin"
# RTSP_PASS = "afeka2016"

# # משתמש לשליטה (ONVIF)
# ONVIF_USER = "admin"
# ONVIF_PASS = "admin12345"

# STREAM_PATH = "Streaming/Channels/101" 
# RTSP_URL = f"rtsp://{RTSP_USER}:{RTSP_PASS}@{CAMERA_IP}:{RTSP_PORT}/{STREAM_PATH}"
# MOVE_SPEED = 0.5

# # ==========================================
# #           פונקציות מערכת
# # ==========================================

# def get_wsdl_path():
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     wsdl_path = os.path.join(script_dir, "wsdl")
#     if os.path.exists(os.path.join(wsdl_path, "devicemgmt.wsdl")):
#         return wsdl_path
#     return None

# def setup_hikvision_forced():
#     """
#     פונקציה זו עוקפת את הבדיקה 'Device support'.
#     במקום לשאול את המצלמה אם יש לה PTZ, אנחנו יוצרים את השירות ידנית.
#     """
#     print(f"🔌 Connecting to Hikvision Control (Force Mode)...")
    
#     wsdl_path = get_wsdl_path()
#     try:
#         if wsdl_path:
#             mycam = ONVIFCamera(CAMERA_IP, ONVIF_PORT, ONVIF_USER, ONVIF_PASS, wsdl_dir=wsdl_path)
#         else:
#             print("❌ Error: WSDL folder missing! Cannot force connection.")
#             return None, None
        
#         # --- הטריק: יצירה ידנית של השירות ---
#         # אנחנו לא משתמשים ב-create_ptz_service() כי הוא בודק יכולות ונכשל.
#         # אנחנו יוצרים אותו ישירות מקובץ ה-WSDL.
        
#         ptz_wsdl = os.path.join(wsdl_path, 'ptz.wsdl')
#         ptz_url = f'http://{CAMERA_IP}:{ONVIF_PORT}/onvif/ptz_service'
        
#         print(f"   Forcing service at: {ptz_url}")
        
#         # יצירת השירות בכוח
#         ptz = mycam.create_service(ptz_wsdl, ptz_url, 'PTZ')
        
#         # השגת טוקן מהמדיה
#         media = mycam.create_media_service()
#         profiles = media.GetProfiles()
#         token = profiles[0].token
        
#         # יצירת אובייקט תזוזה
#         request = ptz.create_type('ContinuousMove')
#         request.ProfileToken = token
        
#         # אתחול מהירות ידני (עוקף בדיקת סטטוס)
#         request.Velocity = ptz.create_type('PTZSpeed')
#         request.Velocity.PanTilt = ptz.create_type('Vector2D')
#         request.Velocity.PanTilt.x = 0
#         request.Velocity.PanTilt.y = 0
#         request.Velocity.PanTilt.space = "http://www.onvif.org/ver10/tptz/PanTiltSpaces/VelocityGenericSpace"
        
#         print("✅ PTZ Service Forced Successfully!")
#         return ptz, request
        
#     except Exception as e:
#         print(f"⚠️ PTZ Not Available: {e}")
#         print("   (System will run in VIDEO ONLY mode)")
#         return None, None

# def move_camera(ptz, request, x, y):
#     if ptz is None: return
#     try:
#         request.Velocity.PanTilt.x = float(x)
#         request.Velocity.PanTilt.y = float(y)
#         ptz.ContinuousMove(request)
#     except Exception as e:
#         # מתעלמים משגיאות קטנות
#         pass

# def main():
#     # 1. חיבור למנועים (במצב כוח)
#     ptz, move_req = setup_hikvision_forced()

#     # 2. חיבור לוידאו
#     print(f"📷 Opening Stream: {RTSP_URL}")
#     cap = cv2.VideoCapture(RTSP_URL)
#     os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#     if not cap.isOpened():
#         print("❌ Error: Cannot open video. Check RTSP password.")
#         return

#     print("\n🎮 SYSTEM READY!")
#     if ptz:
#         print("   [WASD] Move Camera")
#     else:
#         print("   [!] Video Only Mode (No PTZ detected)")
        
#     print("   [Q] Quit")

#     while True:
#         ret, frame = cap.read()
#         if not ret: break

#         frame = cv2.resize(frame, (640, 480))
        
#         status_text = "PTZ Active" if ptz else "NO PTZ"
#         cv2.putText(frame, f"HIKVISION | {status_text}", (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         cv2.imshow('Hikvision Control', frame)

#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('q'): break
#         elif key == ord('w'): move_camera(ptz, move_req, 0, MOVE_SPEED)
#         elif key == ord('s'): move_camera(ptz, move_req, 0, -MOVE_SPEED)
#         elif key == ord('a'): move_camera(ptz, move_req, -MOVE_SPEED, 0)
#         elif key == ord('d'): move_camera(ptz, move_req, MOVE_SPEED, 0)
#         elif key == 32:       move_camera(ptz, move_req, 0, 0)

#     if ptz: move_camera(ptz, move_req, 0, 0)
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()