# uav_vision/camera_AC.py
import cv2
import os
import sys
import time
import requests
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

IP = getattr(config, 'CAMERA_IP', '192.168.1.90')
USER = "admin"
PASS = "admin"
ONVIF_PORT = 8899 

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
    print(f"🔌 Connecting to ONVIF Camera Controller at {IP}:{ONVIF_PORT}...")
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
        
        ptz_url = ptz.location if hasattr(ptz, 'location') else f"http://{IP}:{ONVIF_PORT}/onvif/ptz_service"
        
        move_request = ptz.create_type('ContinuousMove')
        move_request.ProfileToken = profile_token
        if move_request.Velocity is None:
            status = ptz.GetStatus({'ProfileToken': profile_token})
            move_request.Velocity = status.Position
            move_request.Velocity.PanTilt.space = None
            move_request.Velocity.Zoom.space = None

        print("✅ ONVIF Hardware Controller Connected Successfully!")
        return ptz, move_request, profile_token, ptz_url
        
    except Exception as e:
        print(f"❌ ONVIF Hardware Connection Failed: {e}")
        return None, None, None, None

def move_camera(ptz, request, x, y):
    if ptz is None: return
    try:
        request.Velocity.PanTilt.x = float(x)
        request.Velocity.PanTilt.y = float(y)
        ptz.ContinuousMove(request)
    # 🚀 FIXED: Printing motor faults instead of forcing silent failures
    except Exception as e:
        print(f"\n[PTZ ERROR] Failed to push motor speed vector: {e}")

def set_light_raw(ptz_url, profile_token, command):
    if not ptz_url: return
    print(f"💡 Sending RAW Auxiliary XML Command: {command} ... ", end="")
    
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
            auth=HTTPDigestAuth(USER, PASS),
            timeout=2
        )
        if response.status_code == 200:
            print("✅ Sent! (200 OK)")
        else:
            print(f"❌ Failed (Status {response.status_code})")
            
    except Exception as e:
        print(f"\n❌ Auxiliary Connection Error: {e}")