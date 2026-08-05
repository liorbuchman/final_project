import cv2
import os
import sys
import requests
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Fetch credentials from the configuration
IP = getattr(config, 'CAMERA_IP', '192.168.1.90')
USER = getattr(config, 'CAMERA_USER', 'admin')
PASS = getattr(config, 'CAMERA_PASS', 'admin')
ONVIF_PORT = 8899 

# State caching to prevent network spamming
last_pan_speed = None
last_tilt_speed = None

def find_wsdl_path():
    """Locates the WSDL directory required by ONVIFCamera."""
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
    """Initializes the connection to the physical PTZ camera using ONVIF."""
    print(f"Connecting to ONVIF Camera Controller at {IP}:{ONVIF_PORT}...")
    wsdl_path = find_wsdl_path()
    
    try:
        if wsdl_path:
            mycam = ONVIFCamera(IP, ONVIF_PORT, USER, PASS, wsdl_path)
        else:
            print("WSDL directory not found. Assuming standard site-packages install.")
            mycam = ONVIFCamera(IP, ONVIF_PORT, USER, PASS)
            
        media = mycam.create_media_service()
        ptz = mycam.create_ptz_service()
        media_profile = media.GetProfiles()[0]
        token = media_profile.token

        # We create a dummy request just to pass the token around, 
        # but we will reconstruct the actual movement payload dynamically in move_camera()
        request = ptz.create_type('ContinuousMove')
        request.ProfileToken = token
            
        print("ONVIF Hardware Controller Connected Successfully!")
        return ptz, request, token, f"http://{IP}/onvif/ptz_service"
    except Exception as e:
        print(f"CRITICAL ONVIF FAILURE: Could not connect to PTZ camera. Error: {e}")
        return None, None, None, None

def move_camera(ptz, request, x, y):
    """Sends a continuous movement vector (Pan, Tilt) to the camera using a failsafe dictionary payload."""
    global last_pan_speed, last_tilt_speed
    if ptz is None or request is None: return
    
    # Round to avoid micro-jitter sending
    x_rounded = round(float(x), 2)
    y_rounded = round(float(y), 2)
    
    # Skip HTTP request if motor speeds haven't changed
    if x_rounded == last_pan_speed and y_rounded == last_tilt_speed:
        return
        
    try:
        # BULLETPROOF FIX: Dynamically construct the SOAP payload as a Python Dictionary.
        # This completely prevents 'NoneType' errors caused by the Zeep library losing object states.
        safe_payload = {
            'ProfileToken': request.ProfileToken,
            'Velocity': {
                'PanTilt': {
                    'x': x_rounded,
                    'y': y_rounded
                }
            }
        }
        
        ptz.ContinuousMove(safe_payload)
        
        # Update cache on success
        last_pan_speed = x_rounded
        last_tilt_speed = y_rounded
    except Exception as e:
        print(f"\n[PTZ ERROR] Failed to push motor speed vector: {e}")
        # Reset cache on error to force a retry next time
        last_pan_speed = None
        last_tilt_speed = None

def stop_camera(ptz, request):
    """Sends a zero velocity command to instantly halt the camera motors."""
    move_camera(ptz, request, 0.0, 0.0)

def set_light_raw(ptz_url, profile_token, command):
    """Sends raw auxiliary commands (like turning on IR lights) via SOAP."""
    if not ptz_url: return
    print(f" Sending RAW Auxiliary Command: {command} ... ", end="")
    
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
        response = requests.post(ptz_url, data=xml_payload, headers=headers, auth=HTTPDigestAuth(USER, PASS), timeout=2)
        if response.status_code == 200:
            print("OK!")
        else:
            print(f"Failed! (HTTP {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"Network Error! {e}")
        response = requests.post(ptz_url, data=xml_payload, headers=headers, auth=HTTPDigestAuth(USER, PASS), timeout=2)
        if response.status_code == 200:
            print("OK!")
        else:
            print(f"Failed! (HTTP {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"Network Error! {e}")