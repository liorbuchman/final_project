import cv2
import os
import sys
import time
import logging
import requests
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

logger = logging.getLogger("OpticalSystem.PTZ")

# Fetch credentials from the configuration
IP = getattr(config, 'CAMERA_IP', '192.168.1.90')
USER = getattr(config, 'CAMERA_USER', 'admin')
PASS = getattr(config, 'CAMERA_PASS', 'admin')
ONVIF_PORT = 8899

# State caching to prevent network spamming
last_pan_speed = None
last_tilt_speed = None
last_send_time = 0.0

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
    """Sends a continuous movement vector (Pan, Tilt) to the camera using a failsafe dictionary payload.

    Also guards the underlying HTTP call two ways:
    - PTZ_MIN_SEND_INTERVAL caps how often a *changed* velocity can trigger a
      real HTTP request, so PD-loop jitter in execute_visual_closed_loop
      (which can otherwise produce a distinct value ~10x/sec) can't hammer
      the camera's embedded HTTP server into falling over. Stop commands
      (0, 0) bypass this - halting the motors is never delayed.
    - PTZ_KEEPALIVE_INTERVAL periodically resends an *unchanged*, non-zero
      velocity. Many budget ONVIF PTZ cameras silently auto-stop
      ContinuousMove a few seconds after the last command if it isn't
      refreshed (no Timeout element below guarantees they won't); a long
      open-loop pan phase in the acoustic search otherwise sends one command
      and then does not re-push it for up to ~20s, so without this the camera
      can sit physically still while the software keeps dead-reckoning
      current_camera_pan/current_tilt as if it were still moving.
    """
    global last_pan_speed, last_tilt_speed, last_send_time
    if ptz is None or request is None: return

    # Round to avoid micro-jitter sending
    x_rounded = round(float(x), 2)
    y_rounded = round(float(y), 2)

    now = time.time()
    elapsed = now - last_send_time
    velocity_changed = (x_rounded != last_pan_speed or y_rounded != last_tilt_speed)
    is_moving = (x_rounded != 0.0 or y_rounded != 0.0)

    if velocity_changed:
        is_stop = not is_moving
        if not is_stop and elapsed < config.PTZ_MIN_SEND_INTERVAL:
            return
    else:
        if not (is_moving and elapsed >= config.PTZ_KEEPALIVE_INTERVAL):
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
            },
            'Timeout': 'PT30S',
        }

        call_start = time.time()
        ptz.ContinuousMove(safe_payload)
        call_elapsed = time.time() - call_start
        if call_elapsed > config.PTZ_HTTP_SLOW_THRESHOLD:
            logger.warning(f"[PTZ] ContinuousMove HTTP call took {call_elapsed:.2f}s (camera may be overloaded/unresponsive)")

        # Update cache on success
        last_pan_speed = x_rounded
        last_tilt_speed = y_rounded
        last_send_time = now
    except Exception as e:
        logger.error(f"[PTZ] Failed to push motor speed vector: {e}")
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
        try:
            response = requests.post(ptz_url, data=xml_payload, headers=headers, auth=HTTPDigestAuth(USER, PASS), timeout=2)
            if response.status_code == 200: print("OK (on retry)!")
            else: print(f"Failed on retry! (HTTP {response.status_code})")
        except requests.exceptions.RequestException as retry_e:
            print(f"Total Network Failure: {retry_e}")