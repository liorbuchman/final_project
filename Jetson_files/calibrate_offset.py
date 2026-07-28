# calibrate_offset.py

import time
import struct
import usb.core
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config
from uav_vision.camera_AC import setup_camera

def get_acoustic_doa(dev):
    if dev is None:
        return 0.0
    try:
        res = dev.ctrl_transfer(0xC0, 0, 0xC0, 21, 8, 500)
        if len(res) >= 4:
            return float(struct.unpack(b'ii', res)[0])
    except Exception:
        pass
    return 0.0

def main():
    print("---  Hardware Calibration Tool: Acoustic to Optical Offset ---")

    print("Connecting to ReSpeaker DSP...")
    usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if usb_dev is None:
        print("ReSpeaker not found! Please check USB connection.")
        return
    print("ReSpeaker connected.")

    print("Connecting to ONVIF Camera...")
    ptz, req, token, url = setup_camera()
    if ptz is None:
        print("Camera connection failed!")
        return

    print("\n" + "="*50)
    print("CALIBRATION INSTRUCTIONS:")
    print("1. Play a continuous sound (e.g., from your phone) near the system.")
    print("2. Manually move the camera (via Web UI or joystick) to point EXACTLY at the sound source.")
    print("3. Look at the 'Suggested Offset' below.")
    print("4. Press Ctrl+C to exit, and copy that number into config.py")
    print("="*50 + "\n")

    try:
        while True:
            doa = get_acoustic_doa(usb_dev)

            try:
                status = ptz.GetStatus({'ProfileToken': token})
                cam_pan_raw = status.Position.PanTilt.x * 170.0
                cam_pan_360 = cam_pan_raw % 360
            except Exception:
                cam_pan_360 = 0.0

            offset = (cam_pan_360 - doa) % 360

            print(f" Mic Angle: {doa:6.1f}° |  Cam Angle: {cam_pan_360:6.1f}° |   Suggested CAMERA_MOUNT_OFFSET = {offset:6.1f}°", end='\r')
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n Calibration stopped. Update your config.py with the suggested offset.")

if __name__ == "__main__":
    main()