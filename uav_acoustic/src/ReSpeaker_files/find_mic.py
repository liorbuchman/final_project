import usb.core
import usb.util
import libusb_package
import time
from uav_acoustic.src.ReSpeaker_files.respeaker_usb_led import ReSpeakerV31Leds

if __name__ == "__main__":
    leds = ReSpeakerV31Leds()
    
    try:
        print("Testing: RED (Drone Detected Scenario)")
        leds.set_mono(255, 0, 0)
        time.sleep(1)
        
        print("Testing: BLUE (Scanning Scenario)")
        leds.set_mono(0, 0, 255)
        time.sleep(1)
        
        print("Testing: GREEN (System Ready)")
        leds.set_mono(0, 255, 0)
        time.sleep(1)
        
        print("Turning off...")
        leds.off()
        print("Test complete. The hardware is ready for the project.")
        
    except KeyboardInterrupt:
        leds.off()