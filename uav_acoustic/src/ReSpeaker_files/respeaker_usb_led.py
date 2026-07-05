import usb.core
import usb.util
import libusb_package
import time

class ReSpeakerV31Leds:
    def __init__(self):
        # setup libusb backend
        self.backend = libusb_package.get_libusb1_backend()
        self.dev = usb.core.find(idVendor=0x2886, idProduct=0x0018, backend=self.backend)
        
        if self.dev is None:
            raise RuntimeError("ReSpeaker V3.1 not found. Check USB connection.")
        
        # Windows/Linux setup
        self.dev.set_configuration()
        print("Successfully connected to ReSpeaker V3.1 LED Controller.")

    def set_mono(self, r, g, b):
        """Set all LEDs to a single color."""
        try:
            self.dev.ctrl_transfer(
                0x40,      # bmRequestType: Vendor | Host-to-Device
                0x00,      # bRequest: 0
                0x01,      # wValue: Command 1 (Mono)
                0x1C,      # wIndex: 0x1C for ReSpeaker V3.1
                [int(r), int(g), int(b), 0],
                timeout=1000
            )
        except usb.core.USBError as e:
            print(f"Failed to set LEDs: {e}")

    def off(self):
        self.set_mono(0, 0, 0)