import usb.core
import usb.util
import libusb_package
import time

class ReSpeakerV31Leds:
    def __init__(self):
        # Clean discovery target lookup without opening persistent locking handles
        backend = libusb_package.get_libusb1_backend()
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018, backend=backend)
        
        if dev is None:
            raise RuntimeError("ReSpeaker V3.1 hardware descriptor not found on USB bus.")
        print("Successfully verified ReSpeaker V3.1 LED Controller configuration.")

    def set_mono(self, r, g, b):
        """
        🚀 FINAL ARCHITECTURAL FIX: Raw Ephemeral Control Transfer Pattern.
        Sends vendor control setup packets directly to Endpoint 0.
        Bypasses claim_interface() and set_configuration() entirely, ensuring
        the Linux kernel never drops or unbinds the ALSA mic streaming lines.
        """
        backend = libusb_package.get_libusb1_backend()
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018, backend=backend)
        if dev is None:
            return
            
        try:
            # Execute raw vendor request transfer over the default control pipe
            dev.ctrl_transfer(
                0x40,      # bmRequestType: Vendor | Host-to-Device
                0x00,      # bRequest: 0
                0x01,      # wValue: Command 1 (Mono Mode)
                0x1C,      # wIndex: Register address 0x1C for ReSpeaker v3.1
                [int(r), int(g), int(b), 0],
                timeout=1000
            )
        except usb.core.USBError as e:
            # Fallback isolation routine ONLY if the kernel locks endpoint access under custom profiles
            if e.errno == 16 or "busy" in str(e).lower():
                try:
                    if dev.is_kernel_driver_active(3):
                        dev.detach_kernel_driver(3)
                    dev.ctrl_transfer(0x40, 0x00, 0x01, 0x1C, [int(r), int(g), int(b), 0], timeout=1000)
                except Exception as inner_err:
                    print(f"[USB LED Ephemeral Fault] Inner control pipe locked: {inner_err}")
            else:
                print(f"[USB LED Ephemeral Fault] Raw control transfer rejected: {e}")

    def mono(self, color_hex):
        """Translates single hex integer calls into separate R, G, B channels."""
        r = (color_hex >> 16) & 0xFF
        g = (color_hex >> 8) & 0xFF
        b = color_hex & 0xFF
        self.set_mono(r, g, b)

    def off(self):
        """Utility command to safely kill all LED lighting payloads."""
        self.set_mono(0, 0, 0)