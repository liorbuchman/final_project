import os
import logging
import time
import struct
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import usb.core
import usb.util
import config
from uav_acoustic.model import SmallCNN                  
from uav_acoustic.respeaker_usb_led import ReSpeakerV31Leds

try:
    from uav_acoustic.model import load_system
except ImportError:
    pass

logger = logging.getLogger("AcousticSystem")
logger.setLevel(logging.INFO)
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/acoustic_system.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
logger.addHandler(file_handler)

class AcousticDetector:
    def __init__(self):
        self.model_dir = config.AUDIO_MODEL_DIR
        self.model = None
        self.mean = None
        self.std = None
        self.is_triggered = False
        self.current_azimuth = 0.0
        self.current_elevation = 0.0
        self.last_valid_azimuth = 0.0
        self.lock_initialized = False # Tracks initial target acquisition state
        
        # Exact DSP matching parameters
        self.sr = 16000
        self.n_mels = 128
        self.n_fft = 1024
        self.hop_length = 256
        self.fixed_length = 63

        # Connect to the raw XMOS USB interface for hardware parameter tuning
        try:
            self.usb_dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if self.usb_dev:
                logger.info("Successfully connected to ReSpeaker XMOS USB hardware tuning core.")
            else:
                logger.error("ReSpeaker USB hardware core mismatch or interface not found.")
        except Exception as e:
            self.usb_dev = None
            logger.error(f"Failed to bind pyusb interface: {e}")

        # Connect to the integrated LED controller
        try:
            self.leds = ReSpeakerV31Leds()
            self.leds.mono(0x001100) # Soft Green
        except Exception as e:
            self.leds = None
            logger.error(f"Failed to initialize ReSpeaker LEDs: {e}")

    def load_model(self):
        print("[Acoustic] Loading neural network checkpoints and weights...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SmallCNN(n_classes=2).to(device)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "best_model.pt")
            
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.mean = checkpoint["mean"]
            self.std = checkpoint["std"]
            self.model.eval() 
            print("[Acoustic] Model and normalization matrices loaded successfully.")
        except Exception as e:
            print(f"[Acoustic ERROR] Failed to load model: {e}")
            raise e

    def compute_live_logmel(self, y):
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, 
                                           hop_length=self.hop_length, n_mels=self.n_mels, power=2.0)
        spec_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
        if spec_db.shape[1] < self.fixed_length:
            spec_db = np.pad(spec_db, ((0, 0), (0, self.fixed_length - spec_db.shape[1])))
        else:
            spec_db = spec_db[:, :self.fixed_length]
        return spec_db

    def read_hardware_doa_angle(self):
        """
        Queries the ReSpeaker XMOS hardware register directly via USB control transfer 
        to fetch the built-in calculated DOAANGLE (Parameter ID 21).
        """
        if self.usb_dev is None:
            return 0.0
        try:
            # Control Transfer Specs for reading integer parameter 21 (DOAANGLE):
            # bmRequestType = 0xC0 (Vendor Class Incoming Device Read)
            # bRequest = 0
            # wValue = 0xC0 (Command payload construction indicator for int)
            # wIndex = 21 (DOAANGLE register identifier index)
            # length = 8 bytes (Returns value and type payload)
            res = self.usb_dev.ctrl_transfer(0xC0, 0, 0xC0, 21, 8, 500)
            if len(res) >= 4:
                # Unpack the native 4-byte little-endian signed integer
                hardware_angle = float(struct.unpack(b'ii', res)[0])
                return hardware_angle
        except Exception as e:
            logger.error(f"USB Control Transfer failed to read hardware DOA register: {e}")
        return 0.0

    def process_audio_buffer(self, raw_buffer):
        """Processes audio data loop, handles CNN execution, and updates hardware telemetry."""
        # Step 1: Execute CNN Classification
        y_chunk = raw_buffer[:, 0].astype(np.float32)
        max_amp = np.max(np.abs(y_chunk))
        if max_amp > 1e-8:
            y_chunk /= max_amp
            
        mel_spec = self.compute_live_logmel(y_chunk)
        x_tensor = torch.from_numpy(mel_spec).float()
        x_tensor = (x_tensor - self.mean) / self.std
        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)
        
        device = next(self.model.parameters()).device
        x_tensor = x_tensor.to(device)
        
        with torch.no_grad():
            logits = self.model(x_tensor)
            probabilities = F.softmax(logits, dim=1)
            prediction_score = float(probabilities[0, 1].item())

        # Step 2: Extract real-time hardware tracking telemetry from chip
        calculated_azimuth = self.read_hardware_doa_angle()
        self.current_elevation = 0.0 # XMOS integrated chipset operates in 2D plane tracking only

        # Step 3: Threshold and System lock pipeline management
        if prediction_score > config.AUDIO_CLASSIFICATION_THRESHOLD:
            if not self.is_triggered:
                logger.warning(f"EVENT START - Target detected. Initial Confidence: {prediction_score:.4f}")
                self.lock_initialized = False # Force lock reset to capture new position on event start
                
            self.is_triggered = True
            self.current_azimuth = self.apply_azimuth_smoothing(calculated_azimuth)
            
            logger.warning(f"DRONE HARDWARE TRACKING - Conf: {prediction_score:.4f} | Hardware Azimuth: {self.current_azimuth:.1f}°")
            if self.leds:
                self.leds.mono(0xFF0000) # Bright Red
        else:
            if self.is_triggered:
                logger.info("EVENT END - Target tracking lost or cleared.")
            self.is_triggered = False
            self.lock_initialized = False
            if self.leds:
                self.leds.mono(0x001100) # Soft Green
                
        return prediction_score

    def apply_azimuth_smoothing(self, raw_azimuth):
        """Filters tracking anomalies using shortest angular distance tracking logic."""
        if not self.lock_initialized:
            self.last_valid_azimuth = raw_azimuth
            self.lock_initialized = True
            return raw_azimuth
            
        # Compute dynamic angular wrap-around difference
        angular_diff = (raw_azimuth - self.last_valid_azimuth + 180) % 360 - 180
        if abs(angular_diff) > 45.0:
            return self.last_valid_azimuth
            
        self.last_valid_azimuth = raw_azimuth
        return raw_azimuth