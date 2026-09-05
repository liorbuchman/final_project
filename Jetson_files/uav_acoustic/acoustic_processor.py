import os
import logging
import time
import struct
import math
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import usb.core
import usb.util
import config
import libusb_package
import queue
import threading
from uav_acoustic.model import SmallCNN                  
from uav_acoustic.respeaker_usb_led import ReSpeakerV31Leds

try:
    from uav_acoustic.model import load_system
except ImportError:
    pass

logger = logging.getLogger("AcousticSystem")

class AcousticDetector:
    def __init__(self):
        self.model = None
        self.mean = None
        self.std = None
        self.is_triggered = False
        self.current_azimuth = 0.0
        self.last_valid_azimuth = 0.0
        self.lock_initialized = False # Tracks initial target acquisition state
        
        # Outlier rejection variables for abrupt angle jumps (Problem 2)
        self.outlier_streak_count = 0
        self.candidate_azimuth = 0.0

        # Debounce streaks for is_triggered - requires several consecutive
        # buffers on the same side of the threshold before flipping state, so
        # a single noisy CNN frame can't flicker is_triggered on/off.
        self.trigger_on_streak = 0
        self.trigger_off_streak = 0

        # Energy-gate hysteresis latch + last-event-end clock. The energy gate
        # used to hard-reset the event (and wipe the DOA filter) on a single
        # quiet buffer; now it feeds the OFF debounce streak like any other
        # "below threshold" observation, and the DOA smoothing history is only
        # dropped when a genuinely new event starts after a real gap.
        self._energy_gate_open = False
        self._last_event_end_time = 0.0

        # Exact DSP matching parameters from config (Single Source of Truth)
        self.sr = config.SAMPLE_RATE
        self.n_mels = config.MEL_N_MELS
        self.n_fft = config.MEL_N_FFT
        self.hop_length = config.MEL_HOP_LENGTH
        self.fixed_length = config.MEL_FIXED_LENGTH

        # Connect to the raw XMOS USB interface for hardware parameter tuning
        try:
            backend = libusb_package.get_libusb1_backend()
            self.usb_dev = usb.core.find(idVendor=config.IDVENDOR, idProduct=config.IDPRODUCT, backend=backend)
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
        except Exception as e:
            self.leds = None
            logger.error(f"Failed to initialize ReSpeaker LEDs: {e}")

        self.current_led_color = None
        self.led_queue = queue.Queue(maxsize=2)
        self.led_worker_running = True
        self.led_thread = threading.Thread(target=self._led_worker_loop, daemon=True, name="LED_Worker")
        self.led_thread.start()
        self.set_led_color(0x001100)#green

    def _led_worker_loop(self):
        """Threaded loop to asynchronously update LED colors without blocking the main audio processing."""
        while self.led_worker_running:
            try:
                color = self.led_queue.get(timeout=0.5)
                if self.leds:
                    self.leds.mono(color)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"[LED Worker] USB communication error: {e}")

    def set_led_color(self, color_hex):
        """Updates the LED color only if it has changed, and queues the update for asynchronous processing."""
        if self.current_led_color != color_hex:
            if self.led_queue.full():
                try:
                    self.led_queue.get_nowait() 
                except queue.Empty:
                    pass
            self.led_queue.put(color_hex)
            self.current_led_color = color_hex

    def load_model(self):
        print("[Acoustic] Loading neural network checkpoints and weights...")
        try:
            device = config.DEVICE
            self.model = SmallCNN(n_classes=2).to(device)
            model_path = os.path.join(config.AUDIO_MODEL_DIR, "best_model.pt")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found exactly at: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
        Queries the ReSpeaker XVF-3000 DOAANGLE parameter (ID 21) via USB control transfer.
        wValue 0xC0 = read flag (0x80) | int type (0x40) | offset 0.
        Response: 8 bytes = [int32 value][int32 metadata].
        """
        if self.usb_dev is None:
            return self.last_valid_azimuth
        try:
            res = self.usb_dev.ctrl_transfer(0xC0, 0, 0xC0, 21, 8, 500)
            if len(res) >= 8:
                hardware_angle, _ = struct.unpack('ii', res.tobytes())
                return float(hardware_angle)
        except Exception as e:
            logger.error(f"USB Control Transfer failed to read hardware DOA register: {e}")
        return self.last_valid_azimuth

    def process_audio_buffer(self, raw_buffer):
        """Processes audio data loop, handles CNN execution, and updates hardware telemetry."""
        ch = getattr(config, 'AUDIO_CHANNEL', 0)
        y_chunk = raw_buffer[:, ch].astype(np.float32) / 32768.0

        on_confirm = getattr(config, 'AUDIO_TRIGGER_ON_CONFIRM_COUNT', 2)
        off_confirm = getattr(config, 'AUDIO_TRIGGER_OFF_CONFIRM_COUNT', 4)

        # Step 1: Energy gate - now hysteretic, and a "below threshold" buffer is
        # treated exactly like a low classification score: it feeds the OFF
        # debounce streak instead of instantly killing the event. A single quiet
        # buffer between rotor-noise peaks no longer ends the event or wipes the
        # DOA smoothing history (which caused EVENT START/END to churn ~1-2s and
        # defeated apply_azimuth_smoothing entirely).
        rms_energy = float(np.sqrt(np.mean(y_chunk ** 2)))
        energy_ok = True
        if getattr(config, 'ENABLE_ENERGY_GATE', True):
            gate_on = getattr(config, 'AUDIO_MIN_RMS_THRESHOLD', 0.025)
            gate_off = min(getattr(config, 'AUDIO_RMS_GATE_OFF_THRESHOLD', gate_on), gate_on)
            energy_ok = (rms_energy >= gate_off) if self._energy_gate_open else (rms_energy >= gate_on)
            self._energy_gate_open = energy_ok

        if not energy_ok:
            self.trigger_off_streak += 1
            self.trigger_on_streak = 0
            if self.is_triggered and self.trigger_off_streak >= off_confirm:
                logger.info(f"EVENT END - Energy below threshold (RMS: {rms_energy:.4f})")
                self.is_triggered = False
                self._last_event_end_time = time.time()
            if not self.is_triggered and self.leds:
                self.set_led_color(0x001100) # Soft Green
            return 0.0

        # Normalize chunk for CNN matching
        y_norm = y_chunk.copy()
        max_amp = np.max(np.abs(y_norm))
        if max_amp > 1e-8:
            y_norm /= max_amp
            
        mel_spec = self.compute_live_logmel(y_norm)
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

        # Step 3: Threshold and System lock pipeline management
        # Debounced: a streak of several consecutive buffers on the same side
        # of the threshold is required before is_triggered actually flips, so
        # a single noisy CNN frame can't flicker the FSM's audio_alert signal
        # (see EVENT START/END logs previously firing under a second apart).
        if prediction_score > config.AUDIO_CLASSIFICATION_THRESHOLD:
            self.trigger_on_streak += 1
            self.trigger_off_streak = 0

            if not self.is_triggered and self.trigger_on_streak >= on_confirm:
                logger.warning(f"EVENT START - Target detected. Initial Confidence: {prediction_score:.4f}")
                # Only forget the DOA smoothing history when this is a genuinely
                # new target - i.e. the previous event ended more than
                # DOA_FILTER_RESET_GAP_SECS ago. A quick re-trigger after a brief
                # dropout keeps last_valid_azimuth so the EMA + outlier filter
                # stay continuous across the blip instead of taking a raw
                # (possibly front/back-mirrored) hardware angle at face value.
                gap = time.time() - self._last_event_end_time
                if gap > getattr(config, 'DOA_FILTER_RESET_GAP_SECS', 3.0):
                    self.lock_initialized = False
                self.is_triggered = True
                if self.leds:
                    self.set_led_color(0xFF0000) # Bright Red

            if self.is_triggered:
                self.current_azimuth = self.apply_azimuth_smoothing(calculated_azimuth)
                logger.warning(f"DRONE HARDWARE TRACKING - Conf: {prediction_score:.4f} | Hardware Azimuth: {self.current_azimuth:.1f}°")
        else:
            self.trigger_off_streak += 1
            self.trigger_on_streak = 0

            if self.is_triggered and self.trigger_off_streak >= off_confirm:
                logger.info("EVENT END - Target tracking lost or cleared.")
                self.is_triggered = False
                self._last_event_end_time = time.time()
                if self.leds:
                    self.set_led_color(0x001100) # Soft Green

        return prediction_score

    def apply_azimuth_smoothing(self, raw_azimuth):
        """
        Filters tracking anomalies using shortest angular distance tracking logic 
        combined with an outlier consensus filter to prevent jumping to 180° opposite noise.
        """
        if not self.lock_initialized:
            self.last_valid_azimuth = raw_azimuth
            self.lock_initialized = True
            self.outlier_streak_count = 0
            return raw_azimuth
            
        # Compute dynamic short-arc angular difference: range [-180, +180]
        angular_diff = (raw_azimuth - self.last_valid_azimuth + 180.0) % 360.0 - 180.0
        
        max_jump = getattr(config, 'DOA_MAX_JUMP_DEG', 45.0)
        confirm_count = getattr(config, 'DOA_OUTLIER_CONFIRM_COUNT', 3)
        alpha = getattr(config, 'DOA_SMOOTHING_ALPHA', 0.35)

        # Reject abrupt outliers (e.g., sudden 180° room reflections)
        if abs(angular_diff) > max_jump:
            if self.outlier_streak_count == 0:
                self.candidate_azimuth = raw_azimuth
                self.outlier_streak_count = 1
            else:
                cand_diff = abs((raw_azimuth - self.candidate_azimuth + 180.0) % 360.0 - 180.0)
                if cand_diff < max_jump:
                    self.outlier_streak_count += 1
                else:
                    self.candidate_azimuth = raw_azimuth
                    self.outlier_streak_count = 1

            if self.outlier_streak_count >= confirm_count:
                self.last_valid_azimuth = self.candidate_azimuth
                self.outlier_streak_count = 0
                return self.last_valid_azimuth

            return self.last_valid_azimuth

        self.outlier_streak_count = 0
        smoothed_azimuth = self.last_valid_azimuth + alpha * angular_diff
        self.last_valid_azimuth = (smoothed_azimuth + 360.0) % 360.0
        return self.last_valid_azimuth