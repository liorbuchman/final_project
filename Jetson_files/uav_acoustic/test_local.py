#!/usr/bin/env python3
"""
=============================================================================
TACTICAL UAV DEFENSE SYSTEM - STANDALONE ACOUSTIC TEST BENCH & DIAGNOSTICS
=============================================================================
Features:
  1. Universal DOA Tracking: Tracks angle on ANY active sound (claps, voice, drone).
  2. Parallel CNN Sanity Check: Verifies high confidence on drones vs low on voice/claps.
  3. Microsecond-level Latency Profiler across all acoustic pipeline stages.
  4. Real-Time Raw vs. Smoothed Circular DOA Telemetry with Compass.
  5. Interactive Live WAV Recorder (Press 'R' or 'Space' to toggle recording).
=============================================================================
"""

import os
import sys
import time
import math
import wave
import struct
import select
import datetime
import threading
import numpy as np
import pyaudio
import torch

# Resolve workspace mapping to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import config
from uav_acoustic.acoustic_processor import AcousticDetector

# --- Terminal Colors for Rich Diagnostics ---
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'

class AcousticDiagnosticBench:
    def __init__(self):
        print(f"{Colors.BOLD}{Colors.CYAN}===================================================={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}   ACOUSTIC SENSOR BENCHMARK & HARDWARE DIAGNOSTICS {Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}===================================================={Colors.RESET}")
        print(f"Target Compute Device: {Colors.BOLD}{config.DEVICE}{Colors.RESET}")
        
        self.detector = AcousticDetector()
        self.detector.load_model()
        
        # Audio Stream Configuration from config (Single Source of Truth)
        self.sample_rate = config.SAMPLE_RATE
        self.window_secs = getattr(config, 'WINDOW_SECS', 1.0)
        self.step_secs = getattr(config, 'STEP_SECS', 0.2)
        self.window_samples = int(self.sample_rate * self.window_secs)
        self.step_samples = int(self.sample_rate * self.step_secs)
        
        # Recording State Management
        self.is_recording = False
        self.recorded_frames = []
        self.record_lock = threading.Lock()
        self.record_dir = os.path.join(parent_dir, "recordings")
        os.makedirs(self.record_dir, exist_ok=True)

        self.running = True

    def find_respeaker_device(self, p):
        """Scans system audio cards for ReSpeaker array index."""
        target_idx = None
        target_channels = 6
        
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                name = info.get("name", "")
                channels = info.get("maxInputChannels", 0)
                if "ReSpeaker" in name or "seeed" in name.lower():
                    target_idx = i
                    target_channels = channels if channels > 0 else 6
                    print(f"Found Hardware: {Colors.GREEN}{name}{Colors.RESET} (Device Index: {i}, Channels: {target_channels})")
                    return target_idx, target_channels
            except Exception:
                pass
                
        print(f"{Colors.YELLOW}ReSpeaker descriptor not explicitly named. Falling back to index {config.RESPEAKER_INDEX}.{Colors.RESET}")
        return config.RESPEAKER_INDEX, target_channels

    def get_compass_direction(self, angle_deg):
        """Translates 0-360 degree azimuth to cardinal compass string."""
        compass_points = [
            "North (0°)", "North-East (45°)", "East (90°)", "South-East (135°)",
            "South (180°)", "South-West (225°)", "West (270°)", "North-West (315°)"
        ]
        idx = int((angle_deg + 22.5) / 45.0) % 8
        return compass_points[idx]

    def check_keyboard_input(self):
        """Non-blocking terminal input listener."""
        if sys.platform != "win32":
            dr, _, _ = select.select([sys.stdin], [], [], 0.0)
            if dr:
                return sys.stdin.read(1).lower()
        return None

    def toggle_recording(self, channels):
        """Starts or stops saving audio frames to a timestamped WAV file."""
        with self.record_lock:
            if not self.is_recording:
                self.is_recording = True
                self.recorded_frames = []
                print(f"\n{Colors.BG_RED}{Colors.BOLD} [REC START] Recording Live Audio... Press 'R' or 'Space' again to save {Colors.RESET}\n")
            else:
                self.is_recording = False
                if len(self.recorded_frames) > 0:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_name = f"acoustic_record_{ts}.wav"
                    file_path = os.path.join(self.record_dir, file_name)
                    
                    wf = wave.open(file_path, 'wb')
                    wf.setnchannels(channels)
                    wf.setsampwidth(2) # 16-bit PCM
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(self.recorded_frames))
                    wf.close()
                    print(f"\n{Colors.BG_GREEN}{Colors.BOLD} [REC SAVED] Audio file written to: {file_path} {Colors.RESET}\n")
                self.recorded_frames = []

    def run_benchmark(self):
        p = pyaudio.PyAudio()
        dev_idx, total_channels = self.find_respeaker_device(p)

        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=total_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=self.step_samples
            )
        except Exception as e:
            print(f"{Colors.RED}Fatal Error opening PyAudio stream: {e}{Colors.RESET}")
            p.terminate()
            return

        # Sliding ring buffer for 1.0s window
        ring_buffer = np.zeros((self.window_samples, total_channels), dtype=np.int16)
        
        print("\n--- Diagnostic Controls ---")
        print(" [R] / [Space] : Start / Stop WAV Audio Recording (Saved to /recordings)")
        print(" [Q] / [Ctrl+C]: Quit Diagnostic Bench")
        print("----------------------------\n")

        use_tty = False
        old_settings = None
        if sys.platform != "win32" and sys.stdin.isatty():
            try:
                import tty, termios
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                use_tty = True
            except Exception:
                pass

        try:
            while self.running:
                key = self.check_keyboard_input()
                if key == 'q':
                    break
                elif key in ['r', ' ']:
                    self.toggle_recording(total_channels)

                # =============================================================
                # Stage 1: Audio I/O Read (Sliding Step Chunk)
                # =============================================================
                t0 = time.perf_counter()
                raw_bytes = stream.read(self.step_samples, exception_on_overflow=False)
                t_read = (time.perf_counter() - t0) * 1000.0 # ms

                if self.is_recording:
                    with self.record_lock:
                        self.recorded_frames.append(raw_bytes)

                new_chunk = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, total_channels)
                ring_buffer = np.roll(ring_buffer, -len(new_chunk), axis=0)
                ring_buffer[-len(new_chunk):, :] = new_chunk

                # Channel selection from config & float normalization
                ch = getattr(config, 'AUDIO_CHANNEL', 0)
                y_window = ring_buffer[:, ch].astype(np.float32) / 32768.0

                # =============================================================
                # Stage 2: RMS Energy Gating
                # =============================================================
                t1 = time.perf_counter()
                rms_energy = float(np.sqrt(np.mean(y_window ** 2)))
                peak_amp = float(np.max(np.abs(y_window)))
                energy_threshold = getattr(config, 'AUDIO_MIN_RMS_THRESHOLD', 0.015)
                is_gated = (rms_energy < energy_threshold) and getattr(config, 'ENABLE_ENERGY_GATE', True)
                t_rms = (time.perf_counter() - t1) * 1000.0 # ms

                # =============================================================
                # Stage 3: Feature Extraction & CNN Inference
                # =============================================================
                t_mel = 0.0
                t_cnn = 0.0
                conf_score = 0.0

                if not is_gated:
                    t2 = time.perf_counter()
                    y_norm = y_window.copy()
                    max_amp = np.max(np.abs(y_norm))
                    if max_amp > 1e-8:
                        y_norm /= max_amp
                    
                    mel_spec = self.detector.compute_live_logmel(y_norm)
                    x_tensor = torch.from_numpy(mel_spec).float()
                    x_tensor = (x_tensor - self.detector.mean) / self.detector.std
                    x_tensor = x_tensor.unsqueeze(0).unsqueeze(0).to(config.DEVICE)
                    t_mel = (time.perf_counter() - t2) * 1000.0

                    t3 = time.perf_counter()
                    with torch.no_grad():
                        logits = self.detector.model(x_tensor)
                        probs = torch.softmax(logits, dim=1)
                        conf_score = float(probs[0, 1].item())
                    t_cnn = (time.perf_counter() - t3) * 1000.0

                # =============================================================
                # Stage 4: Universal DOA Diagnostics (Active on ANY sound event)
                # =============================================================
                t4 = time.perf_counter()
                raw_doa = self.detector.read_hardware_doa_angle()
                
                if not is_gated:
                    smooth_doa = self.detector.apply_azimuth_smoothing(raw_doa)
                else:
                    smooth_doa = self.detector.last_valid_azimuth
                    
                t_doa = (time.perf_counter() - t4) * 1000.0

                t_total = (time.perf_counter() - t0) * 1000.0
                effective_hz = 1000.0 / t_total if t_total > 0 else 0.0

                # =============================================================
                # Stage 5: Terminal Dashboard Presentation
                # =============================================================
                # RMS Bar Visualizer
                bar_len = 18
                filled_rms = min(bar_len, int((rms_energy / max(energy_threshold * 2.5, 0.05)) * bar_len))
                rms_bar = f"[{'#' * filled_rms}{'.' * (bar_len - filled_rms)}]"

                # Visual Event & Classification Tag
                if is_gated:
                    event_tag = f"{Colors.CYAN}[SILENT / NOISE FLOOR]{Colors.RESET}"
                    conf_display = f"{Colors.CYAN}0.00% (GATED BYPASS){Colors.RESET}"
                    doa_status = f"{Colors.CYAN}[IDLE]{Colors.RESET}"
                else:
                    doa_status = f"{Colors.GREEN}[TRACKING]{Colors.RESET}"
                    if conf_score >= config.AUDIO_CLASSIFICATION_THRESHOLD:
                        event_tag = f"{Colors.BG_RED}{Colors.BOLD} [TARGET: DRONE] {Colors.RESET}"
                        conf_display = f"{Colors.RED}{Colors.BOLD}{conf_score*100:5.1f}% (ALERT){Colors.RESET}"
                    else:
                        event_tag = f"{Colors.YELLOW}[SOUND / VOICE / CLAP]{Colors.RESET}"
                        conf_display = f"{Colors.YELLOW}{conf_score*100:5.1f}% (BACKGROUND){Colors.RESET}"

                rec_badge = f"{Colors.RED}{Colors.BOLD}[● REC]{Colors.RESET}" if self.is_recording else f"{Colors.BOLD}[IDLE]{Colors.RESET}"
                compass_str = self.get_compass_direction(smooth_doa)

                # Render dynamic HUD line
                sys.stdout.write(
                    f"\r{rec_badge} "
                    f"RMS: {rms_energy:.4f} (Peak: {peak_amp:.3f}) {rms_bar} {event_tag} | "
                    f"CNN: {conf_display}\n"
                    f"   DOA {doa_status}: Raw {raw_doa:5.1f}° -> {Colors.BOLD}Smooth {smooth_doa:5.1f}°{Colors.RESET} ({compass_str:<16}) | "
                    f"Latency: [I/O: {t_read:3.0f}ms | RMS: {t_rms:2.1f}ms | Mel: {t_mel:3.1f}ms | CNN: {t_cnn:2.1f}ms | DOA: {t_doa:2.1f}ms] "
                    f"Total: {Colors.BOLD}{t_total:4.0f}ms ({effective_hz:4.1f}Hz){Colors.RESET}\033[F"
                )
                sys.stdout.flush()

        except KeyboardInterrupt:
            pass
        finally:
            if use_tty and old_settings:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            print("\n\nShutting down diagnostic bench...")
            if self.is_recording:
                self.toggle_recording(total_channels)
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    bench = AcousticDiagnosticBench()
    bench.run_benchmark()