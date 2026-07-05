#!/usr/bin/env python3
# main_system.py

import time
import threading
import cv2
import logging
import datetime
import os
import sys
from enum import Enum

# Resolve workspace paths for multi-modal submodule discovery
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config
from uav_acoustic.acoustic_processor import AcousticDetector
from uav_vision.optical_processor import OpticalDetector


class SystemState(Enum):
    SCANNING = 1
    SLEWING = 2
    TRACKING = 3
    ENGAGED = 4


class IntegratedDroneDefenseSystem:
    def __init__(self):
        self.state = SystemState.SCANNING
        self.running = True
        self.state_timestamp = time.time()

        # Thread synchronization primitive for cross-sensor data sharing
        self.data_lock = threading.Lock()

        # Shared telemetry variables (Thread-Safe)
        self.acoustic_triggered = False
        self.acoustic_azimuth = 0.0
        self.visual_lock_acquired = False

        # Self-health hardware monitoring status metrics
        self.acoustic_hw_status = "INITIALIZING"
        self.optical_hw_status = "INITIALIZING"

        # Instantiate localized modular sub-processors
        self.audio_processor = AcousticDetector()
        self.video_processor = OpticalDetector()

    def init_logging(self):
        """Initializes non-destructive timestamped logging with Ultralytics override protection."""
        optical_logs_dir = os.path.join(script_dir, "logs", "optical")
        os.makedirs(optical_logs_dir, exist_ok=True)

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"integrated_system_{current_time}.log"
        log_file_path = os.path.join(optical_logs_dir, log_file_name)

        # Force dynamic file handler to bypass any internal logging set by YOLO/Ultralytics
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)],
            force=True
        )
        logging.info("--- INTEGRATED TACTICAL DRONE DEFENSE GRID ONLINE ---")

    def start_defense_grid(self):
        """Initializes hardware contexts and spawns concurrent subsystem loops."""
        # 1. Initialize synchronous logging configurations
        self.init_logging()

        # 2. Safely attempt validation of sub-module asset layers
        try:
            self.audio_processor.load_model()
            if self.audio_processor.usb_dev is None:
                self.acoustic_hw_status = "UNAVAILABLE"
        except Exception as audio_err:
            logging.error(f"Acoustic core initial load failure: {audio_err}")
            self.acoustic_hw_status = "UNAVAILABLE"

        try:
            self.video_processor.initialize_hardware()
            if self.video_processor.ptz is None:
                self.optical_hw_status = "UNAVAILABLE"
        except Exception as video_err:
            logging.error(f"Optical ONVIF driver initial load failure: {video_err}")
            self.optical_hw_status = "UNAVAILABLE"

        # 3. Spawn independent thread for background acoustic array processing
        audio_thread = threading.Thread(target=self.acoustic_background_loop, daemon=True, name="AcousticThread")
        audio_thread.start()

        # 4. Spawn independent thread for the tactical FSM (Decoupled from video frame rate)
        fsm_thread = threading.Thread(target=self.tactical_fsm_loop, daemon=True, name="FSMThread")
        fsm_thread.start()

        # 5. Bind master loop directly into the main thread video channel
        self.optical_master_loop()

    def acoustic_background_loop(self):
        """Asynchronous execution pipe dedicated solely to live ReSpeaker hardware streaming."""
        import pyaudio
        import numpy as np

        logging.info("[Subsystem] Acoustic monitoring thread successfully deployed.")
        chunk_size = int(config.SAMPLE_RATE * config.WINDOW_SECS)
        p = pyaudio.PyAudio()
        stream = None
        
        # --- HARDWARE AUTO-DISCOVERY BLOCK (Fixed for native 6-channel XMOS stream) ---
        target_idx = None
        target_channels = 6 # Standard native channel width for ReSpeaker USB firmware (4 Mics + 2 Loopback)
        
        logging.info("[Acoustic HW] Scanning active system audio interfaces...")
        for i in range(p.get_device_count()):
            try:
                dev_info = p.get_device_info_by_index(i)
                dev_name = dev_info.get("name", "")
                max_inputs = dev_info.get("maxInputChannels", 0)
                
                # Check if this hardware node matches our ReSpeaker chipset
                if "ReSpeaker" in dev_name or "seeed" in dev_name.lower():
                    target_idx = i
                    # CRITICAL FIX: Mandate native hardware channel size (6) to satisfy ALSA constraints
                    target_channels = max_inputs if max_inputs > 0 else 6
                    logging.info(f"🎤 [FOUND] Device '{dev_name}' bound at Index {target_idx} (Native Channels: {max_inputs})")
                    break
            except Exception as scan_err:
                logging.debug(f"Skipping audio node index {i}: {scan_err}")

        if target_idx is None:
            logging.warning(f"⚠️ [Acoustic HW] ReSpeaker not found by name. Falling back to config default index ({config.RESPEAKER_INDEX}).")
            target_idx = config.RESPEAKER_INDEX

        # Guard initialization check before attempting to lock local USB soundcard descriptors
        if self.acoustic_hw_status != "UNAVAILABLE":
            try:
                # Open live streaming channel using the exact native hardware channel count
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=target_channels,
                    rate=config.SAMPLE_RATE,
                    input=True,
                    input_device_index=target_idx,
                    frames_per_buffer=chunk_size
                )
                with self.data_lock:
                    self.acoustic_hw_status = "ONLINE"
                logging.info(f"🎤 [AUDIO HOT] Direct connection bound to audio node at Index {target_idx} using {target_channels} channels.")
            except Exception as e:
                logging.error(f"❌ [Acoustic HW] PyAudio hardware streaming link rejected: {e}")
                with self.data_lock:
                    self.acoustic_hw_status = "UNAVAILABLE"

        while self.running:
            with self.data_lock:
                is_hw_active = (self.acoustic_hw_status == "ONLINE")

            if is_hw_active and stream is not None:
                try:
                    raw_bytes = stream.read(chunk_size, exception_on_overflow=False)
                    # Dynamic matrix reshape based on native target channel geometry
                    audio_chunk = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, target_channels)
                    
                    # Forward structured multi-channel array matrix directly to the CNN core pipeline
                    # acoustic_processor pulls index 0 ([:, 0]) smoothly without any index bounding problems
                    self.audio_processor.process_audio_buffer(audio_chunk)

                    with self.data_lock:
                        self.acoustic_triggered = self.audio_processor.is_triggered
                        self.acoustic_azimuth = self.audio_processor.current_azimuth
                except Exception as loop_err:
                    logging.error(f"[Acoustic Loop Fault] Telemetry acquisition failure: {loop_err}")
                    with self.data_lock:
                        self.acoustic_hw_status = "UNAVAILABLE"
                    time.sleep(config.STEP_SECS)
            else:
                # Safe Fallback Cadence: Maintain system loops without hardware starvation crashes
                with self.data_lock:
                    self.acoustic_triggered = False
                    self.acoustic_hw_status = "UNAVAILABLE"
                time.sleep(1.0)

        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()
        logging.info("[Subsystem] Live acoustic streaming interface safely terminated.")

        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()
        logging.info("[Subsystem] Live acoustic streaming interface safely terminated.")

    def tactical_fsm_loop(self):
        """Independent asynchronous loop dedicated entirely to tactical state management (10Hz)."""
        logging.info("[Subsystem] Tactical FSM thread deployed and decoupled from frame-rate.")
        while self.running:
            curr_time = time.time()

            # Thread-safe read snapshot of cross-sensor parameters
            with self.data_lock:
                audio_alert = self.acoustic_triggered
                target_azimuth = self.acoustic_azimuth
                video_lock = self.visual_lock_acquired
                cam_status = self.optical_hw_status

            # --- Finite State Machine State Evaluations ---

            if self.state == SystemState.SCANNING:
                if audio_alert:
                    logging.info(f"[FSM TRANSITION] SCANNING -> SLEWING. Vector: {target_azimuth}°")
                    self.state = SystemState.SLEWING
                    self.state_timestamp = curr_time
                    # Safe Driver Check: Only dispatch PTZ commands if camera layer is authenticated online
                    if cam_status == "ONLINE":
                        self.video_processor.track_target(target_azimuth, config.DEFAULT_ELEVATION_ANGLE)

            elif self.state == SystemState.SLEWING:
                if video_lock:
                    logging.info("[FSM TRANSITION] SLEWING -> ENGAGED. Dynamic visual intercept achieved.")
                    self.state = SystemState.ENGAGED
                elif curr_time - self.state_timestamp > 2.0:
                    logging.info("[FSM TRANSITION] SLEWING -> TRACKING. Transit complete. Arresting payloads.")
                    if cam_status == "ONLINE":
                        self.video_processor.track_target(0, 0)
                    self.state = SystemState.TRACKING
                    self.state_timestamp = curr_time

            elif self.state == SystemState.TRACKING:
                if video_lock:
                    logging.info("[FSM TRANSITION] TRACKING -> ENGAGED. YOLO target confirmation acquired.")
                    self.state = SystemState.ENGAGED
                    if cam_status == "ONLINE":
                        self.video_processor.trigger_deterrent(True)
                elif curr_time - self.state_timestamp > config.TARGET_LOST_TIMEOUT:
                    logging.info("[FSM TRANSITION] TRACKING -> SCANNING. Search window expired. Resetting grid.")
                    self.state = SystemState.SCANNING

            elif self.state == SystemState.ENGAGED:
                if not video_lock:
                    logging.info("[FSM TRANSITION] ENGAGED -> TRACKING. Target trace lost. Dropping payloads.")
                    self.state = SystemState.TRACKING
                    self.state_timestamp = curr_time
                    if cam_status == "ONLINE":
                        self.video_processor.trigger_deterrent(False)

            time.sleep(0.1)

    def optical_master_loop(self):
        """High-throughput GStreamer video capture loop running on native execution thread."""
        import numpy as np
        
        pipeline_string = config.get_gstreamer_pipeline()
        cap = cv2.VideoCapture(pipeline_string, cv2.CAP_GSTREAMER)

        with self.data_lock:
            is_cam_driver_ok = (self.optical_hw_status != "UNAVAILABLE")

        if not cap.isOpened() or not is_cam_driver_ok:
            logging.critical("Failed to instantiate hardware accelerated GStreamer pipeline or ONVIF node.")
            with self.data_lock:
                self.optical_hw_status = "UNAVAILABLE"
        else:
            with self.data_lock:
                self.optical_hw_status = "ONLINE"

        logging.info("[Subsystem] Master optical processing loop online.")

        while self.running:
            frame_fetched = False
            frame = None

            with self.data_lock:
                is_video_feed_active = (self.optical_hw_status == "ONLINE")

            # Try to grab frame matrix only if pipeline is confirmed healthy
            if is_video_feed_active and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret:
                        frame_fetched = True
                    else:
                        logging.warning("Hardware digital video stream link dropped frames unexpectedly.")
                        with self.data_lock:
                            self.optical_hw_status = "UNAVAILABLE"
                except Exception as stream_err:
                    logging.error(f"GStreamer low-level capture fault: {stream_err}")
                    with self.data_lock:
                        self.optical_hw_status = "UNAVAILABLE"

            if frame_fetched and frame is not None:
                # Spatial correction mapping for inverted physical camera positioning
                frame = cv2.flip(frame, -1)
                
                # Scale frame matrix to 640x480 BEFORE inference to resolve Domain Shift pixel soup
                inference_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

                # Compute convolutional YOLO inference matching current tactical requirements
                if self.state in [SystemState.SLEWING, SystemState.TRACKING, SystemState.ENGAGED]:
                    inference_frame = self.video_processor.run_inference(inference_frame)

                with self.data_lock:
                    self.visual_lock_acquired = self.video_processor.visual_lock
            else:
                # FALLBACK SAFE VIEWPORT: Create localized safe blank matrix canvas to prevent rendering freeze
                inference_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(inference_frame, "VIDEO SIGNAL: UNAVAILABLE", (80, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                with self.data_lock:
                    self.visual_lock_acquired = False

            # Thread-safe telemetry snapshot for tactical UI HUD rendering
            with self.data_lock:
                current_state_name = self.state.name
                mic_status = self.acoustic_hw_status
                cam_status = self.optical_hw_status

            # Dynamic Banner Generation: Shift background to red warning zone if assets are lost
            banner_color = (0, 0, 0) # Neutral Black
            if mic_status == "UNAVAILABLE" or cam_status == "UNAVAILABLE":
                banner_color = (0, 0, 160) # High-Visibility Tactical Red Warning

            # Construct Integrated UI Telemetry Overlay on the display frame buffer
            overlay_text = f"FSM: {current_state_name} | MIC: {mic_status} | CAM: {cam_status}"
            cv2.rectangle(inference_frame, (0, 0), (inference_frame.shape[1], 40), banner_color, -1)
            cv2.putText(inference_frame, overlay_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display structural frame matrix directly to monitor layout window
            cv2.imshow("Tactical Multi-Modal Interceptor Grid", inference_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("[Main] Termination sequence requested via keyboard interrupt.")
                self.running = False
                break

        # Graceful hardware destruction sequence
        logging.info("Arresting PTZ physical payloads and freeing context links.")
        with self.data_lock:
            can_shutdown_motors = (self.optical_hw_status == "ONLINE")
        if can_shutdown_motors:
            try:
                self.video_processor.track_target(0, 0)
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    defense_grid = IntegratedDroneDefenseSystem()
    defense_grid.start_defense_grid()