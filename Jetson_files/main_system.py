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
    TRACKING = 2
    ENGAGED = 3


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

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)],
            force=True
        )
        logging.info("--- INTEGRATED TACTICAL DRONE DEFENSE GRID ONLINE ---")

    def start_defense_grid(self):
        """Initializes hardware contexts and spawns concurrent subsystem loops."""
        self.init_logging()

        try:
            self.audio_processor.load_model()
            if getattr(self.audio_processor, 'usb_dev', None) is None:
                self.acoustic_hw_status = "UNAVAILABLE"
        except Exception as audio_err:
            logging.error(f"Acoustic core initial load failure: {audio_err}")
            self.acoustic_hw_status = "UNAVAILABLE"

        try:
            self.video_processor.initialize_hardware()
            if getattr(self.video_processor, 'ptz', None) is None:
                self.optical_hw_status = "UNAVAILABLE"
        except Exception as video_err:
            logging.error(f"Optical ONVIF driver initial load failure: {video_err}")
            self.optical_hw_status = "UNAVAILABLE"

        # Spawn independent background threads
        audio_thread = threading.Thread(target=self.acoustic_background_loop, daemon=True, name="AcousticThread")
        audio_thread.start()

        fsm_thread = threading.Thread(target=self.tactical_fsm_loop, daemon=True, name="FSMThread")
        fsm_thread.start()

        # Run optical pipeline on the main thread
        self.optical_master_loop()

    def acoustic_background_loop(self):
        """Asynchronous execution pipe dedicated solely to live ReSpeaker hardware streaming."""
        import pyaudio
        import numpy as np

        logging.info("[Subsystem] Acoustic monitoring thread successfully deployed.")
        chunk_size = int(config.SAMPLE_RATE * config.WINDOW_SECS)
        p = pyaudio.PyAudio()
        stream = None
        
        target_idx = None
        target_channels = 6
        
        logging.info("[Acoustic HW] Scanning active system audio interfaces...")
        for i in range(p.get_device_count()):
            try:
                dev_info = p.get_device_info_by_index(i)
                dev_name = dev_info.get("name", "")
                max_inputs = dev_info.get("maxInputChannels", 0)
                
                if "ReSpeaker" in dev_name or "seeed" in dev_name.lower():
                    target_idx = i
                    target_channels = max_inputs if max_inputs > 0 else 6
                    logging.info(f"🎤 [FOUND] Device '{dev_name}' bound at Index {target_idx} (Native Channels: {max_inputs})")
                    break
            except Exception as scan_err:
                logging.debug(f"Skipping audio node index {i}: {scan_err}")

        if target_idx is None:
            logging.warning(f"⚠️ [Acoustic HW] ReSpeaker not found by name. Falling back to default index ({config.RESPEAKER_INDEX}).")
            target_idx = config.RESPEAKER_INDEX

        while self.running:
            # Attempt reconnection if stream is inactive
            if stream is None and self.acoustic_hw_status != "UNAVAILABLE":
                try:
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
                    logging.info(f"[AUDIO HOT] Connection bound to index {target_idx}.")
                except Exception as e:
                    logging.error(f"[Acoustic HW] Stream connection failed: {e}")
                    with self.data_lock:
                        self.acoustic_hw_status = "UNAVAILABLE"

            with self.data_lock:
                is_hw_active = (self.acoustic_hw_status == "ONLINE")

            if is_hw_active and stream is not None:
                try:
                    raw_bytes = stream.read(chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, target_channels)
                    self.audio_processor.process_audio_buffer(audio_chunk)

                    with self.data_lock:
                        self.acoustic_triggered = self.audio_processor.is_triggered
                        self.acoustic_azimuth = self.audio_processor.current_azimuth
                except Exception as loop_err:
                    logging.error(f"[Acoustic Loop Fault] Hardware read error: {loop_err}")
                    if stream:
                        try:
                            stream.close()
                        except Exception:
                            pass
                        stream = None
                    with self.data_lock:
                        self.acoustic_hw_status = "UNAVAILABLE"
                    time.sleep(config.STEP_SECS)
            else:
                with self.data_lock:
                    self.acoustic_triggered = False
                time.sleep(1.0)

        # Cleanup block (Single, clean termination)
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
        logging.info("[Subsystem] Tactical FSM thread deployed.")
        last_seen_visual_time = time.time()

        while self.running:
            curr_time = time.time()

            with self.data_lock:
                audio_alert = self.acoustic_triggered
                target_azimuth = self.acoustic_azimuth
                video_lock = self.visual_lock_acquired
                cam_status = self.optical_hw_status
                current_state = self.state

            # --- Finite State Machine ---
            if current_state == SystemState.SCANNING:
                if audio_alert:
                    logging.info(f"[FSM TRANSITION] SCANNING -> TRACKING. Acoustic vector: {target_azimuth}°")
                    with self.data_lock:
                        self.state = SystemState.TRACKING
                        self.state_timestamp = curr_time

            elif current_state == SystemState.TRACKING:
                if video_lock:
                    logging.info("[FSM TRANSITION] TRACKING -> ENGAGED. YOLO visual lock confirmed.")
                    with self.data_lock:
                        self.state = SystemState.ENGAGED
                    last_seen_visual_time = curr_time
                    if cam_status == "ONLINE":
                        self.video_processor.track_target(0, 0)
                elif audio_alert:
                    if cam_status == "ONLINE" and hasattr(self.video_processor, 'handle_acoustic_search'):
                        self.video_processor.handle_acoustic_search(target_azimuth)
                    with self.data_lock:
                        self.state_timestamp = curr_time 
                elif curr_time - self.state_timestamp > config.TARGET_LOST_TIMEOUT:
                    logging.info("[FSM TRANSITION] TRACKING -> SCANNING. Search window expired.")
                    if cam_status == "ONLINE":
                        self.video_processor.track_target(0, 0)
                    with self.data_lock:
                        self.state = SystemState.SCANNING

            elif current_state == SystemState.ENGAGED:
                if video_lock:
                    last_seen_visual_time = curr_time
                    if cam_status == "ONLINE":
                        self.video_processor.execute_visual_closed_loop()
                else:
                    time_since_last_sight = curr_time - last_seen_visual_time
                    if time_since_last_sight > config.VISUAL_LOCK_COOLDOWN:
                        logging.info(f"[FSM TRANSITION] ENGAGED -> TRACKING. Target lost for {time_since_last_sight:.1f}s.")
                        with self.data_lock:
                            self.state = SystemState.TRACKING
                            self.state_timestamp = curr_time
                        if cam_status == "ONLINE":
                            self.video_processor.track_target(0, 0)

            time.sleep(0.1)

    def optical_master_loop(self):
        """High-throughput video capture loop running on native execution thread."""
        import numpy as np
        
        pipeline_string = config.get_gstreamer_pipeline()
        cap = cv2.VideoCapture(pipeline_string, cv2.CAP_GSTREAMER)

        with self.data_lock:
            is_cam_driver_ok = (self.optical_hw_status != "UNAVAILABLE")

        if not cap.isOpened() or not is_cam_driver_ok:
            logging.critical("Failed to instantiate GStreamer pipeline or ONVIF node.")
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
                current_state = self.state

            if is_video_feed_active and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret:
                        frame_fetched = True
                    else:
                        logging.warning("Video stream dropped frames.")
                        with self.data_lock:
                            self.optical_hw_status = "UNAVAILABLE"
                except Exception as stream_err:
                    logging.error(f"GStreamer low-level capture fault: {stream_err}")
                    with self.data_lock:
                        self.optical_hw_status = "UNAVAILABLE"

            if frame_fetched and frame is not None:
                frame = cv2.flip(frame, -1)
                inference_frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

                if current_state in [SystemState.TRACKING, SystemState.ENGAGED]:
                    inference_frame = self.video_processor.run_inference(inference_frame)

                with self.data_lock:
                    self.visual_lock_acquired = self.video_processor.visual_lock
            else:
                inference_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(inference_frame, "VIDEO SIGNAL: UNAVAILABLE", (80, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                with self.data_lock:
                    self.visual_lock_acquired = False

            with self.data_lock:
                current_state_name = self.state.name
                mic_status = self.acoustic_hw_status
                cam_status = self.optical_hw_status

            banner_color = (0, 0, 0)
            if mic_status == "UNAVAILABLE" or cam_status == "UNAVAILABLE":
                banner_color = (0, 0, 160)

            overlay_text = f"FSM: {current_state_name} | MIC: {mic_status} | CAM: {cam_status}"
            cv2.rectangle(inference_frame, (0, 0), (inference_frame.shape[1], 40), banner_color, -1)
            cv2.putText(inference_frame, overlay_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Safe GUI display catch for headless environments
            try:
                cv2.imshow("Tactical Multi-Modal Interceptor Grid", inference_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("[Main] Termination requested via keyboard interrupt.")
                    self.running = False
                    break
            except cv2.error as gui_err:
                # Log once or skip GUI rendering if no display is available
                pass

        logging.info("Arresting PTZ physical payloads and freeing context links.")
        with self.data_lock:
            can_shutdown_motors = (self.optical_hw_status == "ONLINE")
        if can_shutdown_motors:
            try:
                self.video_processor.track_target(0, 0)
            except Exception:
                pass
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    defense_grid = IntegratedDroneDefenseSystem()
    defense_grid.start_defense_grid()