#!/usr/bin/env python3
import time
import threading
import cv2
import logging
import datetime
import os
import sys
import pyaudio
import numpy as np
from enum import Enum

# Resolve workspace paths for multi-modal submodule discovery
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import config
from uav_acoustic.acoustic_processor import AcousticDetector
from uav_vision.optical_processor import OpticalDetector

class SystemState(Enum):
    SCANNING = 1    # Idle/Acoustic Search
    TRACKING = 2    # Moving camera to acoustic DOA / Scanning visually
    ENGAGED = 3     # Visual lock acquired, YOLO tracking active

class ComrandInBattle:
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

        # Self-health hardware monitoring
        self.acoustic_hw_status = "INITIALIZING"
        self.optical_hw_status = "INITIALIZING"

        # Instantiate localized modular sub-processors
        self.audio_processor = AcousticDetector()
        self.video_processor = OpticalDetector()

    def init_logging(self):
        """Initializes non-destructive timestamped logging for the integrated system and all subsystems."""
        # Base logs directory and subsystem subdirectories
        logs_base_dir = os.path.join(script_dir, "logs")
        system_logs_dir = os.path.join(logs_base_dir, "system")
        optical_logs_dir = os.path.join(logs_base_dir, "optical")
        acoustic_logs_dir = os.path.join(logs_base_dir, "acoustic")

        # Create all log directories if they do not exist
        os.makedirs(system_logs_dir, exist_ok=True)
        os.makedirs(optical_logs_dir, exist_ok=True)
        os.makedirs(acoustic_logs_dir, exist_ok=True)

        # Unique timestamped log filename for the current session
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"FullSystem_{current_time}.log"
        log_file_path = os.path.join(system_logs_dir, log_file_name)

        # Configure root logger to capture logs from all subsystems and stdout
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        logging.info("--- INTEGRATED TACTICAL DRONE DEFENSE GRID ONLINE ---")
        
    def initialize_system(self):
        """Initializes hardware contexts and spawns concurrent subsystem threads."""
        self.init_logging()

        # Load acoustic model
        try:
            self.audio_processor.load_model()
            if getattr(self.audio_processor, 'usb_dev', None) is None:
                self.acoustic_hw_status = "UNAVAILABLE"
        except Exception as audio_err:
            logging.error(f"Acoustic core initial load failure: {audio_err}")
            self.acoustic_hw_status = "UNAVAILABLE"

        # Initialize ONVIF camera
        try:
            self.video_processor.initialize_hardware()
            self.video_processor.auto_calibrate_and_home()
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

        # Run optical pipeline (Video streaming & YOLO) on the main thread
        self.optical_master_loop()

    def acoustic_background_loop(self):
        """Continuous audio capture and processing loop running on a dedicated thread."""
        window_samples = int(config.SAMPLE_RATE * getattr(config, 'WINDOW_SECS', 1.0))
        step_samples = int(config.SAMPLE_RATE * getattr(config, 'STEP_SECS', 0.2))
        target_channels = 6
        target_idx = None

        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            try:
                dev_info = p.get_device_info_by_index(i)
                dev_name = dev_info.get("name", "")
                max_inputs = dev_info.get("maxInputChannels", 0)
                if "ReSpeaker" in dev_name or "seeed" in dev_name.lower():
                    target_idx = i
                    target_channels = max_inputs if max_inputs > 0 else 6
                    break
            except Exception:
                pass

        if target_idx is None:
            target_idx = config.RESPEAKER_INDEX

        ring_buffer = np.zeros((window_samples, target_channels), dtype=np.int16)
        stream = None

        while self.running:
            if stream is None and self.acoustic_hw_status != "UNAVAILABLE":
                try:
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=target_channels,
                        rate=config.SAMPLE_RATE,
                        input=True,
                        input_device_index=target_idx,
                        frames_per_buffer=step_samples
                    )
                    with self.data_lock:
                        self.acoustic_hw_status = "ONLINE"
                except Exception as e:
                    with self.data_lock:
                        self.acoustic_hw_status = "UNAVAILABLE"
                    time.sleep(1.0)
                    continue

            with self.data_lock:
                is_hw_active = (self.acoustic_hw_status in ["ONLINE", "WARNING_NO_USB"])

            if is_hw_active and stream is not None:
                try:
                    raw_bytes = stream.read(step_samples, exception_on_overflow=False)
                    new_chunk = np.frombuffer(raw_bytes, dtype=np.int16).reshape(-1, target_channels)

                    ring_buffer = np.roll(ring_buffer, -len(new_chunk), axis=0)
                    ring_buffer[-len(new_chunk):, :] = new_chunk

                    self.audio_processor.process_audio_buffer(ring_buffer)

                    with self.data_lock:
                        self.acoustic_triggered = self.audio_processor.is_triggered
                        self.acoustic_azimuth = self.audio_processor.current_azimuth

                except Exception as stream_err:
                    logging.warning(f"Acoustic stream read error: {stream_err}")
                    if stream:
                        try: stream.close()
                        except: pass
                        stream = None
                    time.sleep(getattr(config, 'STEP_SECS', 0.2))
            else:
                with self.data_lock:
                    self.acoustic_triggered = False
                time.sleep(0.5)

        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()

    def tactical_fsm_loop(self):
        """Independent asynchronous loop managing system states at 10Hz."""
        last_seen_visual_time = time.time()

        while self.running:
            curr_time = time.time()

            # Read current status from all sensors safely
            with self.data_lock:
                audio_alert = self.acoustic_triggered
                target_azimuth = self.acoustic_azimuth
                video_lock = self.visual_lock_acquired
                cam_status = self.optical_hw_status
                current_state = self.state

            # --- Finite State Machine Logic ---
            if current_state == SystemState.SCANNING:
                if audio_alert:
                    logging.info(f"[FSM] Drone detected acoustically at {target_azimuth}°. Transitioning to TRACKING.")
                    with self.data_lock:
                        self.state = SystemState.TRACKING
                        self.state_timestamp = curr_time

            elif current_state == SystemState.TRACKING:
                # 1. Target found visually -> ENGAGE
                if video_lock:
                    logging.info("[FSM] Target visually locked! Transitioning to ENGAGED.")
                    with self.data_lock:
                        self.state = SystemState.ENGAGED
                    last_seen_visual_time = curr_time
                    if cam_status == "ONLINE":
                        self.video_processor.track_target(0, 0) # Stop scan motors
                        
                # 2. Moving camera to search for target
                elif audio_alert:
                    if cam_status == "ONLINE" and hasattr(self.video_processor, 'handle_acoustic_search'):
                        # This function handles pan and tilt sweep. Blocks until done or drone is found.
                        self.video_processor.handle_acoustic_search(target_azimuth)
                    # Reset timeout only AFTER movement/scan is complete
                    with self.data_lock:
                        self.state_timestamp = time.time() 
                        
                # 3. Timeout - Target lost, revert to scan
                elif curr_time - self.state_timestamp > config.TARGET_LOST_TIMEOUT:
                    logging.info("[FSM] Search window expired. Returning to SCANNING.")
                    if cam_status == "ONLINE":
                        self.video_processor.track_target(0, 0)
                    with self.data_lock:
                        self.state = SystemState.SCANNING

            elif current_state == SystemState.ENGAGED:
                # Continue visual tracking
                if video_lock:
                    last_seen_visual_time = curr_time
                    if cam_status == "ONLINE":
                        self.video_processor.execute_visual_closed_loop()
                # Target lost briefly
                else:
                    time_since_last_sight = curr_time - last_seen_visual_time
                    if time_since_last_sight > config.VISUAL_LOCK_COOLDOWN:
                        logging.info(f"[FSM] Visual target lost for {time_since_last_sight:.1f}s. Reverting to TRACKING.")
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
            with self.data_lock:
                self.optical_hw_status = "UNAVAILABLE"
        else:
            with self.data_lock:
                self.optical_hw_status = "ONLINE"

        while self.running:
            frame_fetched = False
            frame = None

            with self.data_lock:
                is_video_feed_active = (self.optical_hw_status == "ONLINE")
                current_state = self.state

            # Capture frame
            if is_video_feed_active and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret:
                        frame_fetched = True
                    else:
                        with self.data_lock:
                            self.optical_hw_status = "UNAVAILABLE"
                except Exception:
                    with self.data_lock:
                        self.optical_hw_status = "UNAVAILABLE"

            # Process frame
            if frame_fetched and frame is not None:
                frame = cv2.flip(frame, -1)
                inference_frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

                # Only run heavy YOLO inference if we are searching or tracking
                if current_state in [SystemState.TRACKING, SystemState.ENGAGED]:
                    inference_frame = self.video_processor.run_inference(inference_frame)

                with self.data_lock:
                    self.visual_lock_acquired = self.video_processor.visual_lock
            else:
                # Black screen fallback
                inference_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(inference_frame, "VIDEO SIGNAL: UNAVAILABLE", (80, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                with self.data_lock:
                    self.visual_lock_acquired = False

            # Draw UI Overlay
            with self.data_lock:
                current_state_name = self.state.name
                mic_status = self.acoustic_hw_status
                cam_status = self.optical_hw_status

            banner_color = (0, 0, 0)
            if mic_status == "UNAVAILABLE" or cam_status == "UNAVAILABLE":
                banner_color = (0, 0, 160) # Red banner for hardware faults

            overlay_text = f"FSM: {current_state_name} | MIC: {mic_status} | CAM: {cam_status}"
            cv2.rectangle(inference_frame, (0, 0), (inference_frame.shape[1], 40), banner_color, -1)
            cv2.putText(inference_frame, overlay_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display video
            try:
                cv2.imshow("Tactical Multi-Modal Interceptor Grid", inference_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            except cv2.error:
                pass

        # Cleanup on exit
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
    DroneSystem = ComrandInBattle()
    
    try:
        DroneSystem.initialize_system()
    except KeyboardInterrupt:
        logging.warning("[System] Manual interruption (Ctrl+C) received. Shutting down...")
        DroneSystem.running = False  
        
        if DroneSystem.optical_hw_status == "ONLINE":
            try:
                DroneSystem.video_processor.track_target(0.0, 0.0)
                logging.info("[System] PTZ motors stopped securely.")
            except Exception as e:
                logging.error(f"[System] Failed to stop PTZ motors during shutdown: {e}")
                
    finally:
        logging.info("--- INTEGRATED TACTICAL DRONE DEFENSE GRID OFFLINE ---")
        logging.shutdown()