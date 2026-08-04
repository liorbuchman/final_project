"""
hardware_bridge.py — connects the FastAPI dashboard to the live
IntegratedDroneDefenseSystem running in Jetson_files/main_system.py.

Zero edits to anything under Jetson_files/. Every hook is a runtime
monkey-patch applied from this module only, each one *wrapping* the
original method (never replacing its logic) so the real algorithms,
ONVIF motor control, and GStreamer pipeline run exactly as written:

  - cv2.imshow / cv2.waitKey are intercepted so the already-rendered frame
    (YOLO box + status banner already burned in by the untouched code) is
    captured instead of shown on a display that doesn't exist on a headless
    Jetson.
  - AcousticDetector.process_audio_buffer is wrapped to capture the
    confidence score it already computes and returns (main_system.py
    itself discards that return value).
  - OpticalDetector.initialize_hardware is wrapped to, after the real ONVIF
    + YOLO setup finishes untouched, tap the YOLO model's own call so the
    per-frame confidence it already computes is captured too.

Activated only when main.py calls activate() (gated behind the
HARDWARE_MODE=1 environment variable). If the hardware stack isn't
importable (missing deps, not running on a Jetson), activation fails
safely and the caller stays on the simulated pipelines.

HardwareVisionPipeline / HardwareAudioPipeline implement the exact same
structural protocol as SimulatedVisionPipeline/ReplayVisionPipeline etc.
in main.py, so runtime.vision/runtime.audio can point at them with zero
changes to telemetry_loop, video_encode_loop, or the WebSocket handlers.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("dashboard")

JETSON_FILES_DIR = Path(__file__).resolve().parent.parent / "Jetson_files"

SPEC_ROWS = 64
SPEC_COLS = 48

# FSM state names main_system.py doesn't use verbatim - only the rename the
# frontend already understands (SPEC.md §5); SLEWING/TRACKING/ENGAGED match as-is.
_STATE_NAME_MAP = {"SCANNING": "SEARCHING"}

_state_lock = threading.Lock()
_latest_frame: Optional[np.ndarray] = None
_frame_count = 0
_fps_window_start = time.time()
_latest_fps = 0.0
_latest_vision_confidence = 0.0
_latest_acoustic_confidence = 0.0

defense_grid = None  # set by activate() on success; live IntegratedDroneDefenseSystem instance
_config_module = None
_activated = False


def _patch_video_capture():
    """optical_master_loop's cv2.imshow call is the only place the fully
    rendered frame (box + banner already drawn by untouched code) exists.
    There's no display on a headless Jetson to show it on anyway."""

    def _patched_imshow(_window_name, frame):
        global _latest_frame, _frame_count
        with _state_lock:
            _latest_frame = frame
            _frame_count += 1

    cv2.imshow = _patched_imshow
    cv2.waitKey = lambda *_a, **_kw: -1
    cv2.destroyAllWindows = lambda *_a, **_kw: None


def _patch_acoustic_confidence(acoustic_detector_cls):
    """process_audio_buffer already computes and returns a live confidence
    score every audio chunk; main_system.py's acoustic_background_loop just
    never captures it. Wrap, don't touch the DSP/CNN logic itself."""
    original = acoustic_detector_cls.process_audio_buffer

    def wrapped(self, raw_buffer):
        score = original(self, raw_buffer)
        global _latest_acoustic_confidence
        with _state_lock:
            _latest_acoustic_confidence = float(score)
        return score

    acoustic_detector_cls.process_audio_buffer = wrapped


def _patch_vision_confidence(optical_detector_cls):
    """Taps the YOLO model's own call so run_inference's thresholding,
    box-drawing, and visual_lock logic all run completely untouched - this
    only reads the confidence already computed for the frame about to be
    returned. Applied after initialize_hardware's real ONVIF+YOLO setup
    finishes, since self.model doesn't exist before that."""
    original_init = optical_detector_cls.initialize_hardware

    def patched_init(self):
        original_init(self)
        if self.model is None:
            return
        original_call = self.model.__call__

        def wrapped_call(frame, **kwargs):
            for r in original_call(frame, **kwargs):
                conf = float(r.boxes.conf.max().item()) if len(r.boxes) > 0 else 0.0
                global _latest_vision_confidence
                with _state_lock:
                    _latest_vision_confidence = conf
                yield r

        self.model.__call__ = wrapped_call

    optical_detector_cls.initialize_hardware = patched_init


def activate() -> bool:
    """Brings the real hardware stack online in this process. Returns True
    on success. Never raises - any failure (missing deps, not on a Jetson)
    leaves the caller free to fall back to the simulated pipelines."""
    global defense_grid, _config_module, _activated
    if _activated:
        return True

    if not JETSON_FILES_DIR.is_dir():
        logger.warning("HARDWARE_MODE requested but %s not found - staying simulated", JETSON_FILES_DIR)
        return False

    if str(JETSON_FILES_DIR) not in sys.path:
        sys.path.insert(0, str(JETSON_FILES_DIR))

    try:
        import main_system
        import config as jetson_config
        from uav_acoustic.acoustic_processor import AcousticDetector
        from uav_vision.optical_processor import OpticalDetector
    except Exception as exc:
        logger.error("Hardware stack unavailable (%s) - staying simulated", exc)
        return False

    _patch_video_capture()
    _patch_acoustic_confidence(AcousticDetector)
    _patch_vision_confidence(OpticalDetector)

    _config_module = jetson_config
    defense_grid = main_system.IntegratedDroneDefenseSystem()
    threading.Thread(target=defense_grid.start_defense_grid, daemon=True, name="DefenseGridThread").start()

    _activated = True
    logger.info("Hardware bridge activated - IntegratedDroneDefenseSystem running live")
    return True


def get_vision_fps() -> float:
    global _frame_count, _fps_window_start, _latest_fps
    with _state_lock:
        now = time.time()
        elapsed = now - _fps_window_start
        if elapsed >= 1.0:
            _latest_fps = round(_frame_count / elapsed, 1)
            _frame_count = 0
            _fps_window_start = now
        return _latest_fps


def get_camera_ok() -> bool:
    if defense_grid is None:
        return False
    with defense_grid.data_lock:
        return defense_grid.optical_hw_status == "ONLINE"


def get_mic_ok() -> bool:
    if defense_grid is None:
        return False
    with defense_grid.data_lock:
        return defense_grid.acoustic_hw_status == "ONLINE"


def get_fsm_state() -> str:
    """Mirrors the live IntegratedDroneDefenseSystem FSM instead of running
    the dashboard's own simulated FSM.tick() - main_system.py owns the
    authoritative state. DEGRADED (a GUI-only concept) is derived from the
    hardware status flags, same semantics as the simulated path."""
    if defense_grid is None:
        return "IDLE"
    with defense_grid.data_lock:
        raw_state = defense_grid.state.name
        cam_ok = defense_grid.optical_hw_status == "ONLINE"
        mic_ok = defense_grid.acoustic_hw_status == "ONLINE"
    if not cam_ok or not mic_ok:
        return "DEGRADED"
    return _STATE_NAME_MAP.get(raw_state, raw_state)


class HardwareVisionPipeline:
    """Live-hardware implementation of main.py's VisionPipeline protocol.
    start()/stop() are deliberate no-ops: the real capture/inference loop
    runs continuously from server startup, independent of the GUI's
    START/STOP switch (that switch only matters for replay mode)."""

    def __init__(self, vision_result_cls, vision_detection_cls):
        self._VisionResult = vision_result_cls
        self._VisionDetection = vision_detection_cls

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def connected(self) -> bool:
        return get_camera_ok()

    @property
    def fps(self) -> float:
        return get_vision_fps()

    def update_params(self, yolo_threshold: float, nms_threshold: float, draw_boxes: bool) -> None:
        # Only the threshold is simply supported today; NMS and draw_boxes
        # have no live hook (the box is burned into the frame unconditionally
        # by the untouched run_inference()) and are intentionally ignored.
        if _config_module is not None:
            _config_module.YOLO_CONF_THRESHOLD = yolo_threshold

    def latest_result(self):
        if not get_camera_ok() or defense_grid is None:
            return None
        with defense_grid.data_lock:
            visual_lock = defense_grid.visual_lock_acquired
        conf = round(_latest_vision_confidence, 3)
        dets = [self._VisionDetection(cls="drone", conf=conf, bbox=[0, 0, 0, 0], azimuth_deg=0.0)] if visual_lock else []
        return self._VisionResult(confidence=conf, detections=dets, sensor_ts=time.time())

    def latest_frame(self) -> Optional[bytes]:
        with _state_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return buf.tobytes() if ok else None


class HardwareAudioPipeline:
    """Live-hardware implementation of main.py's AudioPipeline protocol.
    start()/stop() are no-ops for the same reason as HardwareVisionPipeline."""

    def __init__(self, audio_result_cls):
        self._AudioResult = audio_result_cls

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def active(self) -> bool:
        return get_mic_ok()

    def update_params(self, audio_threshold: float, audio_gain: float) -> None:
        # Only the threshold is simply supported today; audio_gain has no
        # live hook in AcousticDetector and is intentionally ignored.
        if _config_module is not None:
            _config_module.AUDIO_CLASSIFICATION_THRESHOLD = audio_threshold

    def latest_result(self):
        if not get_mic_ok() or defense_grid is None:
            return None
        with defense_grid.data_lock:
            doa = defense_grid.acoustic_azimuth
        conf = round(_latest_acoustic_confidence, 3)
        return self._AudioResult(
            confidence=conf,
            doa_deg=round(doa, 1),
            db=None,
            waveform=[],
            spectrogram=np.zeros((SPEC_ROWS, SPEC_COLS), dtype=np.uint8),
            sensor_ts=time.time(),
        )
