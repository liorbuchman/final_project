"""
hardware_bridge.py — connects the FastAPI dashboard to the live
ComrandInBattle running in Jetson_files/main_system.py.

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

# FSM state names main_system.py doesn't use verbatim. The live SystemState
# enum only has SCANNING/TRACKING/ENGAGED (no SLEWING - the new TRACKING state
# covers both "slewing to the acoustic DOA" and "visually scanning" at once);
# TRACKING/ENGAGED already match the frontend's vocabulary (SPEC.md §5) as-is,
# so only SCANNING needs a rename. SLEWING simply never occurs in hardware mode.
_STATE_NAME_MAP = {"SCANNING": "SEARCHING"}

_state_lock = threading.Lock()
_latest_frame: Optional[np.ndarray] = None
_frame_count = 0
_fps_window_start = time.time()
_latest_fps = 0.0
_latest_vision_confidence = 0.0
_latest_acoustic_confidence = 0.0

DroneSystem = None  # set by activate() on success; live ComrandInBattle instance
_config_module = None
_activated = False

# --- Rolling raw-audio buffer for the acoustic sample recorder ---
# Always holds the last AUDIO_RING_MAX_S seconds of the single channel the
# acoustic CNN itself classifies (config.AUDIO_CHANNEL), fed from the same
# raw_buffer process_audio_buffer() already receives - no new hook needed.
AUDIO_RING_MAX_S = 5.0
_audio_lock = threading.Lock()
_audio_ring: Optional[np.ndarray] = None


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
    never captures it. Wrap, don't touch the DSP/CNN logic itself. Also feeds
    the rolling audio ring buffer from the same raw_buffer, for the acoustic
    sample recorder - one extra read, no new hook into main_system.py."""
    original = acoustic_detector_cls.process_audio_buffer

    def wrapped(self, raw_buffer):
        score = original(self, raw_buffer)
        global _latest_acoustic_confidence
        with _state_lock:
            _latest_acoustic_confidence = float(score)
        _update_audio_ring(raw_buffer)
        return score

    acoustic_detector_cls.process_audio_buffer = wrapped


def _update_audio_ring(raw_buffer):
    """main_system.py's own ring_buffer (raw_buffer here) is a sliding window
    that only advances by step_samples each call - the newest audio is always
    its last step_samples rows. Append just that into our own longer,
    independent ring so a 1-5s clip can be sliced out later regardless of the
    model's own (fixed, shorter) inference window."""
    global _audio_ring
    if _config_module is None:
        return
    channel = getattr(_config_module, "AUDIO_CHANNEL", 0)
    sample_rate = getattr(_config_module, "SAMPLE_RATE", 16000)
    step_s = getattr(_config_module, "STEP_SECS", 0.2)
    step_samples = max(1, int(sample_rate * step_s))

    try:
        newest = raw_buffer[-step_samples:, channel].astype(np.int16)
    except Exception:
        return

    ring_len = int(AUDIO_RING_MAX_S * sample_rate)
    with _audio_lock:
        if _audio_ring is None or len(_audio_ring) != ring_len:
            _audio_ring = np.zeros(ring_len, dtype=np.int16)
        n = min(len(newest), ring_len)
        _audio_ring = np.roll(_audio_ring, -n)
        _audio_ring[-n:] = newest[-n:]


def get_audio_window(seconds: float) -> Optional[np.ndarray]:
    """Returns the last `seconds` (clamped to AUDIO_RING_MAX_S) of raw mono
    int16 audio - the same channel the acoustic CNN itself classifies. None
    if no audio has flowed through process_audio_buffer yet."""
    if _config_module is None or _audio_ring is None:
        return None
    sample_rate = getattr(_config_module, "SAMPLE_RATE", 16000)
    n = max(1, int(min(seconds, AUDIO_RING_MAX_S) * sample_rate))
    with _audio_lock:
        n = min(n, len(_audio_ring))
        return _audio_ring[-n:].copy()


def get_audio_sample_rate() -> int:
    if _config_module is None:
        return 16000
    return getattr(_config_module, "SAMPLE_RATE", 16000)


def _patch_vision_confidence(optical_detector_cls):
    """Taps the YOLO model's own track() call so run_inference's hysteresis
    (YOLO_LOW/HIGH_CONF_THRESHOLD lock-hold logic), box-drawing, and
    visual_lock all run completely untouched - this only reads the confidence
    already computed for the frame about to be returned. Applied after
    initialize_hardware's real ONVIF+YOLO setup finishes, since self.model
    doesn't exist before that.

    run_inference() calls self.model.track(frame, stream=False, ...), which
    returns a plain list of Results (not a generator like a bare model call
    would) - wrap that specific method, not __call__, and mirror exactly
    what run_inference itself reads (first box of the first result)."""
    original_init = optical_detector_cls.initialize_hardware

    def patched_init(self):
        original_init(self)
        if self.model is None:
            return
        original_track = self.model.track

        def wrapped_track(source, **kwargs):
            results = original_track(source, **kwargs)
            conf = 0.0
            if results and len(results[0].boxes) > 0:
                conf = float(results[0].boxes.conf[0].item())
            global _latest_vision_confidence
            with _state_lock:
                _latest_vision_confidence = conf
            return results

        self.model.track = wrapped_track

    optical_detector_cls.initialize_hardware = patched_init


def activate() -> bool:
    """Brings the real hardware stack online in this process. Returns True
    on success. Never raises - any failure (missing deps, not on a Jetson)
    leaves the caller free to fall back to the simulated pipelines."""
    global DroneSystem, _config_module, _activated
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
    DroneSystem = main_system.ComrandInBattle()
    threading.Thread(target= DroneSystem.initialize_system, daemon=True, name="DroneSystemThread").start()

    _activated = True
    logger.info("Hardware bridge activated - ComrandInBattle running live")
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
    if DroneSystem is None:
        return False
    with DroneSystem.data_lock:
        return DroneSystem.optical_hw_status == "ONLINE"


def get_mic_ok() -> bool:
    if DroneSystem is None:
        return False
    with DroneSystem.data_lock:
        return DroneSystem.acoustic_hw_status == "ONLINE"


def get_camera_pan_deg() -> float:
    """Live PTZ pan angle, mirrored from OpticalDetector's own open-loop
    position tracking (current_camera_pan) - drives the GUI's compass strip
    (camera bearing vs acoustic DOA). Not lock-protected in main_system.py
    itself (a plain float attribute), but a single read is GIL-safe, same as
    the other unprotected attributes this bridge already reads."""
    if DroneSystem is None:
        return 0.0
    try:
        return float(getattr(DroneSystem.video_processor, "current_camera_pan", 0.0))
    except Exception:
        return 0.0


def get_camera_pan_range():
    """(min_deg, max_deg) mechanical pan limits, read from Jetson_files/config.py
    (MIN_ANGLE/MAX_ANGLE) rather than hardcoded, so editing those constants and
    restarting the server is all that's needed to change what the GUI's
    compass strip displays - no frontend change required. Falls back to a
    generic +/-90 deg when running without the hardware config loaded."""
    if _config_module is None:
        return (-90.0, 90.0)
    lo = getattr(_config_module, "MIN_ANGLE", -90.0)
    hi = getattr(_config_module, "MAX_ANGLE", 90.0)
    return (float(lo), float(hi))


def get_fsm_state() -> str:
    """Mirrors the live ComrandInBattle FSM instead of running
    the dashboard's own simulated FSM.tick() - main_system.py owns the
    authoritative state. DEGRADED (a GUI-only concept) is derived from the
    hardware status flags, same semantics as the simulated path."""
    if DroneSystem is None:
        return "IDLE"
    with DroneSystem.data_lock:
        raw_state = DroneSystem.state.name
        cam_ok = DroneSystem.optical_hw_status == "ONLINE"
        mic_ok = DroneSystem.acoustic_hw_status == "ONLINE"
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

    def update_params(self, yolo_threshold_low: float, yolo_threshold_high: float, nms_threshold: float, draw_boxes: bool) -> None:
        # Only the two thresholds are simply supported today (they drive the
        # real hysteresis lock-hold in run_inference()); NMS and draw_boxes
        # have no live hook (the box is burned into the frame unconditionally
        # by the untouched run_inference()) and are intentionally ignored.
        if _config_module is not None:
            _config_module.YOLO_LOW_CONF_THRESHOLD = yolo_threshold_low
            _config_module.YOLO_HIGH_CONF_THRESHOLD = yolo_threshold_high

    def latest_result(self):
        if not get_camera_ok() or DroneSystem is None:
            return None
        with DroneSystem.data_lock:
            visual_lock = DroneSystem.visual_lock_acquired
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
        if not get_mic_ok() or DroneSystem is None:
            return None
        with DroneSystem.data_lock:
            doa = DroneSystem.acoustic_azimuth
        conf = round(_latest_acoustic_confidence, 3)
        return self._AudioResult(
            confidence=conf,
            doa_deg=round(doa, 1),
            db=None,
            waveform=[],
            spectrogram=np.zeros((SPEC_ROWS, SPEC_COLS), dtype=np.uint8),
            sensor_ts=time.time(),
        )
