"""
FastAPI backend for the Jetson multi-modal drone-detection dashboard.

Phase 1: simulated Vision/Audio pipelines behind fixed protocols (SPEC.md §7),
so the real TensorRT YOLO / Audio CNN implementations can be swapped in later
with zero changes to this server or to templates/index.html.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import random
import time
import wave
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

import hardware_bridge

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    from jtop import jtop  # type: ignore
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False
    # start_jtop() silently no-ops when this is False - log it once here so a
    # missing GPU%/temp reading in the GUI is traceable to "jtop isn't
    # installed" instead of looking like a silent runtime failure. Fix:
    # `pip install jetson-stats` in this venv, and make sure the jtop.service
    # systemd daemon is running on the Jetson (`sudo systemctl status jtop`).
    logging.getLogger("dashboard").warning(
        "jtop package not installed - GPU%/temp will show N/A. Install with "
        "'pip install jetson-stats' and ensure jtop.service is running."
    )

# Single persistent jtop session for the whole process lifetime (opened once in
# on_startup, closed once in on_shutdown) - see start_jtop()/jetson_metrics()
# below for why this must never be re-opened per poll.
_jtop_handle = None

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
INDEX_HTML_PATH = BASE_DIR / "templates" / "index.html"
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("dashboard")
logger.setLevel(logging.DEBUG)


# --------------------------------------------------------------------------
# Settings (Component E) — validated, clamped, persisted to config.json
# --------------------------------------------------------------------------

SETTINGS_BOUNDS = {
    "yolo_threshold_low": (0.0, 1.0),
    "yolo_threshold_high": (0.0, 1.0),
    "audio_threshold": (0.0, 1.0),
    "fusion_window_ms": (100, 2000),
    "audio_gain": (1.0, 10.0),
    "nms_threshold": (0.1, 0.9),
    "slew_timeout_s": (1.0, 5.0),
    "lost_lock_timeout_s": (1.0, 10.0),
    "audio_buffer_size": (128, 1024),
}
DEFAULT_SETTINGS = {
    "system_active": False,
    "yolo_threshold_low": 0.25,
    "yolo_threshold_high": 0.5,
    "audio_threshold": 0.70,
    "fusion_window_ms": 500,
    "audio_gain": 2.5,
    "nms_threshold": 0.45,
    "slew_timeout_s": 2.0,
    "lost_lock_timeout_s": 4.0,
    "invert_camera_vector": False,
    "audio_buffer_size": 512,
    "draw_boxes": True,
}


def clamp_settings(values: dict) -> dict:
    out = dict(values)
    for key, (lo, hi) in SETTINGS_BOUNDS.items():
        if key in out:
            try:
                out[key] = max(lo, min(hi, type(DEFAULT_SETTINGS[key])(out[key])))
            except (TypeError, ValueError):
                out.pop(key, None)
    for flag in ("system_active", "invert_camera_vector", "draw_boxes"):
        if flag in out:
            out[flag] = bool(out[flag])
    return out


class SettingsStore:
    def __init__(self, path: Path):
        self.path = path
        self.values = dict(DEFAULT_SETTINGS)
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                on_disk = json.loads(self.path.read_text())
                merged = dict(DEFAULT_SETTINGS)
                merged.update(clamp_settings(on_disk))
                self.values = merged
                return
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("config.json unreadable (%s), using defaults", exc)
        self.save()

    def save(self):
        try:
            self.path.write_text(json.dumps(self.values, indent=2))
        except OSError as exc:
            logger.error("Failed to persist config.json: %s", exc)

    def update(self, patch: dict) -> dict:
        clamped = clamp_settings({k: v for k, v in patch.items() if k in DEFAULT_SETTINGS})
        self.values.update(clamped)
        self.save()
        return clamped

    def as_dict(self) -> dict:
        return dict(self.values)

    def __getattr__(self, item):
        if item in DEFAULT_SETTINGS:
            return self.values[item]
        raise AttributeError(item)


settings_store = SettingsStore(CONFIG_PATH)


# --------------------------------------------------------------------------
# Pipeline abstraction (SPEC.md §7) — result types + protocols
# --------------------------------------------------------------------------

@dataclass
class VisionDetection:
    cls: str
    conf: float
    bbox: list  # [x, y, w, h] pixels
    azimuth_deg: float


@dataclass
class VisionResult:
    confidence: float
    detections: list  # list[VisionDetection]
    sensor_ts: float



@dataclass
class AudioResult:
    confidence: float
    doa_deg: float
    db: float
    waveform: list
    spectrogram: np.ndarray  # rows x cols, uint8
    sensor_ts: float


class VisionPipeline(Protocol):
    connected: bool
    fps: float

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest_frame(self) -> Optional[bytes]: ...
    def latest_result(self) -> Optional[VisionResult]: ...
    def update_params(self, yolo_threshold_low: float, yolo_threshold_high: float, nms_threshold: float, draw_boxes: bool) -> None: ...


class AudioPipeline(Protocol):
    active: bool

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest_result(self) -> Optional[AudioResult]: ...
    def update_params(self, audio_threshold: float, audio_gain: float) -> None: ...


CAMERA_FOV_DEG = 90.0
MIC_ARRAY_RESOLUTION_DEG = 15.0  # +/- => 30 deg sector


def render_vision_frame(cx, cy, size, conf, yolo_threshold_low, yolo_threshold_high, draw_boxes, label_text, visible_margin=0.15):
    """Shared JPEG renderer used by both the simulated and replay vision
    pipelines, so a recorded bbox reproduces pixel-identical output later.

    Mirrors the real hardware's two-stage hysteresis concept (YOLO_LOW/HIGH_
    CONF_THRESHOLD in Jetson_files/config.py): a dim "candidate" circle once
    confidence clears the low threshold, a solid confirmed box + label only
    once it clears the high threshold."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (18, 18, 22)
    for gx in range(0, 640, 80):
        cv2.line(frame, (gx, 0), (gx, 480), (35, 35, 42), 1)
    for gy in range(0, 480, 80):
        cv2.line(frame, (0, gy), (640, gy), (35, 35, 42), 1)
    cv2.putText(frame, label_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 200, 90), 1, cv2.LINE_AA)
    cv2.putText(frame, time.strftime("%H:%M:%S"), (520, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 150), 1, cv2.LINE_AA)

    if cx is not None and conf >= yolo_threshold_low - visible_margin:
        confirmed = conf >= yolo_threshold_high
        color = (60, 200, 255) if confirmed else (90, 90, 110)
        cv2.circle(frame, (int(cx), int(cy)), max(1, int(size / 2)), color, 2, cv2.LINE_AA)
        if draw_boxes and confirmed:
            half = int(size / 2)
            p1 = (int(cx - half), int(cy - half))
            p2 = (int(cx + half), int(cy + half))
            cv2.rectangle(frame, p1, p2, (60, 220, 90), 2)
            label = f"drone {conf:.2f}"
            cv2.putText(frame, label, (p1[0], max(0, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 90), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return buf.tobytes() if ok else None


class SimulatedVisionPipeline:
    """Fabricates a moving synthetic target + JPEG frame. Same protocol surface
    as the real TensorRT/YOLO pipeline (see VisionPipeline)."""

    def __init__(self):
        self.connected = False
        self.fps = 0.0
        self._running = False
        self._t0 = 0.0
        self._yolo_threshold_low = DEFAULT_SETTINGS["yolo_threshold_low"]
        self._yolo_threshold_high = DEFAULT_SETTINGS["yolo_threshold_high"]
        self._nms_threshold = DEFAULT_SETTINGS["nms_threshold"]
        self._draw_boxes = DEFAULT_SETTINGS["draw_boxes"]
        self._frame_count = 0
        self._fps_window_start = 0.0

    def start(self) -> None:
        self._running = True
        self._t0 = time.time()
        self.connected = True
        self._frame_count = 0
        self._fps_window_start = time.time()

    def stop(self) -> None:
        self._running = False
        self.connected = False
        self.fps = 0.0

    def update_params(self, yolo_threshold_low: float, yolo_threshold_high: float, nms_threshold: float, draw_boxes: bool) -> None:
        self._yolo_threshold_low = yolo_threshold_low
        self._yolo_threshold_high = yolo_threshold_high
        self._nms_threshold = nms_threshold
        self._draw_boxes = draw_boxes

    def _sim_target(self, t: float):
        """Deterministic pseudo-target trajectory + confidence used by both
        the frame renderer and the result computer so they stay consistent."""
        x = 320 + 240 * math.sin(t * 0.31)
        y = 240 + 110 * math.sin(t * 0.17 + 1.0)
        conf = 0.5 + 0.48 * math.sin(t * 0.22) + random.uniform(-0.03, 0.03)
        conf = max(0.0, min(1.0, conf))
        azimuth = (x / 640.0 - 0.5) * CAMERA_FOV_DEG
        size = 46 + 18 * math.sin(t * 0.6)
        return x, y, conf, azimuth, size

    def latest_result(self) -> Optional[VisionResult]:
        if not self._running:
            return None
        t = time.time() - self._t0
        x, y, conf, azimuth, size = self._sim_target(t)
        dets = []
        if conf >= self._yolo_threshold_low:
            bbox = [int(x - size / 2), int(y - size / 2), int(size), int(size)]
            dets.append(VisionDetection(cls="drone", conf=round(conf, 3), bbox=bbox, azimuth_deg=round(azimuth, 1)))
        return VisionResult(confidence=round(conf, 3), detections=dets, sensor_ts=time.time())

    def latest_frame(self) -> Optional[bytes]:
        if not self._running:
            return None
        t = time.time() - self._t0
        x, y, conf, azimuth, size = self._sim_target(t)
        frame_bytes = render_vision_frame(x, y, size, conf, self._yolo_threshold_low, self._yolo_threshold_high, self._draw_boxes, "SIMULATED FEED")
        if frame_bytes is None:
            return None
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._fps_window_start = now
        return frame_bytes


class SimulatedAudioPipeline:
    """Fabricates DOA/confidence/waveform/spectrogram. Same protocol surface
    as the real Audio-CNN pipeline (see AudioPipeline)."""

    SPEC_ROWS = 64
    SPEC_COLS = 48

    def __init__(self):
        self.active = False
        self._running = False
        self._t0 = 0.0
        self._audio_threshold = DEFAULT_SETTINGS["audio_threshold"]
        self._audio_gain = DEFAULT_SETTINGS["audio_gain"]
        self._spectrogram = np.zeros((self.SPEC_ROWS, self.SPEC_COLS), dtype=np.uint8)

    def start(self) -> None:
        self._running = True
        self._t0 = time.time()
        self.active = True
        self._spectrogram[:] = 0

    def stop(self) -> None:
        self._running = False
        self.active = False

    def update_params(self, audio_threshold: float, audio_gain: float) -> None:
        self._audio_threshold = audio_threshold
        self._audio_gain = audio_gain

    def _sim_state(self, t: float):
        conf = 0.5 + 0.45 * math.sin(t * 0.26 + 2.0) + random.uniform(-0.03, 0.03)
        conf = max(0.0, min(1.0, conf))
        doa = 70 * math.sin(t * 0.14)
        db = -55 + 25 * math.sin(t * 0.4) + random.uniform(-2, 2)
        return conf, doa, db

    def latest_result(self) -> Optional[AudioResult]:
        if not self._running:
            return None
        t = time.time() - self._t0
        conf, doa, db = self._sim_state(t)

        n = 128
        freq = 2.0 + 4.0 * conf
        samples = [
            self._audio_gain * 0.35 * math.sin(2 * math.pi * freq * (i / n) + t * 3)
            + random.uniform(-0.03, 0.03) * self._audio_gain
            for i in range(n)
        ]

        self._roll_spectrogram(conf, t)

        return AudioResult(
            confidence=round(conf, 3),
            doa_deg=round(doa, 1),
            db=round(db, 1),
            waveform=[round(s, 4) for s in samples],
            spectrogram=self._spectrogram.copy(),
            sensor_ts=time.time(),
        )

    def _roll_spectrogram(self, conf: float, t: float):
        self._spectrogram = np.roll(self._spectrogram, -1, axis=1)
        col = np.random.randint(0, 40, size=self.SPEC_ROWS).astype(np.float32)
        peak_row = int((1.0 - conf) * (self.SPEC_ROWS - 1))
        spread = 6
        rows = np.arange(self.SPEC_ROWS)
        energy = 220 * conf * np.exp(-((rows - peak_row) ** 2) / (2 * spread ** 2))
        col = np.clip(col + energy, 0, 255).astype(np.uint8)
        self._spectrogram[:, -1] = col


# --------------------------------------------------------------------------
# Record / Replay (SPEC.md §9) — a recording is just the exact "telemetry"
# and "spectrogram" messages already broadcast, one JSON object per line.
# Replay is a third implementation of the §7 protocols: same interface,
# same server/frontend code path, zero special-casing elsewhere.
# --------------------------------------------------------------------------

REPLAY_SPEC_ROWS = 64
REPLAY_SPEC_COLS = 48


class ReplaySource:
    """Loads a recording once and lets both replay pipelines walk through it
    in lockstep, paced at the original wall-clock rate and looping forever."""

    def __init__(self, path: Path):
        self.path = path
        self.telemetry: list = []
        self.spectrogram: list = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "telemetry":
                    self.telemetry.append(rec)
                elif rec.get("type") == "spectrogram":
                    self.spectrogram.append(rec)

        all_ts = [r["ts"] for r in self.telemetry] + [r["ts"] for r in self.spectrogram]
        self.t0_rec = min(all_ts) if all_ts else 0.0
        self.duration = (max(all_ts) - self.t0_rec) if all_ts else 0.0
        self.empty = not self.telemetry

        self._t0_wall = time.time()
        self._tel_idx = 0
        self._spec_idx = 0

    def start(self):
        self._t0_wall = time.time()
        self._tel_idx = 0
        self._spec_idx = 0

    def _target_rec_time(self) -> float:
        elapsed = time.time() - self._t0_wall
        if self.duration > 0:
            elapsed = elapsed % self.duration
        return self.t0_rec + elapsed

    def current_telemetry(self) -> Optional[dict]:
        if self.empty:
            return None
        target = self._target_rec_time()
        if self.telemetry[self._tel_idx]["ts"] > target:
            self._tel_idx = 0  # loop wrapped around
        while self._tel_idx < len(self.telemetry) - 1 and self.telemetry[self._tel_idx + 1]["ts"] <= target:
            self._tel_idx += 1
        return self.telemetry[self._tel_idx]

    def current_spectrogram(self) -> Optional[dict]:
        if not self.spectrogram:
            return None
        target = self._target_rec_time()
        if self.spectrogram[self._spec_idx]["ts"] > target:
            self._spec_idx = 0
        while self._spec_idx < len(self.spectrogram) - 1 and self.spectrogram[self._spec_idx + 1]["ts"] <= target:
            self._spec_idx += 1
        return self.spectrogram[self._spec_idx]


class ReplayVisionPipeline:
    """Replays recorded vision telemetry. Same protocol surface as
    SimulatedVisionPipeline / the real TensorRT pipeline (see VisionPipeline)."""

    def __init__(self, source: ReplaySource):
        self.source = source
        self.connected = False
        self.fps = 15.0
        self._running = False
        self._yolo_threshold_low = DEFAULT_SETTINGS["yolo_threshold_low"]
        self._yolo_threshold_high = DEFAULT_SETTINGS["yolo_threshold_high"]
        self._draw_boxes = DEFAULT_SETTINGS["draw_boxes"]

    def start(self) -> None:
        self._running = True
        self.connected = True

    def stop(self) -> None:
        self._running = False
        self.connected = False

    def update_params(self, yolo_threshold_low: float, yolo_threshold_high: float, nms_threshold: float, draw_boxes: bool) -> None:
        self._yolo_threshold_low = yolo_threshold_low
        self._yolo_threshold_high = yolo_threshold_high
        self._draw_boxes = draw_boxes

    def latest_result(self) -> Optional[VisionResult]:
        if not self._running:
            return None
        rec = self.source.current_telemetry()
        if rec is None:
            return None
        vis = rec.get("vision", {})
        dets = [VisionDetection(**d) for d in vis.get("detections", [])]
        return VisionResult(confidence=vis.get("confidence", 0.0), detections=dets, sensor_ts=time.time())

    def latest_frame(self) -> Optional[bytes]:
        if not self._running:
            return None
        rec = self.source.current_telemetry()
        if rec is None:
            return None
        vis = rec.get("vision", {})
        conf = vis.get("confidence", 0.0)
        dets = vis.get("detections") or []
        if dets:
            bx, by, bw, bh = dets[0]["bbox"]
            cx, cy, size = bx + bw / 2, by + bh / 2, max(bw, bh)
        else:
            cx = cy = None
            size = 0
        return render_vision_frame(cx, cy, size, conf, self._yolo_threshold_low, self._yolo_threshold_high, self._draw_boxes, "REPLAY FEED")


class ReplayAudioPipeline:
    """Replays recorded audio telemetry + spectrogram. Same protocol surface
    as SimulatedAudioPipeline / the real Audio-CNN pipeline (see AudioPipeline)."""

    def __init__(self, source: ReplaySource):
        self.source = source
        self.active = False
        self._running = False
        self._audio_threshold = DEFAULT_SETTINGS["audio_threshold"]
        self._audio_gain = DEFAULT_SETTINGS["audio_gain"]

    def start(self) -> None:
        self._running = True
        self.active = True

    def stop(self) -> None:
        self._running = False
        self.active = False

    def update_params(self, audio_threshold: float, audio_gain: float) -> None:
        self._audio_threshold = audio_threshold
        self._audio_gain = audio_gain

    def latest_result(self) -> Optional[AudioResult]:
        if not self._running:
            return None
        rec = self.source.current_telemetry()
        if rec is None:
            return None
        audio = rec.get("audio", {})
        acoustic = rec.get("acoustic", {})

        spec_rec = self.source.current_spectrogram()
        if spec_rec is not None:
            raw = base64.b64decode(spec_rec["data"])
            spec = np.frombuffer(raw, dtype=np.uint8).reshape(spec_rec["rows"], spec_rec["cols"]).copy()
        else:
            spec = np.zeros((REPLAY_SPEC_ROWS, REPLAY_SPEC_COLS), dtype=np.uint8)

        return AudioResult(
            confidence=acoustic.get("confidence", 0.0),
            doa_deg=acoustic.get("doa_deg", 0.0),
            db=audio.get("db") if audio.get("db") is not None else -60.0,
            waveform=rec.get("waveform", []),
            spectrogram=spec,
            sensor_ts=time.time(),
        )


class RecordEngine:
    """Appends broadcast telemetry/spectrogram/event messages to a JSON-lines
    file while enabled. A recording is nothing more than a capture of exactly
    what already went out over the WebSocket."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.enabled = False
        self.filename: Optional[str] = None
        self._file = None

    def start(self, name: Optional[str] = None) -> str:
        self.stop()
        name = name or f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        if not name.endswith(".jsonl"):
            name += ".jsonl"
        self.filename = name
        self._file = (self.directory / name).open("w", encoding="utf-8")
        self.enabled = True
        return name

    def stop(self):
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
        self._file = None
        self.enabled = False

    def write(self, payload: dict):
        if not self.enabled or self._file is None:
            return
        try:
            self._file.write(json.dumps(payload) + "\n")
            self._file.flush()
        except OSError as exc:
            logger.error("Recording write failed: %s", exc)
            self.stop()


class FrameRecordEngine:
    """Saves sampled JPEG frames straight to disk for later model training -
    deliberately separate from RecordEngine's telemetry-only recordings.

    Storage math: 1280x720 JPEG q60 is roughly 80-150KB/frame; at the native
    15 FPS that's ~4-8GB/hour, easily enough to fill a Jetson's onboard
    storage in a single session. Sampling at a low, user-controlled interval
    (default 1 frame/s) and capping the total frame count are the two things
    that keep this from ever becoming a problem - both are set by the caller,
    not hardcoded here."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.enabled = False
        self.interval_s = 1.0
        self.max_frames = 1200
        self.dirname: Optional[Path] = None
        self.frame_count = 0
        self._last_save = 0.0

    def start(self, interval_s: float = 1.0, max_frames: int = 1200) -> str:
        self.stop()
        self.interval_s = max(0.1, float(interval_s))
        self.max_frames = max(1, int(max_frames))
        self.dirname = self.directory / f"frames_{time.strftime('%Y%m%d_%H%M%S')}"
        self.dirname.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self._last_save = 0.0
        self.enabled = True
        return self.dirname.name

    def stop(self):
        self.enabled = False

    def maybe_save(self, jpeg_bytes: Optional[bytes]):
        if not self.enabled or jpeg_bytes is None or self.dirname is None:
            return
        now = time.time()
        if now - self._last_save < self.interval_s:
            return
        if self.frame_count >= self.max_frames:
            logger.warning("Frame recording reached its %d-frame cap - stopping automatically", self.max_frames)
            self.stop()
            return
        self._last_save = now
        path = self.dirname / f"frame_{self.frame_count:06d}_{int(now * 1000)}.jpg"
        try:
            path.write_bytes(jpeg_bytes)
            self.frame_count += 1
        except OSError as exc:
            logger.error("Frame recording write failed: %s", exc)
            self.stop()


class AcousticRecordEngine:
    """Saves short mono WAV clips of the live acoustic feed for further
    training of the acoustic model - hardware mode only (there's no real
    audio in simulated mode). "drone/" is meant to fill automatically off
    real trigger events; "noise/" is meant to fill manually, so the user
    picks representative quiet/background moments themselves and keeps the
    two classes balanced by eye (live counts are broadcast for exactly that).

    Storage math: mono 16kHz WAV is ~64KB/second, so even a 5s clip is only
    ~320KB - at max_samples=300/class that's under 200MB total, a small
    fraction of what unrestrained video frame recording would cost."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.enabled = False           # armed: manual saves + auto-save both possible
        self.auto_save_drone = False   # independent of `enabled` - toggle live, doesn't reset the session; off by default (noisy environments cause false triggers)
        self.window_s = 2.0
        self.max_samples = 300
        self.session_dir: Optional[Path] = None
        self.drone_count = 0
        self.noise_count = 0

    def start(self, window_s: float = 2.0, max_samples: int = 300) -> str:
        self.stop()
        self.window_s = max(1.0, min(5.0, float(window_s)))
        self.max_samples = max(1, int(max_samples))
        self.session_dir = self.directory / f"acoustic_{time.strftime('%Y%m%d_%H%M%S')}"
        (self.session_dir / "drone").mkdir(parents=True, exist_ok=True)
        (self.session_dir / "noise").mkdir(parents=True, exist_ok=True)
        self.drone_count = 0
        self.noise_count = 0
        self.enabled = True
        return self.session_dir.name

    def stop(self):
        self.enabled = False

    def set_auto_save(self, enabled: bool):
        self.auto_save_drone = bool(enabled)

    def save_sample(self, label: str):
        """Returns (ok: bool, info: str) - info is the saved filename on
        success, or a short reason code on failure."""
        if label not in ("drone", "noise"):
            return False, "bad_label"
        if not self.enabled or self.session_dir is None:
            return False, "not_armed"
        count = self.drone_count if label == "drone" else self.noise_count
        if count >= self.max_samples:
            return False, "max_reached"

        window = hardware_bridge.get_audio_window(self.window_s)
        if window is None or len(window) == 0:
            return False, "no_audio"

        sample_rate = hardware_bridge.get_audio_sample_rate()
        path = self.session_dir / label / f"{label}_{count:04d}_{int(time.time() * 1000)}.wav"
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(sample_rate)
                wf.writeframes(window.tobytes())
        except OSError as exc:
            logger.error("Acoustic sample write failed: %s", exc)
            return False, "write_error"

        if label == "drone":
            self.drone_count += 1
        else:
            self.noise_count += 1
        return True, path.name


# --------------------------------------------------------------------------
# Finite State Machine (SPEC.md §5)
# --------------------------------------------------------------------------

class FSM:
    STATES = ("IDLE", "SEARCHING", "SLEWING", "TRACKING", "ENGAGED", "DEGRADED")

    def __init__(self):
        self.state = "IDLE"
        self._pre_degraded_state = "SEARCHING"
        self._slew_start: Optional[float] = None
        self._lost_lock_since: Optional[float] = None
        self._fusion_since: Optional[float] = None

    def _reset_timers(self):
        self._slew_start = None
        self._lost_lock_since = None
        self._fusion_since = None

    def tick(
        self,
        now: float,
        system_active: bool,
        vision: Optional[VisionResult],
        audio: Optional[AudioResult],
        settings: dict,
        camera_ok: bool,
        mic_ok: bool,
    ) -> str:
        if not system_active:
            self.state = "IDLE"
            self._reset_timers()
            return self.state

        hardware_fault = not camera_ok or not mic_ok
        if hardware_fault:
            if self.state != "DEGRADED":
                self._pre_degraded_state = self.state if self.state != "IDLE" else "SEARCHING"
                self.state = "DEGRADED"
            return self.state
        elif self.state == "DEGRADED":
            self.state = self._pre_degraded_state

        if self.state == "IDLE":
            self.state = "SEARCHING"

        visual_hit = bool(vision and vision.detections and vision.confidence >= settings["yolo_threshold_low"])
        acoustic_hit = bool(audio and audio.confidence >= settings["audio_threshold"])
        azimuth_agree = False
        if visual_hit and acoustic_hit:
            azimuth_agree = abs(vision.detections[0].azimuth_deg - audio.doa_deg) <= 15.0
        fused_hit = visual_hit and acoustic_hit and azimuth_agree

        if self.state == "SEARCHING":
            if acoustic_hit:
                self.state = "SLEWING"
                self._slew_start = now
        elif self.state == "SLEWING":
            if self._slew_start is not None and (now - self._slew_start) >= settings["slew_timeout_s"]:
                self.state = "TRACKING"
                self._lost_lock_since = None
        elif self.state == "TRACKING":
            if fused_hit:
                if self._fusion_since is None:
                    self._fusion_since = now
                if (now - self._fusion_since) * 1000.0 <= settings["fusion_window_ms"]:
                    self.state = "ENGAGED"
                    self._lost_lock_since = None
            else:
                self._fusion_since = None
            if not visual_hit:
                if self._lost_lock_since is None:
                    self._lost_lock_since = now
                elif (now - self._lost_lock_since) >= settings["lost_lock_timeout_s"]:
                    self.state = "SEARCHING"
                    self._lost_lock_since = None
                    self._fusion_since = None
            else:
                self._lost_lock_since = None
        elif self.state == "ENGAGED":
            if not fused_hit:
                self.state = "TRACKING"
                self._fusion_since = None

        return self.state


# --------------------------------------------------------------------------
# WebSocket connection manager + log broadcasting
# --------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def send_to(self, ws: WebSocket, payload: dict):
        try:
            await ws.send_json(payload)
        except Exception:
            self.disconnect(ws)


manager = ConnectionManager()
log_history: deque = deque(maxlen=50)
event_history: deque = deque(maxlen=100)


LOG_THROTTLE_INTERVAL_S = 0.3  # per-logger cooldown for repeated INFO/WARNING
                                # bursts (e.g. YOLO-lock or acoustic-trigger
                                # logging once per frame/chunk in hardware
                                # mode) - ERROR/CRITICAL are never throttled.
_last_log_broadcast_by_logger: dict = {}


class WebSocketLogHandler(logging.Handler):
    """Pushes log records to all connected dashboards over the WS channel.
    Attached to the ROOT logger (see on_startup/hardware_bridge.activate) so
    it also catches Jetson_files' own logging, not just this module's."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.loop = loop

    def emit(self, record: logging.LogRecord):
        if record.levelno < logging.ERROR:
            now = time.time()
            last = _last_log_broadcast_by_logger.get(record.name, 0.0)
            if now - last < LOG_THROTTLE_INTERVAL_S:
                return
            _last_log_broadcast_by_logger[record.name] = now
        payload = {
            "type": "log",
            "level": record.levelname,
            "msg": record.getMessage(),
            "ts": time.time(),
        }
        log_history.append(payload)
        try:
            self.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(manager.broadcast(payload))
            )
        except RuntimeError:
            pass


# --------------------------------------------------------------------------
# Global runtime state
# --------------------------------------------------------------------------

@dataclass
class RuntimeState:
    fsm: FSM = field(default_factory=FSM)
    vision: VisionPipeline = field(default_factory=SimulatedVisionPipeline)
    audio: AudioPipeline = field(default_factory=SimulatedAudioPipeline)
    live_vision: Optional[VisionPipeline] = None
    live_audio: Optional[AudioPipeline] = None
    replay_vision: Optional[ReplayVisionPipeline] = None
    replay_audio: Optional[ReplayAudioPipeline] = None
    hw_vision: Optional[VisionPipeline] = None
    hw_audio: Optional[AudioPipeline] = None
    mjpeg_clients: int = 0
    latest_jpeg: Optional[bytes] = None
    mode: str = "live"
    current_recording_file: Optional[str] = None
    sim_badge_until: float = 0.0
    prev_visual_hit: bool = False
    prev_acoustic_hit: bool = False
    prev_state: str = "IDLE"
    record_engine: RecordEngine = field(init=False)
    frame_record_engine: FrameRecordEngine = field(init=False)
    acoustic_record_engine: AcousticRecordEngine = field(init=False)
    hardware_active: bool = False
    sim_camera_pan: float = 0.0  # simulated-mode-only: fabricated PTZ bearing for demo purposes

    def __post_init__(self):
        self.live_vision = self.vision
        self.live_audio = self.audio
        self.record_engine = RecordEngine(RECORDINGS_DIR)
        self.frame_record_engine = FrameRecordEngine(RECORDINGS_DIR)
        self.acoustic_record_engine = AcousticRecordEngine(RECORDINGS_DIR)


runtime = RuntimeState()


def record_if_active(payload: dict):
    if runtime.hardware_active or settings_store.system_active:
        runtime.record_engine.write(payload)


def acoustic_recording_payload() -> dict:
    are = runtime.acoustic_record_engine
    return {
        "type": "acoustic_recording_state",
        "recording": are.enabled,
        "auto_save_drone": are.auto_save_drone,
        "dir": are.session_dir.name if are.session_dir else None,
        "drone_count": are.drone_count,
        "noise_count": are.noise_count,
        "max_samples": are.max_samples,
        "window_s": are.window_s,
    }


def start_jtop():
    """Opens the ONE jtop session for the process's entire lifetime. Must be
    called exactly once, at server startup - never per poll. `jtop()`'s
    connection setup is a blocking IPC handshake with the jetson-stats
    background service; re-creating it every ~1s inside jetson_metrics() (as
    this used to do, via `with jtop() as jetson:` on every call) blocked the
    async event loop for tenths of a second each time, stalling the MJPEG
    video stream running on that same loop and causing visible stutter.
    Once started, jtop runs its own background thread that keeps
    jetson.cpu/gpu/temperature updated in memory - jetson_metrics() below
    just reads those (cheap, non-blocking)."""
    global _jtop_handle
    if not HAS_JTOP:
        return
    try:
        handle = jtop()
        handle.start()
        _jtop_handle = handle
        logger.info("jtop session opened (persistent for process lifetime)")
    except Exception as exc:
        logger.warning("jtop start failed (%s) - falling back to psutil", exc)
        _jtop_handle = None


def stop_jtop():
    global _jtop_handle
    if _jtop_handle is not None:
        try:
            _jtop_handle.close()
        except Exception:
            pass
        _jtop_handle = None


def jetson_metrics() -> dict:
    """jtop is authoritative on-device; psutil is the dev-machine fallback
    (no real GPU/temp reading on non-Jetson hardware -> 'N/A'). Reads the
    single persistent session from start_jtop() - does NOT open a new one."""
    if _jtop_handle is not None:
        try:
            if _jtop_handle.ok():
                # jtop.cpu["cpu"] is a list of per-core dicts in current
                # jetson-stats versions (was a dict keyed by core index in
                # older ones, hence the isinstance guard) - iterate it
                # directly rather than treating its own values as indices.
                cores = _jtop_handle.cpu["cpu"]
                core_list = cores.values() if isinstance(cores, dict) else cores
                usages = [c["user"] for c in core_list if isinstance(c, dict) and "user" in c]
                return {
                    "cpu": round(sum(usages) / max(1, len(usages)), 1),
                    "gpu": round(_jtop_handle.gpu.get("val", 0), 1),
                    "temp_c": round(_jtop_handle.temperature.get("CPU", {}).get("temp", 0), 1),
                }
        except Exception as exc:
            logger.warning("jtop read failed: %s", exc)
    if psutil is not None:
        return {"cpu": round(psutil.cpu_percent(), 1), "gpu": None, "temp_c": None}
    return {"cpu": None, "gpu": None, "temp_c": None}


def fabricate_fused_event(sim: bool = True) -> dict:
    now = time.time()
    azimuth = round(random.uniform(-30, 30), 1)
    visual_conf = round(random.uniform(0.7, 0.95), 2)
    acoustic_conf = round(random.uniform(0.7, 0.95), 2)
    combined_conf = round((visual_conf + acoustic_conf) / 2, 3)
    ev = {
        "ts": now,
        "modality": "Fused",
        "cls": "drone",
        "confidence": combined_conf,
        "visual_confidence": visual_conf,
        "acoustic_confidence": acoustic_conf,
        "combined_confidence": combined_conf,
        "azimuth_deg": azimuth,
        "sim": sim,
    }
    return ev


# --------------------------------------------------------------------------
# Background loops (§8 perf budget: async, no busy-waiting; errors isolated)
# --------------------------------------------------------------------------

TELEMETRY_HZ = 10.0
SPECTROGRAM_HZ = 3.0
VIDEO_FPS = 15.0


async def video_encode_loop():
    interval = 1.0 / VIDEO_FPS
    while True:
        try:
            # Normally encode only while a viewer is connected (SPEC.md §8) - but
            # frame recording for model training must keep sampling even with the
            # browser tab closed, so it also counts as a reason to keep encoding.
            need_frames = runtime.mjpeg_clients > 0 or runtime.frame_record_engine.enabled
            if need_frames and (runtime.hardware_active or settings_store.system_active):
                frame = runtime.vision.latest_frame()
                if frame is not None:
                    runtime.latest_jpeg = frame
                    runtime.frame_record_engine.maybe_save(frame)
            else:
                runtime.latest_jpeg = None
            await asyncio.sleep(interval)
        except Exception as exc:
            logger.error("Video encode loop error: %s", exc)
            await asyncio.sleep(2)


def _record_event_if_new(
    modality: str,
    cls: str,
    azimuth: float,
    vision_result: Optional[VisionResult],
    audio_result: Optional[AudioResult],
):
    """Builds one Detection Event History row. Captures BOTH sensors' current
    confidence (not just whichever one triggered this event) so the table can
    show what the other modality was reading for the same target at that
    moment, plus a simple 50/50 combined score when both are available."""
    visual_conf = round(vision_result.confidence, 3) if vision_result else None
    acoustic_conf = round(audio_result.confidence, 3) if audio_result else None
    combined_conf = round((visual_conf + acoustic_conf) / 2, 3) if (visual_conf is not None and acoustic_conf is not None) else None
    primary_conf = {"Visual": visual_conf, "Acoustic": acoustic_conf, "Fused": combined_conf}.get(modality)

    ev = {
        "ts": time.time(),
        "modality": modality,
        "cls": cls,
        "confidence": primary_conf,
        "visual_confidence": visual_conf,
        "acoustic_confidence": acoustic_conf,
        "combined_confidence": combined_conf,
        "azimuth_deg": azimuth,
        "sim": False,
    }
    event_history.append(ev)
    return ev


STAGE_LATENCY_SIM_BASELINE = {
    "mic_buffer_ms": 200.0,
    "fft_ms": 15.0,
    "cnn_ms": 8.0,
    "frame_capture_ms": 5.0,
    "yolo_ms": 25.0,
    "ptz_ms": 2.0,
}


def fabricate_stage_latencies() -> dict:
    """No real pipeline to instrument in simulated mode - small jitter around
    plausible Jetson Orin Nano figures, purely for a consistent demo (same
    fabrication approach already used for the compass strip)."""
    return {k: round(max(0.0, v + random.uniform(-v * 0.2, v * 0.2)), 1) for k, v in STAGE_LATENCY_SIM_BASELINE.items()}


async def telemetry_loop():
    interval = 1.0 / TELEMETRY_HZ
    metrics_counter = 0
    cached_jetson = jetson_metrics()
    while True:
        try:
            now = time.time()
            active = runtime.hardware_active or settings_store.system_active
            vision_result = runtime.vision.latest_result() if active else None
            audio_result = runtime.audio.latest_result() if active else None

            if runtime.hardware_active:
                # The live IntegratedDroneDefenseSystem owns the authoritative FSM;
                # mirror it instead of recomputing state from simulated thresholds.
                state = hardware_bridge.get_fsm_state()
                camera_ok = hardware_bridge.get_camera_ok()
                mic_ok = hardware_bridge.get_mic_ok()
                camera_pan_deg = hardware_bridge.get_camera_pan_deg()
            else:
                camera_ok = (not active) or runtime.vision.connected
                mic_ok = (not active) or runtime.audio.active
                state = runtime.fsm.tick(
                    now=now,
                    system_active=active,
                    vision=vision_result,
                    audio=audio_result,
                    settings=settings_store.as_dict(),
                    camera_ok=camera_ok,
                    mic_ok=mic_ok,
                )
                # No real PTZ in simulated mode - fabricate a plausible bearing that
                # eases toward the (also fabricated) acoustic DOA, purely so the
                # compass strip has something meaningful to show during a demo.
                target_doa = audio_result.doa_deg if audio_result else 0.0
                runtime.sim_camera_pan += (target_doa - runtime.sim_camera_pan) * 0.15
                camera_pan_deg = runtime.sim_camera_pan

            if state != runtime.prev_state:
                logger.info("State transition: %s -> %s", runtime.prev_state, state)
                if state == "ENGAGED" and vision_result and vision_result.detections:
                    ev = _record_event_if_new(
                        "Fused", "drone", vision_result.detections[0].azimuth_deg, vision_result, audio_result
                    )
                    record_if_active({"type": "event", **ev})
                    await manager.broadcast({"type": "event", **ev})
                runtime.prev_state = state

            if vision_result and vision_result.detections:
                visual_hit = vision_result.confidence >= settings_store.yolo_threshold_low
                if visual_hit and not runtime.prev_visual_hit:
                    ev = _record_event_if_new(
                        "Visual", vision_result.detections[0].cls, vision_result.detections[0].azimuth_deg,
                        vision_result, audio_result,
                    )
                    record_if_active({"type": "event", **ev})
                    await manager.broadcast({"type": "event", **ev})
                runtime.prev_visual_hit = visual_hit
            else:
                runtime.prev_visual_hit = False

            if audio_result:
                acoustic_hit = audio_result.confidence >= settings_store.audio_threshold
                if acoustic_hit and not runtime.prev_acoustic_hit:
                    ev = _record_event_if_new("Acoustic", "drone", audio_result.doa_deg, vision_result, audio_result)
                    record_if_active({"type": "event", **ev})
                    await manager.broadcast({"type": "event", **ev})

                    are = runtime.acoustic_record_engine
                    if runtime.hardware_active and are.enabled and are.auto_save_drone:
                        ok, info = are.save_sample("drone")
                        if ok:
                            logger.info("Auto-saved drone acoustic sample -> %s", info)
                            await manager.broadcast(acoustic_recording_payload())
                        elif info != "not_armed":
                            logger.warning("Auto-save of drone acoustic sample failed: %s", info)
                runtime.prev_acoustic_hit = acoustic_hit
            else:
                runtime.prev_acoustic_hit = False

            metrics_counter += 1
            if metrics_counter % 10 == 0:  # jetson stats read at ~1 Hz, not 10 Hz
                cached_jetson = jetson_metrics()

            payload = {
                "type": "telemetry",
                "ts": now,
                "state": state,
                "mode": runtime.mode,
                "hardware": runtime.hardware_active,
                "sim": runtime.mode == "replay" or now < runtime.sim_badge_until,
                "camera": {"connected": bool(active and runtime.vision.connected), "fps": runtime.vision.fps},
                "audio": {"active": bool(active and runtime.audio.active), "db": audio_result.db if audio_result else None},
                "jetson": cached_jetson,
                "vision": {
                    "confidence": vision_result.confidence if vision_result else 0.0,
                    "detections": [asdict(d) for d in vision_result.detections] if vision_result else [],
                },
                "acoustic": {
                    "confidence": audio_result.confidence if audio_result else 0.0,
                    "doa_deg": audio_result.doa_deg if audio_result else 0.0,
                },
                "waveform": audio_result.waveform if audio_result else [],
                "stage_latency_ms": hardware_bridge.get_stage_timings() if runtime.hardware_active else fabricate_stage_latencies(),
                "sensor_ts": (vision_result.sensor_ts if vision_result else (audio_result.sensor_ts if audio_result else now)),
                "camera_pan_deg": round(camera_pan_deg, 1),
                "frame_recording": {
                    "active": runtime.frame_record_engine.enabled,
                    "count": runtime.frame_record_engine.frame_count,
                    "max_frames": runtime.frame_record_engine.max_frames,
                    "dir": runtime.frame_record_engine.dirname.name if runtime.frame_record_engine.dirname else None,
                },
                "acoustic_recording": {
                    "active": runtime.acoustic_record_engine.enabled,
                    "auto_save_drone": runtime.acoustic_record_engine.auto_save_drone,
                    "drone_count": runtime.acoustic_record_engine.drone_count,
                    "noise_count": runtime.acoustic_record_engine.noise_count,
                    "max_samples": runtime.acoustic_record_engine.max_samples,
                },
            }
            record_if_active(payload)
            await manager.broadcast(payload)
            await asyncio.sleep(interval)
        except Exception as exc:
            logger.error("Telemetry loop error: %s", exc)
            await asyncio.sleep(2)


async def spectrogram_loop():
    interval = 1.0 / SPECTROGRAM_HZ
    while True:
        try:
            if runtime.hardware_active or settings_store.system_active:
                audio_result = runtime.audio.latest_result()
                if audio_result is not None:
                    spec = audio_result.spectrogram
                    payload = {
                        "type": "spectrogram",
                        "ts": time.time(),
                        "rows": spec.shape[0],
                        "cols": spec.shape[1],
                        "data": base64.b64encode(spec.tobytes()).decode("ascii"),
                    }
                    record_if_active(payload)
                    await manager.broadcast(payload)
            await asyncio.sleep(interval)
        except Exception as exc:
            logger.error("Spectrogram loop error: %s", exc)
            await asyncio.sleep(2)


def apply_pipeline_params():
    runtime.vision.update_params(
        yolo_threshold_low=settings_store.yolo_threshold_low,
        yolo_threshold_high=settings_store.yolo_threshold_high,
        nms_threshold=settings_store.nms_threshold,
        draw_boxes=settings_store.draw_boxes,
    )
    runtime.audio.update_params(
        audio_threshold=settings_store.audio_threshold,
        audio_gain=settings_store.audio_gain,
    )


def set_system_active(active: bool):
    settings_store.values["system_active"] = active
    settings_store.save()
    if active:
        runtime.vision.start()
        runtime.audio.start()
        apply_pipeline_params()
        logger.info("System START — pipelines initialized")
    else:
        runtime.vision.stop()
        runtime.audio.stop()
        runtime.fsm._reset_timers()
        logger.info("System STOP — pipelines paused")


def switch_to_live():
    """Third protocol implementation swap (§7/§9): point the active pipeline
    references back at the persistent simulated instances. Server/frontend
    code elsewhere is untouched — it only ever talks to runtime.vision/audio."""
    if runtime.mode == "live":
        return
    was_active = settings_store.system_active
    runtime.vision.stop()
    runtime.audio.stop()
    runtime.vision = runtime.live_vision
    runtime.audio = runtime.live_audio
    runtime.mode = "live"
    runtime.current_recording_file = None
    apply_pipeline_params()
    if was_active:
        runtime.vision.start()
        runtime.audio.start()
    logger.info("Live mode resumed")


def switch_to_replay(filename: str) -> bool:
    path = RECORDINGS_DIR / filename
    if not path.exists():
        logger.error("Replay file not found: %s", filename)
        return False

    source = ReplaySource(path)
    if source.empty:
        logger.error("Replay file has no telemetry records: %s", filename)
        return False

    was_active = settings_store.system_active
    runtime.vision.stop()
    runtime.audio.stop()

    runtime.replay_vision = ReplayVisionPipeline(source)
    runtime.replay_audio = ReplayAudioPipeline(source)
    runtime.vision = runtime.replay_vision
    runtime.audio = runtime.replay_audio
    runtime.mode = "replay"
    runtime.current_recording_file = filename
    apply_pipeline_params()
    if was_active:
        runtime.vision.start()
        runtime.audio.start()
    logger.info("Replay mode engaged: %s", filename)
    return True


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(title="UAV Multi-Modal Detection Dashboard")


@app.on_event("startup")
async def on_startup():
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    handler = WebSocketLogHandler(loop)
    handler.setLevel(logging.INFO)
    # Attached to the ROOT logger (not just "dashboard") so Jetson_files' own
    # logging - main_system.py's bare logging.info/warning/error calls plus
    # the "AcousticSystem"/"OpticalSystem" named loggers - reaches the GUI
    # too, not just our own FSM-transition messages. This logger ("dashboard")
    # already propagates to root by default, so nothing needs to attach there
    # directly, avoiding a double-broadcast.
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    start_jtop()

    if os.environ.get("HARDWARE_MODE") == "1":
        if hardware_bridge.activate(handler):
            runtime.hardware_active = True
            runtime.hw_vision = hardware_bridge.HardwareVisionPipeline(VisionResult, VisionDetection)
            runtime.hw_audio = hardware_bridge.HardwareAudioPipeline(AudioResult)
            runtime.live_vision = runtime.hw_vision
            runtime.live_audio = runtime.hw_audio
            runtime.vision = runtime.hw_vision
            runtime.audio = runtime.hw_audio
        else:
            logger.warning("HARDWARE_MODE=1 requested but the hardware bridge failed to activate — staying simulated")

    if runtime.hardware_active:
        # The live defense grid runs continuously from here on regardless of the
        # GUI's START/STOP switch (that switch only controls replay playback) —
        # just push the persisted threshold settings into the real system once.
        apply_pipeline_params()
        logger.info("Dashboard backend started (hardware bridge — live IntegratedDroneDefenseSystem)")
    else:
        if settings_store.system_active:
            runtime.vision.start()
            runtime.audio.start()
            apply_pipeline_params()
        logger.info("Dashboard backend started (Phase 1 — simulated pipelines)")

    asyncio.create_task(video_encode_loop())
    asyncio.create_task(telemetry_loop())
    asyncio.create_task(spectrogram_loop())


@app.on_event("shutdown")
async def on_shutdown():
    runtime.record_engine.stop()
    if runtime.hardware_active:
        # hardware_bridge.shutdown() blocks (it joins DroneSystemThread), so it
        # runs in the default executor rather than inline - keeps the event
        # loop spinning during the join, which is what lets its "dashboard"
        # logger calls actually reach connected GUI clients via
        # WebSocketLogHandler (call_soon_threadsafe needs a live loop).
        await asyncio.get_event_loop().run_in_executor(None, hardware_bridge.shutdown)
    stop_jtop()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/recordings")
async def list_recordings():
    files = sorted(RECORDINGS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"files": [f.name for f in files]}


@app.get("/video_feed")
async def video_feed():
    async def generator():
        runtime.mjpeg_clients += 1
        try:
            interval = 1.0 / VIDEO_FPS
            while True:
                await asyncio.sleep(interval)
                frame = runtime.latest_jpeg
                if frame is None:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
        finally:
            runtime.mjpeg_clients = max(0, runtime.mjpeg_clients - 1)

    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await manager.send_to(websocket, {"type": "settings_snapshot", **settings_store.as_dict()})
    await manager.send_to(websocket, {
        "type": "recording_state",
        "recording": runtime.record_engine.enabled,
        "file": runtime.record_engine.filename,
    })
    await manager.send_to(websocket, {
        "type": "frame_recording_state",
        "recording": runtime.frame_record_engine.enabled,
        "dir": runtime.frame_record_engine.dirname.name if runtime.frame_record_engine.dirname else None,
        "count": runtime.frame_record_engine.frame_count,
        "max_frames": runtime.frame_record_engine.max_frames,
        "interval_s": runtime.frame_record_engine.interval_s,
    })
    await manager.send_to(websocket, acoustic_recording_payload())
    pan_min, pan_max = hardware_bridge.get_camera_pan_range()
    await manager.send_to(websocket, {"type": "camera_range", "min_deg": pan_min, "max_deg": pan_max})
    await manager.send_to(websocket, {
        "type": "mode",
        "mode": runtime.mode,
        "file": runtime.current_recording_file,
        "ok": True,
    })
    for line in list(log_history):
        await manager.send_to(websocket, line)
    for ev in list(event_history):
        await manager.send_to(websocket, {"type": "event", **ev})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Malformed WS message ignored")
                continue

            command = msg.get("command")

            if command == "update_settings":
                was_active = settings_store.system_active
                patch = {k: v for k, v in msg.items() if k != "command"}
                if "system_active" in patch:
                    patch["system_active"] = bool(patch["system_active"])
                applied = settings_store.update(patch)
                apply_pipeline_params()
                if "system_active" in patch and bool(patch["system_active"]) != was_active:
                    set_system_active(bool(patch["system_active"]))
                await manager.send_to(websocket, {"type": "settings_ack", **settings_store.as_dict()})

            elif command == "set_mode":
                mode = msg.get("mode", "live")
                file = msg.get("file")
                ok = True
                if mode == "replay":
                    if not file:
                        logger.warning("set_mode 'replay' requires a 'file'")
                        ok = False
                    else:
                        ok = switch_to_replay(file)
                else:
                    switch_to_live()
                await manager.broadcast({
                    "type": "mode",
                    "mode": runtime.mode,
                    "file": runtime.current_recording_file,
                    "ok": ok,
                })

            elif command == "set_recording":
                enabled = bool(msg.get("enabled", False))
                if enabled and not runtime.record_engine.enabled:
                    filename = runtime.record_engine.start()
                    logger.info("Recording started -> %s", filename)
                elif not enabled and runtime.record_engine.enabled:
                    logger.info("Recording stopped -> %s", runtime.record_engine.filename)
                    runtime.record_engine.stop()
                await manager.broadcast({
                    "type": "recording_state",
                    "recording": runtime.record_engine.enabled,
                    "file": runtime.record_engine.filename,
                })

            elif command == "set_frame_recording":
                enabled = bool(msg.get("enabled", False))
                fre = runtime.frame_record_engine
                if enabled and not fre.enabled:
                    try:
                        interval_s = max(0.1, float(msg.get("interval_s", 1.0)))
                        max_frames = max(1, int(msg.get("max_frames", 1200)))
                    except (TypeError, ValueError):
                        interval_s, max_frames = 1.0, 1200
                    dirname = fre.start(interval_s=interval_s, max_frames=max_frames)
                    logger.info("Frame recording started -> %s (every %.1fs, cap %d frames)", dirname, interval_s, max_frames)
                elif not enabled and fre.enabled:
                    logger.info("Frame recording stopped -> %s (%d frames saved)",
                                 fre.dirname.name if fre.dirname else "?", fre.frame_count)
                    fre.stop()
                await manager.broadcast({
                    "type": "frame_recording_state",
                    "recording": fre.enabled,
                    "dir": fre.dirname.name if fre.dirname else None,
                    "count": fre.frame_count,
                    "max_frames": fre.max_frames,
                    "interval_s": fre.interval_s,
                })

            elif command == "set_acoustic_recording":
                enabled = bool(msg.get("enabled", False))
                are = runtime.acoustic_record_engine
                if enabled and not are.enabled:
                    try:
                        window_s = max(1.0, min(5.0, float(msg.get("window_s", 2.0))))
                        max_samples = max(1, int(msg.get("max_samples", 300)))
                    except (TypeError, ValueError):
                        window_s, max_samples = 2.0, 300
                    dirname = are.start(window_s=window_s, max_samples=max_samples)
                    logger.info("Acoustic sample recording armed -> %s (window=%.1fs, cap %d/class)", dirname, window_s, max_samples)
                elif not enabled and are.enabled:
                    logger.info("Acoustic sample recording disarmed -> %s (drone=%d, noise=%d)",
                                 are.session_dir.name if are.session_dir else "?", are.drone_count, are.noise_count)
                    are.stop()
                await manager.broadcast(acoustic_recording_payload())

            elif command == "set_acoustic_auto_save":
                are = runtime.acoustic_record_engine
                are.set_auto_save(bool(msg.get("enabled", True)))
                logger.info("Acoustic auto-save on drone trigger: %s", "ON" if are.auto_save_drone else "OFF")
                await manager.broadcast(acoustic_recording_payload())

            elif command == "save_acoustic_sample":
                label = msg.get("label")
                are = runtime.acoustic_record_engine
                if label not in ("drone", "noise"):
                    logger.warning("save_acoustic_sample requires label 'drone' or 'noise'")
                elif not runtime.hardware_active:
                    logger.warning("save_acoustic_sample requires hardware mode - no real audio in simulated mode")
                else:
                    ok, info = are.save_sample(label)
                    if ok:
                        logger.info("Saved %s acoustic sample -> %s", label, info)
                    else:
                        logger.warning("Failed to save %s acoustic sample: %s", label, info)
                    await manager.broadcast(acoustic_recording_payload())

            elif command == "inject_demo_detection":
                ev = fabricate_fused_event(sim=True)
                event_history.append(ev)
                record_if_active({"type": "event", **ev})
                runtime.sim_badge_until = time.time() + 6.0
                logger.info("Demo detection injected (SIM)")
                await manager.broadcast({"type": "event", **ev})

            else:
                logger.warning("Unknown command: %s", command)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.error("WS handler error: %s", exc)
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    # timeout_graceful_shutdown: uvicorn's default graceful shutdown waits
    # indefinitely for every open connection to close on the client's own
    # initiative - and /video_feed (an infinite MJPEG generator) and /ws (a
    # long-lived WebSocket) never do that on their own. A browser tab left
    # open on the dashboard would otherwise make Ctrl+C hang forever, before
    # on_shutdown() (and hardware_bridge.shutdown()'s motor stop) ever runs.
    # After 5s uvicorn force-closes remaining connections and proceeds.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, timeout_graceful_shutdown=5)
