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
    "yolo_threshold": (0.0, 1.0),
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
    "yolo_threshold": 0.55,
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
    def update_params(self, yolo_threshold: float, nms_threshold: float, draw_boxes: bool) -> None: ...


class AudioPipeline(Protocol):
    active: bool

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest_result(self) -> Optional[AudioResult]: ...
    def update_params(self, audio_threshold: float, audio_gain: float) -> None: ...


CAMERA_FOV_DEG = 90.0
MIC_ARRAY_RESOLUTION_DEG = 15.0  # +/- => 30 deg sector


def render_vision_frame(cx, cy, size, conf, yolo_threshold, draw_boxes, label_text, visible_margin=0.15):
    """Shared JPEG renderer used by both the simulated and replay vision
    pipelines, so a recorded bbox reproduces pixel-identical output later."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (18, 18, 22)
    for gx in range(0, 640, 80):
        cv2.line(frame, (gx, 0), (gx, 480), (35, 35, 42), 1)
    for gy in range(0, 480, 80):
        cv2.line(frame, (0, gy), (640, gy), (35, 35, 42), 1)
    cv2.putText(frame, label_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 200, 90), 1, cv2.LINE_AA)
    cv2.putText(frame, time.strftime("%H:%M:%S"), (520, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 150), 1, cv2.LINE_AA)

    if cx is not None and conf >= yolo_threshold - visible_margin:
        color = (60, 200, 255) if conf >= yolo_threshold else (90, 90, 110)
        cv2.circle(frame, (int(cx), int(cy)), max(1, int(size / 2)), color, 2, cv2.LINE_AA)
        if draw_boxes and conf >= yolo_threshold:
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
        self._yolo_threshold = DEFAULT_SETTINGS["yolo_threshold"]
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

    def update_params(self, yolo_threshold: float, nms_threshold: float, draw_boxes: bool) -> None:
        self._yolo_threshold = yolo_threshold
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
        if conf >= self._yolo_threshold:
            bbox = [int(x - size / 2), int(y - size / 2), int(size), int(size)]
            dets.append(VisionDetection(cls="drone", conf=round(conf, 3), bbox=bbox, azimuth_deg=round(azimuth, 1)))
        return VisionResult(confidence=round(conf, 3), detections=dets, sensor_ts=time.time())

    def latest_frame(self) -> Optional[bytes]:
        if not self._running:
            return None
        t = time.time() - self._t0
        x, y, conf, azimuth, size = self._sim_target(t)
        frame_bytes = render_vision_frame(x, y, size, conf, self._yolo_threshold, self._draw_boxes, "SIMULATED FEED")
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
        self._yolo_threshold = DEFAULT_SETTINGS["yolo_threshold"]
        self._draw_boxes = DEFAULT_SETTINGS["draw_boxes"]

    def start(self) -> None:
        self._running = True
        self.connected = True

    def stop(self) -> None:
        self._running = False
        self.connected = False

    def update_params(self, yolo_threshold: float, nms_threshold: float, draw_boxes: bool) -> None:
        self._yolo_threshold = yolo_threshold
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
        return render_vision_frame(cx, cy, size, conf, self._yolo_threshold, self._draw_boxes, "REPLAY FEED")


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

        visual_hit = bool(vision and vision.detections and vision.confidence >= settings["yolo_threshold"])
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


class WebSocketLogHandler(logging.Handler):
    """Pushes log records to all connected dashboards over the WS channel."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.loop = loop

    def emit(self, record: logging.LogRecord):
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
    hardware_active: bool = False

    def __post_init__(self):
        self.live_vision = self.vision
        self.live_audio = self.audio
        self.record_engine = RecordEngine(RECORDINGS_DIR)


runtime = RuntimeState()


def record_if_active(payload: dict):
    if runtime.hardware_active or settings_store.system_active:
        runtime.record_engine.write(payload)


def jetson_metrics() -> dict:
    """jtop is authoritative on-device; psutil is the dev-machine fallback
    (no real GPU/temp reading on non-Jetson hardware -> 'N/A')."""
    if HAS_JTOP:
        try:
            with jtop() as jetson:
                if jetson.ok():
                    return {
                        "cpu": round(sum(jetson.cpu["cpu"][i]["user"] for i in jetson.cpu["cpu"]) / max(1, len(jetson.cpu["cpu"])), 1),
                        "gpu": round(jetson.gpu.get("val", 0), 1),
                        "temp_c": round(jetson.temperature.get("CPU", {}).get("temp", 0), 1),
                    }
        except Exception as exc:
            logger.warning("jtop read failed: %s", exc)
    if psutil is not None:
        return {"cpu": round(psutil.cpu_percent(), 1), "gpu": None, "temp_c": None}
    return {"cpu": None, "gpu": None, "temp_c": None}


def fabricate_fused_event(sim: bool = True) -> dict:
    now = time.time()
    azimuth = round(random.uniform(-30, 30), 1)
    conf = round(random.uniform(0.75, 0.95), 2)
    ev = {
        "ts": now,
        "modality": "Fused",
        "cls": "drone",
        "confidence": conf,
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
            if runtime.mjpeg_clients > 0 and (runtime.hardware_active or settings_store.system_active):
                frame = runtime.vision.latest_frame()
                if frame is not None:
                    runtime.latest_jpeg = frame
            else:
                runtime.latest_jpeg = None
            await asyncio.sleep(interval)
        except Exception as exc:
            logger.error("Video encode loop error: %s", exc)
            await asyncio.sleep(2)


def _record_event_if_new(modality: str, cls: str, conf: float, azimuth: float):
    ev = {
        "ts": time.time(),
        "modality": modality,
        "cls": cls,
        "confidence": conf,
        "azimuth_deg": azimuth,
        "sim": False,
    }
    event_history.append(ev)
    return ev


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

            if state != runtime.prev_state:
                logger.info("State transition: %s -> %s", runtime.prev_state, state)
                if state == "ENGAGED" and vision_result and vision_result.detections:
                    ev = _record_event_if_new(
                        "Fused", "drone", vision_result.detections[0].conf, vision_result.detections[0].azimuth_deg
                    )
                    record_if_active({"type": "event", **ev})
                    await manager.broadcast({"type": "event", **ev})
                runtime.prev_state = state

            if vision_result and vision_result.detections:
                visual_hit = vision_result.confidence >= settings_store.yolo_threshold
                if visual_hit and not runtime.prev_visual_hit:
                    ev = _record_event_if_new(
                        "Visual", vision_result.detections[0].cls, vision_result.confidence,
                        vision_result.detections[0].azimuth_deg,
                    )
                    record_if_active({"type": "event", **ev})
                    await manager.broadcast({"type": "event", **ev})
                runtime.prev_visual_hit = visual_hit
            else:
                runtime.prev_visual_hit = False

            if audio_result:
                acoustic_hit = audio_result.confidence >= settings_store.audio_threshold
                if acoustic_hit and not runtime.prev_acoustic_hit:
                    ev = _record_event_if_new("Acoustic", "drone", audio_result.confidence, audio_result.doa_deg)
                    record_if_active({"type": "event", **ev})
                    await manager.broadcast({"type": "event", **ev})
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
                "sensor_ts": (vision_result.sensor_ts if vision_result else (audio_result.sensor_ts if audio_result else now)),
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
        yolo_threshold=settings_store.yolo_threshold,
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
    logger.addHandler(handler)

    if os.environ.get("HARDWARE_MODE") == "1":
        if hardware_bridge.activate():
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
