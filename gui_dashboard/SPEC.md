# System Specification: Local Web GUI for Embedded Multi-Modal Drone Detection System
## v3.2 — Full Production Specification (Integrated and Completed)

---

## 1. Project Overview & Deployment Topology

* **Target System:** Nvidia Jetson Orin Nano processing real-time multi-modal data.
* **Input Modalities:** 1x Camera Stream (YOLOv8n via CUDA), 1x Microphone Array Stream (Audio CNN).
* **Architecture:** Python backend on the Jetson (sensor pipelines + models + web server). The dashboard is viewed from a **separate laptop's browser** over the network.
* **⚠ Hard rule — no local browser on the Jetson.** The Jetson has unified memory (CPU and GPU share the same RAM). A local Chromium instance steals 1 GB+ RAM and GPU compositing cycles from the models. The Jetson serves; the laptop renders. Streaming the dashboard costs CPU/network only (~1–2 MB/s) and does not touch the GPU.
* **Core Goal:** A clean, dual-purpose dashboard for graduation-project presentation and system debugging, operating without starving the AI models of resources.

### Network options (laptop ↔ Jetson)

| Option | Use for | Notes |
|---|---|---|
| Shared WiFi (router) | Daily development | Do NOT rely on it for the defense — campus WiFi often blocks client-to-client traffic |
| **Direct Ethernet cable** | **The defense (primary)** | Static IPs both sides (e.g., 192.168.1.10 / 192.168.1.20); ms latency, zero surprises |
| Jetson as WiFi hotspot | Field demo, no cables | Built into Ubuntu (Settings → WiFi → Hotspot); configure in advance as backup |

Run uvicorn with `--host 0.0.0.0` (otherwise only localhost can connect).

---

## 2. Technical Stack

* **Backend:** FastAPI (Python) — async + WebSockets.
* **Frontend:** HTML5, Tailwind CSS (via CDN), Single-file modern React application injected via Babel/CDN inside the template. No build step.
* **Visualizations:** Chart.js (waveform, **`animation: false` mandatory**) and HTML5 Canvas (radar + spectrogram).
* **Jetson Metrics:** `jetson-stats` (jtop) API to pull true GPU core utilization and physical thermal zone values — not psutil alone (no GPU/temp on Jetson). Fallback: psutil for CPU, GPU/temp shown "N/A" on non-Jetson development machines.
* **Inference stack:**
  * **YOLO:** Exported once to TensorRT FP16 format — `YOLO("best.pt").export(format="engine", half=True)` — loaded as `best.engine` via Ultralytics API for low-overhead native execution target `cuda:0` (typically 2–4× FPS drop-in acceleration on Orin).
  * **Frame-skip + tracker:** Run YOLO every N frames (default N=3); between inferences, propagate bounding boxes with a lightweight tracker (e.g., ByteTrack) or localized OpenCV tracker. Video stays smooth; GPU load drops ~60–70%.
  * **Capture:** Hardware-accelerated GStreamer pipeline natively reading the H.265 core via native GStreamer endpoints (`v4l2src` / `nvarguscamerasrc`).
  * **JPEG encoding:** Hardware encoder (`nvjpeg` via `jetson-utils`) when available, running on a dedicated hardware engine separate from CUDA cores; falls back to `cv2.imencode`.
* **Communication:**
  * **Video:** MJPEG over HTTP (`multipart/x-mixed-replace`) served via a clean endpoint. **Decision locked:** Bound strictly to **640×480, JPEG quality 60, capped at 15 FPS** to avoid starving memory and inference pipelines. **Encode only while ≥1 client is connected** — no viewer, no encoding cost.
  * **Telemetry/Control:** One bi-directional WebSocket; telemetry JSON broadcasted at **10 Hz**; spectrogram throttled separately (§4.2).

---

## 3. UI/UX Layout & Component Specifications

Dark-themed `grid-cols-12` layout, 1920×1080. All numeric readouts use `font-variant-numeric: tabular-nums` (prevents jitter as digits change). **Projector note:** dark themes wash out on projectors — include a high-contrast/brightness toggle for defense day.

### Display Modes (single hotkey `P` toggles)

* **Debug Mode:** full layout below (all components).
* **Presentation Mode:** video + radar enlarged, header retained; sliders, logs, and spectrogram hidden. This fulfills the "dual-purpose" goal explicitly.

### Component A: Header & System Master Control

* Grid of status cards with green/red LED-style indicators + prominent **START / STOP** master switch ('Stop' pauses inference loops and sensor polling via WebSocket command; 'Start' re-initializes).
* **Monitored:** Camera (Connected/Disconnected + FPS) · Mic Array (Active/Muted + dB) · Jetson (CPU %, GPU %, °C via jtop) · **System State** (FSM State machine §5) · **Link** (WebSocket Connected/Reconnecting) · **End-to-End Latency (ms)** — computed live as `render_timestamp - sensor_ts` ( examiners love this).
* **Alarm mute toggle** (browser audio cue on TARGET_CONFIRMED / ENGAGED).

### Component B: Main Video & Vision Analytics (Left — 7/12)

* Processed YOLO frame with bounding boxes, labels, tracking vectors.
* **YOLO Confidence Indicator:** dynamic horizontal progress bar (0.0–1.0).
* **Toggles:** bounding boxes ON/OFF (for performance debugging).
* **Camera loss:** >2 s without a frame → "NO SIGNAL" placeholder (never a frozen frame), camera card turns red.

### Component C: Audio Analytics & DSP (Right — 5/12)

* **Audio CNN Confidence gauge:** Dynamic probability layout (Drone vs. Background Noise).
* **Live Spectrogram:** drawn as **`ImageData` directly on a Canvas** (not a Chart.js chart) from the throttled matrix in §4.2.
* **Waveform:** Chart.js line chart, decimated samples from the 10 Hz telemetry, animations off.

### Component D: Spatial Awareness Map (Bottom Left — 6/12)

* HTML5 Canvas, drawing a 2D semi-circular polar plot representing the current hardware Field of View (FOV).
* **Acoustic DOA:** Azimuth mapping only. Drawn as a dynamic highlighted sector at a fixed radius. Sector width corresponds directly to the angular resolution of the physical circular microphone array: **Array Size: 4-Microphone Uniform Linear/Circular Array; Angular Resolution: ±15.0 Degrees** (Sector Width = 30° arc).
* **Visual Detections:** Maps the bounding box center to spatial coordinates based on the hardware optics: **Camera Horizontal FOV: 90.0 Degrees**. Visual markers are overlaid at a second fixed radius.
* **Fusion:** Overlapping acoustic and visual sector coordinates within the DOA error margin are highlighted with a glowing red pulse animation.
* **Trails:** Implements frame-refresh physics maintaining a **3-second alpha-fading trail** behind historical targets to show vector paths clearly.

### Component E: Hyperparameter Control Panel & Live Logs (Bottom Right — 6/12)

Every slider has a `title` tooltip explaining its engineering purpose.

| # | Slider / Control | Range / Type | Tooltip |
|---|---|---|---|
| 1 | YOLO Confidence Threshold | 0.0–1.0 | "Minimum confidence score required for YOLOv8n to flag a visual bounding box as a drone." |
| 2 | Acoustic CNN Threshold | 0.0–1.0 | "Probability cutoff for the Audio CNN classifier to trigger an acoustic detection alarm." |
| 3 | Sensor Fusion Window | 100–2000 ms | "The time window in which visual and acoustic detections must overlap to trigger a combined system alarm." |
| 4 | Audio Gain / Pre-amp | 1x–10x | "Digital multiplier applied to raw audio streams before spectrogram generation." |
| 5 | NMS Threshold | 0.1–0.9 | "IoU overlap threshold for filtering duplicate YOLO bounding boxes on the same target." |
| 6 | Slew Window Timeout | 1.0–5.0 s | "The duration in seconds the system waits for mechanical stabilization of the PTZ camera before initiating YOLO tracking loop." |
| 7 | Lost Lock Timeout | 1.0–10.0 s | "The time window the system tolerates an obscured visual track before declaring a false alarm and reverting to acoustic searching." |
| 8 | Invert PTZ Camera Vector | Toggle Switch | "Inverts the camera pan/tilt axes to dynamically compensate for the upside-down physical installation of the PTZ housing on the tactical rig." |
| 9 | Acoustic Buffer Frame Size | 128–1024 samples | "Controls the length of the audio frame buffer used for generating the raw spectrogram before feeding it into the Acoustic CNN classifier." |

* **Settings behavior (mandatory):** on connect the server sends `settings_snapshot` and the UI initializes from it (refresh never desyncs); every change is acknowledged with `settings_ack` (UI shows "pending" until acked; server clamps out-of-range values); settings persist to `config.json` and reload on restart.
* **Live System Log:** scrollable monospaced terminal, last 50 lines, color-coded INFO/WARN/ERROR, fed by a custom `logging.Handler` pushing over the WebSocket.

### Component F: Detection Event History (tab/strip below E)

* Last 100 events: **timestamp · modality (Visual/Acoustic/Fused) · class · confidence · azimuth**. Fused rows highlighted. **Export CSV** button (client-side).

---

## 4. WebSocket Payload Specifications

### 4.1 Client → Server

```json
{
  "command": "update_settings",
  "system_active": true,
  "yolo_threshold": 0.55,
  "audio_threshold": 0.70,
  "fusion_window_ms": 500,
  "audio_gain": 2.5,
  "nms_threshold": 0.45,
  "slew_timeout_s": 2.0,
  "lost_lock_timeout_s": 4.0,
  "invert_camera_vector": true,
  "audio_buffer_size": 512,
  "draw_boxes": true
}

```

Additional commands: `{"command": "set_mode", "mode": "live" | "replay", "file": "<recording>"}` · `{"command": "inject_demo_detection"}` (see §9).

### 4.2 Server → Client

**Telemetry — 10 Hz:**

```json
{
  "type": "telemetry",
  "ts": 1751712000.123,
  "state": "SCANNING",
  "camera": { "connected": true, "fps": 15.0 },
  "audio": { "active": true, "db": -32.5 },
  "jetson": { "cpu": 61.0, "gpu": 74.0, "temp_c": 58.5 },
  "vision": { "confidence": 0.62, "detections": [ { "cls": "drone", "conf": 0.62, "bbox": [100, 120, 50, 80], "azimuth_deg": 12.5 } ] },
  "acoustic": { "confidence": 0.71, "doa_deg": 15.0 },
  "waveform": [/* ~128 decimated samples */],
  "sensor_ts": 1751712000.093
}

```

`sensor_ts` = capture timestamp; frontend computes end-to-end latency as `render_time - sensor_ts` (clocks are same-LAN; for the direct-cable setup skew is negligible, or sync via NTP once at boot).

**Spectrogram — separate message, ≤3 Hz:**
Derived directly from the pre-processing loop: **Audio Input Window Length: 1000 ms** (1-second sliding audio window passed to the pipeline).

```json
{ "type": "spectrogram", "rows": 64, "cols": 48, "data": "<base64 uint8 matrix>" }

```

**Other types:** `settings_snapshot` (on connect) · `settings_ack` · `log` (`{level, msg, ts}`) · `event` (one detection-history row per new event) · `mode` (`{mode: "live"|"replay"}`).

---

## 5. System State Machine

| State | Characterization & UI Node Feedback | Transition Criteria |
| --- | --- | --- |
| `IDLE` | System paused. Master control deactivated. | `system_active == false`. Exit when START pressed. |
| `SEARCHING` | Blue color node with a breathing pulse animation. Acoustic monitoring active; camera at rest. | Active, nothing above thresholds. Any acoustic detection triggers change. |
| `SLEWING` | Solid high-contrast yellow color node. PTZ moving mechanically to target vector. | Acoustic confidence ≥ threshold. Triggers PTZ drive to `acoustic_doa_azimuth`. Holds state until `slew_timeout_s` completes. |
| `TRACKING` | Orange color node blinking at 2Hz. Localized YOLO visual exploration. | Enters after `slew_timeout_s`. YOLO scans. If no track within `lost_lock_timeout_s` -> reverts to `SEARCHING`. |
| `ENGAGED` | Rapidly pulsing red node + global screen border flashing alert banner + audio alert. | Visual and acoustic confirm within `fusion_window_ms` and azimuths agree within ±15°. Reverts to `TRACKING` if lock is lost. |
| `DEGRADED` | Critical component failure or disconnected device. System alert box displayed. | Hardware disconnect or loop exception caught. Recovers to previous state upon reconnection. |

---

## 6. Failure Handling

* **WebSocket drop:** frontend auto-reconnects with exponential backoff (1→2→4→max 10 s); link card shows "Reconnecting…"; controls disabled while down; on reconnect server re-sends `settings_snapshot`.
* **Camera loss:** state → `DEGRADED`, "NO SIGNAL" placeholder overlay, acoustic pipeline keeps running. **Mic loss:** symmetric.
* **Pipeline exception:** caught per loop, logged ERROR (visible in live log), loop restarts after 2 s. One crashing model never takes down the server or the other pipeline.
* **Multiple clients:** telemetry is broadcast to all; settings are last-writer-wins (documented, not prevented). Video encoding runs only while ≥1 MJPEG client is connected.

---

## 7. Pipeline Abstraction (critical for sim → real handoff)

All inference sits behind fixed interfaces so simulated implementations swap for real ones **without touching server or frontend**:

```python
class VisionPipeline(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest_frame(self) -> bytes | None          # JPEG bytes for MJPEG
    def latest_result(self) -> VisionResult | None  # detections, conf, azimuths, sensor_ts
    def update_params(self, yolo_threshold: float, nms_threshold: float, draw_boxes: bool) -> None: ...

class AudioPipeline(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest_result(self) -> AudioResult | None   # conf, doa_deg, db, waveform, spectrogram, sensor_ts
    def update_params(self, audio_threshold: float, audio_gain: float) -> None: ...

```

Deliverables include `SimulatedVisionPipeline` / `SimulatedAudioPipeline`. The real implementations (TensorRT YOLO with frame-skip+tracker; Audio CNN) implement the same protocols later. **Record/Replay (§9) is simply a third implementation of the same protocols.**

---

## 8. Performance Budget (hard rules)

1. No browser on the Jetson (§1).
2. MJPEG caps: 640×480 / q60 / ≤15 FPS; encode only with a connected viewer.
3. Chart.js `animation: false`; spectrogram via Canvas `ImageData`, never a chart.
4. Telemetry 10 Hz, spectrogram ≤3 Hz — no other periodic messages.
5. YOLO deployed as TensorRT FP16 engine; frame-skip N≥3 with tracker in between.
6. All backend loops `async`, no busy-waiting; sensor reads in threads/executors if blocking.

---

## 9. Record / Replay Mode (highest-value practical feature)

* **Record:** while live, optionally write timestamped sensor results (and raw frames at reduced rate) to a file.
* **Replay:** `ReplayVisionPipeline` / `ReplayAudioPipeline` (same §7 protocols) stream a recording back in real time. The entire GUI behaves identically — it cannot tell replay from live.
* **Why:** (a) if the live demo fails at the defense — and live demos fail exactly then — a perfect recording is one command away; (b) threshold tuning becomes repeatable on identical data.
* **Demo injection:** hidden hotkey (`Shift+D`) sends `inject_demo_detection` — the backend fabricates one plausible fused detection event. Backup for when no drone is airborne. A small "SIM" badge appears on screen while any injected/replayed data is shown (academic honesty + avoids examiner confusion).

---

## 10. Deployment

* **systemd service** on the Jetson: backend auto-starts on boot, restarts on crash (`Restart=always`). No SSH-in-front-of-audience moments.
* Static IP configured for the direct-Ethernet defense setup; hotspot profile pre-configured as backup (§1).
* Defense-day checklist: boot Jetson → service auto-runs → connect cable → open `http://192.168.1.10:8000` on laptop → verify latency indicator ≤ ~150 ms → keep a recording loaded for replay.

---

## 11. Open Items (Resolved)

1. **Mic count + expected DOA angular resolution:** 4-Microphone Array, angular resolution bounds configured to ±15.0° (30° highlighted detection sector).
2. **Camera horizontal FOV in degrees:** 90.0° Horizontal field of view mapped precisely to radar overlay boundaries.
3. **Audio CNN input window length:** 1000 ms sliding processing frame window.
4. **Final Jetson model:** Nvidia Jetson Orin Nano ($640 \times 480$, q60, capped at 15 FPS).

---

## 12. Implementation Prompt Instructions for Claude

> Based on this specification, generate a fully functional prototype in two parts:
> 1. **`main.py`** — FastAPI backend that: serves the static frontend and an MJPEG endpoint (640×480, q60, ≤15 FPS, **encoding only while a client is connected**); runs background async loops behind the `VisionPipeline`/`AudioPipeline` protocols (§7) using simulated implementations active only when `system_active` is true; implements the state machine (§5) and failure handling (§6); broadcasts messages exactly per §4.2 (telemetry 10 Hz incl. `sensor_ts`, spectrogram ≤3 Hz, log lines, events, mode); validates + clamps incoming settings, persists to `config.json`, replies `settings_ack`, sends `settings_snapshot` on connect; supports `set_mode` (live/replay stub) and `inject_demo_detection` (fabricates one fused event and flags SIM).
> 2. **`index.html`** — complete single-file frontend: Tailwind dark theme + high-contrast toggle; grid per §3 including Detection Event History with CSV export; **Presentation/Debug mode toggle on hotkey `P**`; WebSocket auto-reconnect with backoff and disabled controls while down; Chart.js waveform with `animation:false`; spectrogram via Canvas ImageData; polar radar with DOA sectors, bbox-azimuth markers, fusion highlight, and 3 s fading trails; live end-to-end latency readout from `sensor_ts`; all Component E sliders with `title` tooltips, initialized from `settings_snapshot`, pending-until-`settings_ack`; `tabular-nums` on all numeric readouts; SIM badge when replay/injected data is displayed; hidden `Shift+D` demo-injection hotkey.
> 
> 
> The simulated pipelines must be drop-in replaceable by real TensorRT/CNN implementations with zero changes to server logic or frontend.

```

```