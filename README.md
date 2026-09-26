# ComradeInBattle:Multi-Modal Drone Detection & Tracking System

## Authors
Lior Buchman - B.Sc. Electrical Engineering, Afeka College

Noy Maymon - B.Sc. Electrical Engineering, Afeka College

## Background
Final Electrical engineering project. The goal is simple to state and hard to do: **hear a
drone, point a camera at it, confirm it visually, and keep the camera locked on it** —
all in real time, on a small embedded computer.

The system fuses two sensors:

| Sensor | What it gives us |
|---|---|
| USB microphone array (ReSpeaker 4-mic) | "Is that a drone?" (audio CNN) + roughly **which direction** the sound comes from (DOA angle) |
| PTZ IP camera (pan/tilt) | A YOLO object detector that finds the drone in the video and a motorized mount that can physically turn toward it |

Neither sensor alone is good enough. The mic hears a drone anywhere in a wide arc but
can't tell you exactly where it is. The camera is precise but only sees a narrow cone
and has no idea where to look. Put together: the mic tells the camera where to point,
the camera confirms and tracks.

Everything runs on an **NVIDIA Jetson Orin Nano**. A separate laptop opens a web
dashboard to watch what the system is doing — the Jetson never runs a browser itself
(it needs all its RAM/GPU for the models).

---

## How it works (the state machine)

The core logic is a 4-state machine in [`Jetson_files/main_system.py`](Jetson_files/main_system.py).
It ticks at 10 Hz on its own thread. The audio and video pipelines run continuously in
parallel and just publish their latest results; the state machine reads those and
decides what the camera motors should do.

```
CALIBRATING ──► SCANNING ──► TRACKING ──► ENGAGED
                   ▲            │  ▲          │
                   └────────────┘  └──────────┘
                   (nothing found)  (lost visual lock)
```

| State | What's happening | How we leave it |
|---|---|---|
| **CALIBRATING** | On startup the camera drives to its mechanical zero and homes both axes so we know exactly where it's pointing. | Calibration finishes → **SCANNING** |
| **SCANNING** | Idle / listening. Only the mic array is working. The camera sits still (and quietly re-homes itself after ~3 min of silence to fight motor drift). | Audio CNN reports "drone" for ~0.4 s straight → **TRACKING** |
| **TRACKING** | We heard something. Pan the camera to the sound's direction, then sweep it up and down through preset tilt angles while YOLO searches each spot for the drone. | YOLO gets a solid visual lock → **ENGAGED**. Nothing seen within `TARGET_LOST_TIMEOUT` (4 s) after the sound stops → back to **SCANNING** |
| **ENGAGED** | We can see the drone. A PD control loop drives the pan/tilt motors to keep the drone centered in the frame as it moves. | Visual lock lost for `VISUAL_LOCK_COOLDOWN` (2.5 s) → back to **TRACKING** |

All the timeouts, thresholds, speeds and controller gains live in one place:
[`Jetson_files/config.py`](Jetson_files/config.py).

> The dashboard also shows two states that only exist in the GUI: `IDLE` (system
> stopped by the operator) and `DEGRADED` (a sensor disconnected). These are not part
> of the real state machine.

---

## The models

**Acoustic — `SmallCNN`** ([`Jetson_files/uav_acoustic/model.py`](Jetson_files/uav_acoustic/model.py))
A small 3-block convolutional net. Input is a 1-second **log-mel spectrogram**
(16 kHz audio, 128 mel bands); output is a 2-class score: *drone* vs *background noise*.
A new 1 s window is classified every 0.2 s. An energy gate skips silent buffers so we
don't waste GPU on nothing.

**Direction of arrival (DOA)** comes straight from the ReSpeaker's built-in DSP, then
gets smoothed and outlier-filtered in software (`acoustic_processor.py`) so a single
bad angle reading can't yank the camera around.

**Vision — YOLOv8n** ([`Jetson_files/uav_vision/optical_processor.py`](Jetson_files/uav_vision/optical_processor.py))
Fine-tuned on drone imagery, exported once to a **TensorRT FP16 engine** for fast
inference on the Jetson GPU. **ByteTrack** keeps a consistent track ID on the locked
target so the lock doesn't hop to a bird or a distractor. A two-threshold scheme
(low conf to keep a track, high conf to acquire one) stops junk detections from
latching a permanent "lock".

---

## Repository layout

```
Jetson_files/            ← the real-time system that runs ON the Jetson
  main_system.py         ← entry point: starts all threads + the state machine
  config.py              ← every tunable parameter
  inference_benchmark.py ← standalone on-device latency benchmark (acoustic/vision/both)
  benchmark_results/     ← recorded benchmark runs (CSV + JSON)
  uav_acoustic/          ← audio capture, CNN inference, DOA, ReSpeaker LED driver
  uav_vision/            ← YOLO detector + ONVIF PTZ camera control

gui_dashboard/           ← web dashboard (FastAPI + a single-file React page)
  main.py                ← the server; runs simulated sensors by default
  hardware_bridge.py     ← hooks into the LIVE Jetson system read-only, no edits to Jetson_files/
  templates/              ← index.html (operator view), manual_test.html (engineering view)

pc_dev/                  ← DEV ONLY, runs on a normal PC (not the Jetson)
  uav_acoustic/          ← dataset cleaning, augmentation, CNN training
  uav_vision/            ← camera/PTZ calibration + DOA-tracking experiments

archive/                 ← retired code (old ReSpeaker vendor firmware, earlier system
                            versions) — git-ignored, kept locally for reference only
```

Datasets, recordings and trained model weights are **not** in the repo (see
[`.gitignore`](.gitignore)) — they're too big. You need to supply/train your own:

* `Jetson_files/uav_vision/models/best_v640.engine` — the TensorRT YOLO engine (YOLOv8n)
* `Jetson_files/uav_acoustic/models/` — the trained CNN checkpoint + normalization stats

---

## Running it

### 1. The detection system (on the Jetson)

```bash
cd Jetson_files
python3 main_system.py
```

Needs: a working GStreamer/NVDEC pipeline to the camera's RTSP stream, the ReSpeaker
plugged in, and both model files present. Camera IP / credentials are set at the top of
`config.py`. Logs are written per-session under `Jetson_files/logs/`.

The Jetson's Python environment is **not** created from the `configs/environment.yml`
files — those are for a normal PC. See the note at the end of section 3.

### 2. The dashboard (on a laptop, same network as the Jetson)

```bash
cd gui_dashboard
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `http://<jetson-ip>:8000` in a browser.

* **Default:** the server runs *simulated* audio/video pipelines — handy for developing
  the UI on any laptop with no hardware attached.
* **Live:** start it with `HARDWARE_MODE=1` on the Jetson and it attaches to the running
  `main_system.py` and shows real telemetry, video and model outputs.

GPU % and temperature in the dashboard come from `jetson-stats` (`pip install jetson-stats`);
on a normal PC they just show "N/A".

### 3. Training / experiments (any normal computer)

The `pc_dev/uav_acoustic/` and `pc_dev/uav_vision/` folders (**not** the copies inside
`Jetson_files/`) are the development code. Each has its own self-contained conda
environment so anyone can recreate exactly what's needed:

```bash
conda env create -f pc_dev/uav_acoustic/configs/environment.yml   # -> env "drone-acoustic"
conda env create -f pc_dev/uav_vision/configs/environment.yml     # -> env "drone-vision"
conda activate drone-acoustic     # or drone-vision, depending which part you're working on
```

They're two separate environments (different names) — pick the one for the side you're
touching. Both install CPU-only builds, so no CUDA/GPU setup is required to run them.

> `pc_dev/uav_acoustic/configs/` and `pc_dev/uav_vision/configs/` are git-ignored, so
> `environment.yml` is **not** included in a fresh clone — build it yourself (or ask
> whoever has a local copy) before running the command above.

Acoustic training pipeline: `pc_dev/uav_acoustic/src/model_files/`
(`cleaning.py` → `augmentation.py` → `preprocess_train.py` → `train.py`).
YOLO is trained with the standard Ultralytics CLI/API on a labelled drone dataset, then
exported to a TensorRT engine on the Jetson: `YOLO("best.pt").export(format="engine", half=True)`.

> ⚠️ **The environment files are for a development PC, not the Jetson.** The Jetson Orin
> Nano runs its own hand-built environment: PyTorch/torchvision from NVIDIA's
> JetPack-specific `aarch64` wheels, TensorRT and the GStreamer plugins from JetPack
> itself, an OpenCV build compiled *with* GStreamer support, `jetson-stats`, and system
> `PortAudio` / `libusb` packages (plus a udev rule for the ReSpeaker). Running
> `conda env create` or `pip install -r requirements.txt` on the Jetson will **not**
> give you a working runtime — the versions have to match the installed JetPack/L4T
> release.

---

## Hardware

* NVIDIA Jetson Orin Nano (unified CPU/GPU memory — this is why the dashboard is remote)
* Seeed ReSpeaker USB 4-Mic Array (v3.1), used for classification **and** DOA (±15° resolution)
* ONVIF PTZ IP camera, H.265 RTSP video, pan range ±175°, tilt 0–70°
* Laptop for the dashboard, on the same Wi-Fi network as the Jetson
