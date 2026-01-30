# Dual-Modal Drone Detection & Tracking System (Intermediate Phase)

This repository contains the ongoing development of an autonomous drone detection and tracking system using acoustic and visual sensors. The project is currently in the **Intermediate Stage**, with a focus on model development, data processing pipelines, and sensor interfacing.

## 📍 Project Status
* [x] **Acoustic Subsystem:** Data collection, augmentation, and initial model training complete.
* [x] **Vision Subsystem:** YOLOv3-based bird/drone detection models trained.
* [x] **Sensor Interfacing:** Initial drivers for ReSpeaker 4-mic array and IP camera control implemented.
* [ ] **System Integration:** Hardware deployment to NVIDIA Jetson (Planned).

## 📂 Project Structure
The project is divided into two main modules:

### 1. UAV Acoustic (`/uav_acoustic`)
Focuses on drone detection via sound signatures.
* `data/`: Contains raw and processed audio, including background noise and labels.
* `src/model_files/`: Core logic for training (`train.py`), model architecture (`model.py`), and real-time preprocessing (`preprocess_RT.py`).
* `src/ReSpeaker_files/`: Hardware-specific scripts like `find_mic.py` and LED control.

### 2. UAV Vision (`/uav_vision`)
Focuses on visual confirmation and tracking.
* `models/`: Contains trained weights such as `best_v3_birds.pt`.
* `src/`: Includes camera control (`camera_AC.py`), model inference, and `wsdl` configurations for ONVIF communication.

## 🛠 Hardware Integration (Planned)
* **Computing:** NVIDIA Jetson (Edge AI).
* **Audio:** ReSpeaker 4-Mic Array (USB) - *Firmware and DFU tools included in root*.
* **Visual:** PTZ Camera with ONVIF support.

## ⚙️ Development Setup
The current environment is managed via Conda/Pip:
1. **Environment:** Use `uav_acoustic/configs/environment.yml` to recreate the environment.
2. **Firmware:** The `6_channels_firmware.bin` and `dfu.py` are provided for microphone array configuration.

