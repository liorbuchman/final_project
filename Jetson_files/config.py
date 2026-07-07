# config.py
import torch
import os

# --- Global Hardware Configuration ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Base Directory Resolution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Optical Settings ---
RTSP_URL = "rtsp://admin:admin@192.168.1.90:554/live/ch0"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "uav_vision", "best_v3_birds.pt")
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YOLO_CONF_THRESHOLD = 0.6
MOVE_SPEED = 0.2
PAN_MIN, PAN_MAX = -170.0, 170.0
TILT_MIN, TILT_MAX = -90.0, 30.0

# --- Acoustic Settings ---
AUDIO_MODEL_DIR = os.path.join(BASE_DIR, "uav_acoustic")
AUDIO_CLASSIFICATION_THRESHOLD = 0.5
SAMPLE_RATE = 16000
WINDOW_SECS = 1.0
STEP_SECS = 0.5
RESPEAKER_INDEX = 1
SMOOTHING_WINDOW = 1

# --- FSM Tactical Timeouts ---
TARGET_LOST_TIMEOUT = 5.0
DEFAULT_ELEVATION_ANGLE = -90
VISUAL_LOCK_COOLDOWN = 1.5
KP_PAN = 0.0015
KP_TILT = -0.0015
TILT_DIRECTION_INVERSION = -1.0 # check if needed

# --- Hardware-Accelerated GStreamer Pipeline---
def get_gstreamer_pipeline():
    return (
        f"rtspsrc location={RTSP_URL} latency=0 drop-on-latency=true ! "
        "rtph265depay ! h265parse ! "
        "nvv4l2decoder disable-blk-pool=1 ! "
        "nvvidconv ! video/x-raw(memory:NVMM), width=1280, height=720 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )