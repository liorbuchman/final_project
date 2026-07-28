# config.py
import torch
import os

# --- Global Hardware Configuration ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Base Directory Resolution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Optical Settings ---
RTSP_URL = "rtsp://admin:admin@192.168.1.90:554/live/ch0"
CAMERA_IP = '192.168.1.90'
CAMERA_USER = 'admin'
CAMERA_PASS = 'admin'
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "uav_vision", "best_v4.pt")
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YOLO_CONF_THRESHOLD = 0.6

# --- Acoustic Settings ---
AUDIO_MODEL_DIR = os.path.join(BASE_DIR, "uav_acoustic")
AUDIO_CLASSIFICATION_THRESHOLD = 0.5
SAMPLE_RATE = 16000
WINDOW_SECS = 1.0
STEP_SECS = 0.5
RESPEAKER_INDEX = 1
SMOOTHING_WINDOW = 1

# --- Acoustic Targeting & Sweep Settings ---
CAMERA_MOUNT_OFFSET = 0.0      # Calibration: Angle offset between Mic 0° and Camera front
KP_PAN_ACOUSTIC = 0.02         # Proportional gain for acoustic panning speed
SWEEP_FREQUENCY = 1.5          # Frequency of vertical sine wave sweep
SWEEP_SPEED_MULTIPLIER = 0.3   # Max tilt motor speed during search (0.0 to 1.0)

# --- FSM Tactical Timeouts & Visual Tracking ---
TARGET_LOST_TIMEOUT = 5.0
VISUAL_LOCK_COOLDOWN = 1.5
KP_PAN = 0.0015
KP_TILT = -0.0015
TILT_DIRECTION_INVERSION = -1.0 # Inverts tilt axis for upside-down ceiling mounted PTZ

# --- Hardware-Accelerated GStreamer Pipeline ---
def get_gstreamer_pipeline():
    """
    Constructs a highly optimized GStreamer pipeline utilizing NVIDIA's nvvidconv.
    Hardware scales the frame directly to FRAME_WIDTH x FRAME_HEIGHT to save CPU cycles.
    """
    return (
        f"rtspsrc location={RTSP_URL} latency=0 drop-on-latency=true ! "
        "rtph265depay ! h265parse ! "
        "nvv4l2decoder disable-blk-pool=1 ! "
        f"nvvidconv ! video/x-raw(memory:NVMM), width={FRAME_WIDTH}, height={FRAME_HEIGHT} ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true sync=false max-buffers=1"
    )