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
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "uav_vision", "best.pt")
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
YOLO_LOW_CONF_THRESHOLD = 0.2
YOLO_HIGH_CONF_THRESHOLD = 0.6

# --- Acoustic Settings ---
AUDIO_MODEL_DIR = os.path.join(BASE_DIR, "uav_acoustic")
IDVENDOR = 0x2886
IDPRODUCT = 0x0018
AUDIO_CHANNEL = 0
AUDIO_CLASSIFICATION_THRESHOLD = 0.65
SAMPLE_RATE = 16000
WINDOW_SECS = 1.0
STEP_SECS = 0.2
RESPEAKER_INDEX = 1
SMOOTHING_WINDOW = 1
ENABLE_ENERGY_GATE = True
AUDIO_MIN_RMS_THRESHOLD = 0.025 #enrgey lower threshod -> going to be zero
DOA_SMOOTHING_ALPHA = 0.35         
DOA_MAX_JUMP_DEG = 30.0             

#dsp card function  
RESPEAKER_TUNE_DSP = True #endable DSP tuning for ReSpeaker V3.1
RESPEAKER_DISABLE_AGC = True       
RESPEAKER_DISABLE_NS = True        
RESPEAKER_DISABLE_AEC = True      
RESPEAKER_DISABLE_HPF = True


#Camera Physical Limits & Calibration
MOVE_SPEED = 0.65
MIN_ANGLE = -170.0              
MAX_ANGLE = 170.0  
PAN_TIME_END_TO_END = 26            
TIME_PER_DEGREE = 0.075

# Tilt Scanning settings
DEFAULT_ELEVATION_ANGLE = 45

MIN_TILT = 5.0
MAX_TILT = 45.0
TILT_RANGE_SOFTWARE = 85.0
TILT_TIME_END_TO_END = 12
TIME_PER_DEGREE_TILT = TILT_TIME_END_TO_END / TILT_RANGE_SOFTWARE

# --- FSM Tactical Timeouts & Visual Tracking ---
TARGET_LOST_TIMEOUT = 4
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