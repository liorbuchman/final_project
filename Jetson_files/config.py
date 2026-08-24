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
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "uav_vision", "best_v6.pt")
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
YOLO_LOW_CONF_THRESHOLD = 0.25
YOLO_HIGH_CONF_THRESHOLD = 0.5
SAVE_LAST_FRAME_FLAG = False

# --- Acoustic Settings ---
AUDIO_MODEL_DIR = os.path.join(BASE_DIR, "uav_acoustic")
IDVENDOR = 0x2886
IDPRODUCT = 0x0018
AUDIO_CHANNEL = 0
AUDIO_CLASSIFICATION_THRESHOLD = 0.65
#fft parameters
SAMPLE_RATE = 16000
MEL_HOP_LENGTH = 256
MEL_N_FFT = 1024
MEL_N_MELS = 128
MEL_FIXED_LENGTH = 63
WINDOW_SECS = 1.0
STEP_SECS = 0.2
RESPEAKER_INDEX = 0
SMOOTHING_WINDOW = 1
ENABLE_ENERGY_GATE = True
AUDIO_MIN_RMS_THRESHOLD = 0.025 #enrgey lower threshod -> going to be zero
DOA_SMOOTHING_ALPHA = 0.35         
DOA_MAX_JUMP_DEG = 30.0  
DOA_OUTLIER_CONFIRM_COUNT = 3           

#dsp card function  
RESPEAKER_TUNE_DSP = True #endable DSP tuning for ReSpeaker V3.1
RESPEAKER_DISABLE_AGC = True       
RESPEAKER_DISABLE_NS = True        
RESPEAKER_DISABLE_AEC = True      
RESPEAKER_DISABLE_HPF = True


#Camera Physical Limits & Calibration
MOVE_SPEED = 0.6
MIN_ANGLE = -177.5              
MAX_ANGLE = 177.5 
PAN_RANGE_SOFTWARE = MAX_ANGLE - MIN_ANGLE
PAN_TIME_END_TO_END = 21.17 # for speed of 0.6, measured with calibration tool            
TIME_PER_DEGREE_PAN = PAN_TIME_END_TO_END / PAN_RANGE_SOFTWARE

# Tilt Scanning settings
DEFAULT_ELEVATION_ANGLE = 40
TILT_CHECKPOINTS = [25.0, 35.0, 55.0] 

MIN_TILT = 0.0
MAX_TILT = 70.0
TILT_RANGE_SOFTWARE = MAX_TILT - MIN_TILT
TILT_TIME_END_TO_END = 3.76 # for speed of 0.6, measured with calibration tool
TIME_PER_DEGREE_TILT = TILT_TIME_END_TO_END / TILT_RANGE_SOFTWARE

# --- FSM Tactical Timeouts & Visual Tracking ---
TARGET_LOST_TIMEOUT = 4
VISUAL_LOCK_COOLDOWN = 1.5
KP_PAN = 0.0015
KP_TILT = -0.0015
KD_PAN = 0.0005
KD_TILT = 0.0005
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
        f"nvvidconv flip-method=2 ! video/x-raw(memory:NVMM), width={FRAME_WIDTH}, height={FRAME_HEIGHT} ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true sync=false max-buffers=1"
    )