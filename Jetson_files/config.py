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
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "uav_vision", "models", "best_v9.engine")
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
YOLO_LOW_CONF_THRESHOLD = 0.25
YOLO_HIGH_CONF_THRESHOLD = 0.5
SAVE_LAST_FRAME_FLAG = False

# --- Visual lock stability (OpticalDetector.run_inference) ---
YOLO_LOCK_HOLD_FRAMES = 15            # frames visual_lock is bridged after the target stops being detected at all
YOLO_ID_RELEASE_FRAMES = 12          # frames the locked ByteTrack track-id may go unmatched before the lock is released (position is held, not handed to a distractor, during this grace)
YOLO_LOCK_MAX_LOWCONF_FRAMES = 25    # drop the lock after this many consecutive frames with NO detection at/above YOLO_HIGH_CONF_THRESHOLD - the target has degraded to noise (implements "not just: there is something")

# --- Acoustic Settings ---
AUDIO_MODEL_DIR = os.path.join(BASE_DIR, "uav_acoustic", "models")
IDVENDOR = 0x2886
IDPRODUCT = 0x0018
AUDIO_CHANNEL = 0
AUDIO_CLASSIFICATION_THRESHOLD = 0.65
AUDIO_TRIGGER_ON_CONFIRM_COUNT = 2  # consecutive buffers above threshold required to raise is_triggered (~0.4s at STEP_SECS)
AUDIO_TRIGGER_OFF_CONFIRM_COUNT = 4  # consecutive buffers below threshold required to clear is_triggered (~0.8s at STEP_SECS)
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
AUDIO_RMS_GATE_OFF_THRESHOLD = 0.018   # hysteresis floor: once the energy gate is open it stays open until RMS drops below this (<= AUDIO_MIN_RMS_THRESHOLD). A single sub-threshold buffer no longer force-ends an event / wipes the DOA filter - it feeds the same OFF debounce streak as a low classification score.
DOA_FILTER_RESET_GAP_SECS = 3.0        # a new EVENT START within this many seconds of the previous EVENT END keeps the DOA smoothing/outlier history instead of re-seeding from a raw hardware angle

#dsp card function  
RESPEAKER_TUNE_DSP = True #endable DSP tuning for ReSpeaker V3.1
RESPEAKER_DISABLE_AGC = True       
RESPEAKER_DISABLE_NS = True        
RESPEAKER_DISABLE_AEC = True      
RESPEAKER_DISABLE_HPF = True


#Camera Physical Limits & Calibration
MOVE_SPEED = 0.6  # kept for standalone calibration/test tools (calibrate_axis.py, doa_tracker.py, etc.) - the main pipeline uses PAN_MOVE_SPEED/TILT_MOVE_SPEED below
PAN_MOVE_SPEED = 0.95
TILT_MOVE_SPEED = MOVE_SPEED  # tune independently from PAN_MOVE_SPEED; if changed, TILT_TIME_END_TO_END must be re-measured with the calibration tool since motor speed-to-angle isn't necessarily linear
MIN_ANGLE = -175
MAX_ANGLE = 175 
PAN_RANGE_SOFTWARE = MAX_ANGLE - MIN_ANGLE
PAN_TIME_END_TO_END = 17.66 # for speed of 0.95, measured with calibration tool            
TIME_PER_DEGREE_PAN = PAN_TIME_END_TO_END / PAN_RANGE_SOFTWARE

# Tilt Scanning settings
DEFAULT_ELEVATION_ANGLE = 40
TILT_CHECKPOINTS = [55.0, 15.0, DEFAULT_ELEVATION_ANGLE]

MIN_TILT = 0.0
MAX_TILT = 70.0
TILT_RANGE_SOFTWARE = MAX_TILT - MIN_TILT
TILT_TIME_END_TO_END = 3.8 # for speed of 0.6, measured with calibration tool
TIME_PER_DEGREE_TILT = TILT_TIME_END_TO_END / TILT_RANGE_SOFTWARE

# --- FSM Tactical Timeouts & Visual Tracking ---
TARGET_LOST_TIMEOUT = 4
VISUAL_LOCK_COOLDOWN = 2.5    # seconds ENGAGED tolerates no visual lock before reverting to TRACKING (was 1.5 - widened to stop ENGAGED<->TRACKING flapping)
RE_SEARCH_GRACE_PERIOD = 2.0  # seconds to hold position after entering TRACKING before a non-blocking acoustic macro-scan starts (was 1.0)
VISUAL_ERROR_STALE_SECS = 0.5  # execute_visual_closed_loop halts the motors if the visual error hasn't been refreshed by a real detection within this window (bounds blind slewing during an id-grace bridge)

# --- Non-blocking acoustic search (OpticalDetector.start/step_acoustic_search) ---
SEARCH_TILT_STARE_SECS = 0.8       # dwell time at each vertical macro-scan checkpoint while the camera stares for a YOLO lock
SEARCH_PAN_REPLAN_DEG = 15.0       # while slewing to the acoustic azimuth, re-issue the pan move if a fresh DOA differs from the in-progress target by more than this many degrees (0 disables)
SEARCH_RESCAN_MIN_INTERVAL = 2.0   # minimum seconds between two full macro-scans aimed at (roughly) the same azimuth - stops a tight re-sweep loop on a stationary acoustic target that can't be seen
RETURN_TO_DEFAULT_ELEVATION_ON_LOST = True  # tilt back to DEFAULT_ELEVATION_ANGLE (non-blocking) when a search is abandoned and the FSM returns to SCANNING

# --- Periodic PTZ Re-Home (open-loop position-drift correction) ---
PERIODIC_REHOME_ENABLED = True
PERIODIC_REHOME_IDLE_SECS = 180.0     # continuous time in SCANNING with no acoustic/visual activity before an automatic re-home runs on its own thread
PERIODIC_REHOME_TILT_ENABLED = True   # include the timed TILT homing in the periodic re-home; False re-homes PAN only and leaves TILT untouched

KP_PAN = 0.0015
KP_TILT = -0.0010
KD_PAN = 0.0005
KD_TILT = 0.0001
TILT_DIRECTION_INVERSION = -1.0 # Inverts tilt axis for upside-down ceiling mounted PTZ

# --- PTZ Hardware Comm Guardrails ---
PTZ_MIN_SEND_INTERVAL = 0.15   # floor between real HTTP ContinuousMove sends when velocity changes - protects the camera's embedded HTTP server from PD-loop jitter during ENGAGED (stop commands bypass this)
PTZ_KEEPALIVE_INTERVAL = 1.0   # resend an unchanged, non-zero velocity at least this often - many budget ONVIF cameras auto-stop ContinuousMove a few seconds after the last command if it isn't refreshed
PTZ_HTTP_SLOW_THRESHOLD = 1.5  # seconds; log a warning if a single ContinuousMove HTTP call takes longer than this

# --- FSM Watchdog ---
FSM_WATCHDOG_STALL_THRESHOLD = 2.0  # seconds with no FSM tick before logging a stall warning

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