import sys
import torch
import torch.nn.functional as F
import numpy as np
import sounddevice as sd
import threading
import json
import time
import csv
import librosa 
from datetime import datetime
from pathlib import Path
sys.path.append(r"C:\final_project")
# --- External Hardware Import ---
from uav_acoustic.src.ReSpeaker_files.respeaker_usb_led import ReSpeakerV31Leds

# --- Model Import ---
from uav_acoustic.src.model_files.model import SmallCNN

# === 1. System Configurations ===
SR = 16000
WINDOW_SECS = 1.0
STEP_SECS = 0.5 #
DEVICE = torch.device('cpu')
RESPEAKER_INDEX = 1 
DETECTION_THRESHOLD = 0.5 # Threshold for detection
SMOOTHING_WINDOW = 1  # Number of samples for smoothing

class LEDController:
    def __init__(self):
        self.led = ReSpeakerV31Leds()
        self.drone_detected = False
        self.running = True

    def blink_logic(self):
        while self.running:
            try:
                if not self.drone_detected:
                    self.led.set_mono(0, 0, 50) # Blue (Not Drone)
                    time.sleep(0.5)
                    self.led.off()
                    time.sleep(0.5)
                else:
                    self.led.set_mono(255, 0, 0) # Red (Drone)
                    time.sleep(0.1)
            except:
                break

    def stop(self):
        self.running = False
        time.sleep(0.2)
        try: self.led.off()
        except: pass

# === 2. Processing Functions ===
def load_system(run_dir):
    run_path = Path(run_dir)
    with open(run_path / "norm_stats.json", 'r') as f:
        stats = json.load(f)
    model = SmallCNN(n_classes=2).to(DEVICE)
    checkpoint = torch.load(run_path / "best_model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, stats['mean'], stats['std']

audio_buffer = np.zeros(int(SR * WINDOW_SECS), dtype=np.float32)
buffer_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    new_samples = indata.flatten()
    with buffer_lock:
        audio_buffer = np.roll(audio_buffer, -len(new_samples))
        audio_buffer[-len(new_samples):] = new_samples

def run_pc_test(run_dir, duration_secs=120):
    model, mean, std = load_system(run_dir)
    leds = LEDController()
    led_thread = threading.Thread(target=leds.blink_logic, daemon=True)
    led_thread.start()

    # --- Configuration Parameters to Track ---
    # Use the existing global threshold or define a local one
    threshold = DETECTION_THRESHOLD 
    model_name = Path(run_dir).name
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_filename = results_dir / f"drone_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    log_file = open(log_filename, mode='w', newline='')
    log_writer = csv.writer(log_file)
    
    # --- Updated CSV Header with parameters for later analysis ---
    log_writer.writerow([
        "Timestamp", "Status", "Smooth_P", "RMS", "Raw_P", 
        "Threshold", "Smoothing_Window", "STEP_SECS", "Model_Run"
    ])

    prob_history = []
    stream = sd.InputStream(device=RESPEAKER_INDEX, samplerate=SR, channels=1, 
                            blocksize=int(SR * STEP_SECS), callback=audio_callback)

    print(f"[*] Monitoring Active. Window: {SMOOTHING_WINDOW} | Threshold: {threshold}")
    start_test_time = time.time()
    
    
    
    try:
        with stream:
            while time.time() - start_test_time < duration_secs:
                with buffer_lock:
                    y_chunk = audio_buffer.copy()
                
                rms = np.sqrt(np.mean(y_chunk**2))
                
                # 1. Normalization (Peak)
                max_val = np.max(np.abs(y_chunk))
                if max_val > 1e-8:
                    y_chunk = y_chunk / max_val
                
                # 2. Spectrogram (Librosa parity)
                S = librosa.feature.melspectrogram(y=y_chunk, sr=SR, n_fft=1024, 
                                                   hop_length=256, n_mels=128, power=2.0)
                spec_db = librosa.power_to_db(S, ref=np.max)
                
                # 3. Shape Adjustment
                if spec_db.shape[1] < 63:
                    spec_db = np.pad(spec_db, ((0, 0), (0, 63 - spec_db.shape[1])))
                else:
                    spec_db = spec_db[:, :63]
                
                # 4. Inference
                spec_tensor = torch.from_numpy(spec_db).float()
                spec_norm = (spec_tensor - mean) / std
                
                with torch.no_grad():
                    output = model(spec_norm.unsqueeze(0).unsqueeze(0))
                    probs = torch.softmax(output, dim=1)
                
                p_drone = probs[0][1].item()
                
                # --- [SMOOTHING LOGIC] ---
                # Managing the moving average buffer based on SMOOTHING_WINDOW
                prob_history.append(p_drone)
                if len(prob_history) > SMOOTHING_WINDOW: 
                    prob_history.pop(0)
                avg_p_drone = sum(prob_history) / len(prob_history)
                
                # Detection decision
                is_drone = (avg_p_drone > threshold)
                leds.drone_detected = is_drone
                
                # --- [LABELS & LOGGING] ---
                status = "DRONE" if is_drone else "NOT DRONE"
                curr_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # Save all metrics and configurations to CSV
                log_writer.writerow([
                    curr_ts, 
                    status, 
                    f"{avg_p_drone:.4f}", 
                    f"{rms:.4f}", 
                    f"{p_drone:.4f}",
                    threshold,
                    SMOOTHING_WINDOW,
                    STEP_SECS,
                    model_name
                ])
                
                # Terminal UI with consistent alignment
                color = "\033[91m" if is_drone else "\033[94m" # Red / Blue
                print(f"\r{color}[{status:^10}]\033[0m Smooth_P:{avg_p_drone:.2f} | RMS:{rms:.4f}", end="")
                
                time.sleep(0.05)

    except KeyboardInterrupt: pass
    finally:
        log_file.close()
        leds.stop()
        print(f"\n[DONE] Test results and config saved to: {log_filename}")

if __name__ == "__main__":
    MODEL_PATH = r"C:\final_project\uav_acoustic\models\run_20260123_104930"
    run_pc_test(MODEL_PATH)