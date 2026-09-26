import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import shutil
import random

# === Path define ===
ESC50_CSV = Path(r"C:\final_project\uav_acoustic\data\background_noise\esc50.csv")
ESC50_AUDIO_DIR = Path(r"C:\final_project\uav_acoustic\data\background_noise\audio")
DRONE_DATA_DIR = Path(r"C:\final_project\uav_acoustic\data\raw\drone")
OUTPUT_BASE_DIR = Path(r"C:\final_project\uav_acoustic\data\raw\augmented_noisy")

# === paramaters ===
SR = 16000
SNR_VALUES = [15, 12, 10, 7, 5, 0, -5] # SNR values for each group
RELEVANT_IDS = [44, 23, 12, 19, 14, 48, 15, 16, 25, 10, 43, 42, 47, 11, 7, 13, 24, 40, 41, 36, 9] # ID's numbers for background noise, 40 is hlicopter and 41 is chainsow

def setup_noise_library(csv_path, src_audio, target_dir, ids):
    """  reading the csv and copy the right file to a new folder for easier access later  """
    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    filtered_df = df[df['target'].isin(ids)]
    
    print(f"[INFO] Copying {len(filtered_df)} noise files...")
    for _, row in filtered_df.iterrows():
        shutil.copy(src_audio / row['filename'], target_dir / row['filename'])
    return list(target_dir.glob("*.wav"))

def mix_audio_with_snr(signal, noise, snr_db):
    """ Mixes signal with noise at a specified SNR level """
    # calculate RMS of both signals
    rms_signal = np.sqrt(np.mean(signal**2)) + 1e-9
    rms_noise = np.sqrt(np.mean(noise**2)) + 1e-9
    
    # calculate target RMS for noise
    # Formula: target_rms_noise = rms_signal / (10^(snr/20))
    target_rms_noise = rms_signal / (10**(snr_db / 20))

    # adjust noise amplitude
    noise = noise * (target_rms_noise / rms_noise)

    # mix signal and noise
    mixed = signal + noise
    
    # normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed /= max_val
        
    return mixed

def process_augmentation():
    df_esc = pd.read_csv(ESC50_CSV)
    noise_to_category = dict(zip(df_esc['filename'], df_esc['category']))
    # prepare noise library
    noise_lib_dir = OUTPUT_BASE_DIR / "noise_library"
    noise_files = setup_noise_library(ESC50_CSV, ESC50_AUDIO_DIR, noise_lib_dir, RELEVANT_IDS)
    
    drone_files = list(DRONE_DATA_DIR.rglob("*.wav"))
    
    for snr in SNR_VALUES:
        print(f"\n[PROCESS] Generating data for SNR: {snr}dB")
        snr_out_dir = OUTPUT_BASE_DIR / f"SNR_{snr}"
        snr_out_dir.mkdir(parents=True, exist_ok=True)
        
        for drone_path in drone_files:
           # load drone audio
            y_drone, _ = librosa.load(drone_path, sr=SR)
            
            # select random noise file
            noise_path = random.choice(noise_files)
            y_noise, _ = librosa.load(noise_path, sr=SR)

            category_name = noise_to_category.get(noise_path.name, "unknown")
            
            # --- handling audio length (independent of length) ---
            if len(y_noise) < len(y_drone):
                # handle case where noise is shorter than drone audio
                repeats = (len(y_drone) // len(y_noise)) + 1
                y_noise = np.tile(y_noise, repeats)[:len(y_drone)]
            else:
                # if noise is longer than drone audio - cut a random segment from it
                start = random.randint(0, len(y_noise) - len(y_drone))
                y_noise = y_noise[start : start + len(y_drone)]
            
            # mix signal and noise
            y_mixed = mix_audio_with_snr(y_drone, y_noise, snr)
            
            # save in clear format: [original_name]_[category_name].wav
            out_filename = f"{drone_path.stem}_noise_{category_name}.wav"
            sf.write(snr_out_dir / out_filename, y_mixed, SR)

    print(f"\n[DONE] All noisy recordings saved under: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    process_augmentation()