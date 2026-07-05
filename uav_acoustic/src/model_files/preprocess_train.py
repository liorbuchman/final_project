import pandas as pd
import numpy as np
import librosa
import hashlib
from pathlib import Path

# === Path Configuration ===
# Source folder for clean background noises (non-drone)
NOT_DRONE_RAW = Path(r"C:\final_project\uav_acoustic\data\raw\not_drone")
# Source folder for augmented drone recordings (output of the augmentation script)
AUGMENTED_BASE_DIR = Path(r"C:\final_project\uav_acoustic\data\raw\augmented_noisy")
# Destination folder for the processed .npy files
PROCESSED_BASE_DIR = Path(r"C:\final_project\uav_acoustic\data\processed")
# Folder where the labeling and metadata CSV will be saved
METADATA_DIR = Path(r"C:\final_project\uav_acoustic\data\labels")

# === Processing Parameters ===
SR = 16000          # Target Sample Rate
CHUNK_SECS = 1.0    # Duration of each segment for the model input
N_MELS = 128        # Number of Mel bands
N_FFT = 1024        # FFT window size
HOP = 256           # Hop length (75% overlap)
SKIP_IF_EXISTS = True # Skip processing if the .npy file already exists

def get_split_deterministically(filename: str, ratios=(80, 10, 10)) -> str:
    """
    Assigns a file to train/val/test split based on its original filename.
    This ensures that all noisy versions of the same original recording 
    end up in the same split to prevent data leakage.
    """
    # Extract the original filename cleanly for both drone and non-drone files
    if '_noise_' in filename:
        base_name = filename.split('_noise_')[0]
    else:
        base_name = filename.replace('.wav', '')
    
    # Generate a stable hash using SHA1
    h = hashlib.sha1(base_name.encode("utf-8")).hexdigest()
    val = int(h[:9], 16) % 100
    
    t, v, s = ratios
    if val < t:
        return "train"
    elif val < t + v:
        return "val"
    else:
        return "test"

def compute_logmel(y, sr, fixed_length=63):
    """
    Computes a Log-Mel Spectrogram from a raw audio signal and forces 
    a fixed time-dimension length to match real-time deployment constraints.
    """
    # Compute the Mel Spectrogram using exact matching parameters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, 
                                       hop_length=HOP, n_mels=N_MELS, power=2.0)
    
    # Convert power spectrogram to decibel units (log scale)
    spec_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    
    # Force exact frame width parity (63 time bins) to strictly match the RT script
    if spec_db.shape[1] < fixed_length:
        spec_db = np.pad(spec_db, ((0, 0), (0, fixed_length - spec_db.shape[1])))
    else:
        spec_db = spec_db[:, :fixed_length]
        
    return spec_db

def process_folder(input_path, snr_label, label_name):
    """
    Iterates through a folder of .wav files, splits them into 1-second chunks,
    calculates spectrograms, and saves them as .npy files.
    """
    if not input_path.exists():
        print(f"[WARN] Path not found: {input_path}")
        return []

    files = list(input_path.glob("*.wav"))
    records = []
    chunk_samples = int(SR * CHUNK_SECS) # 16,000 samples for 1 second
    
    for path in files:
        # Determine split (train/val/test) for this specific source file
        split = get_split_deterministically(path.name)
        
        try:
            # Load audio at the specified Sample Rate
            y, _ = librosa.load(path, sr=SR)
            
            # Error handling: Skip files that are shorter than the FFT window
            if len(y) < N_FFT:
                print(f"[SKIP] File too short ({len(y)} samples): {path.name}")
                continue

            # Calculate how many 1-second segments can be extracted
            num_chunks = len(y) // chunk_samples
            
            if num_chunks == 0:
                # Skip files shorter than 1 second
                continue

            for i in range(num_chunks):
                start = i * chunk_samples
                end = start + chunk_samples
                y_chunk = y[start:end]

                # Normalize amplitude if there is content in the chunk
                if np.max(np.abs(y_chunk)) > 1e-8:
                    y_chunk /= np.max(np.abs(y_chunk))

                # Define the output path for the segment
                out_dir = PROCESSED_BASE_DIR / snr_label / split / label_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{path.stem}_seg{i:03d}.npy"
                
                # Metadata record for the CSV
                record = {
                    "filename": path.name,
                    "label": label_name,
                    "snr": snr_label,
                    "split": split,
                    "output_path": str(out_path.relative_to(PROCESSED_BASE_DIR))
                }

                if SKIP_IF_EXISTS and out_path.exists():
                    records.append(record)
                    continue
                
                # Compute and save the spectrogram
                mel = compute_logmel(y_chunk, SR)
                np.save(out_path, mel)
                records.append(record)
                
        except Exception as e:
            print(f"[ERROR] Failed processing {path.name}: {e}")
            
    return records

# === Main Execution Logic ===
if __name__ == "__main__":
    # Ensure the metadata folder exists
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    all_metadata = []

    # Step 1: Process background noise files (not_drone)
    # These are stored in a 'common' folder as they don't change with SNR
    print("\n[STEP 1] Processing 'not_drone' (Common data)...")
    all_metadata.extend(process_folder(NOT_DRONE_RAW, "common", "not_drone"))

    # Step 2: Process augmented drone recordings
    # This iterates through SNR folders (e.g., SNR_15, SNR_5...)
    print("\n[STEP 2] Processing augmented drone folders...")
    snr_folders = [d for d in AUGMENTED_BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("SNR_")]
    
    for snr_dir in snr_folders:
        print(f"Processing drones in {snr_dir.name}...")
        # Note: We pass snr_dir.name to create a structured output folder
        all_metadata.extend(process_folder(snr_dir, snr_dir.name, "drone"))

    # Step 3: Save metadata for documentation and model loading
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        df.to_csv(METADATA_DIR / "dataset_metadata.csv", index=False)
        print(f"\n[DONE] Preprocessing finished! Total segments created: {len(df)}")
    else:
        print("\n[INFO] No files were processed.")