from pathlib import Path
import json
import hashlib
import numpy as np
import soundfile as sf
import librosa

# === Configuration ===
RAW_DIR = Path(r"C:\final_project\uav_acoustic\data\raw")
PROCESSED_DIR = Path(r"C:\final_project\uav_acoustic\data\processed")
SPLIT_JSON = Path(r"C:\final_project\uav_acoustic\data\labels\split_map.json")

SR = 16000
CHUNK_SECS = 1.0
N_MELS = 128
N_FFT = 1024
HOP = 256
PAD_SHORT = False
SKIP_IF_EXISTS = True

# === Split creation ===
def stable_hash_int(s: str) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:9], 16) % 1_000_000_000

def choose_split(key: str, ratios=(80,10,10)) -> str:
    t, v, s = ratios
    total = t + v + s
    r = stable_hash_int(key) % total
    if r < t:
        return "train"
    elif r < t + v:
        return "val"
    else:
        return "test"

def create_split():
    if SPLIT_JSON.exists():
        print("[INFO] Split file already exists, skipping.")
        return
    files = list(RAW_DIR.rglob("*.wav"))
    if not files:
        print("[WARN] No .wav files found.")
        return
    split_map = {}
    for f in files:
        key = str(f.resolve())
        split = choose_split(key)
        label = f.parent.name
        split_map[str(f.resolve())] = {"split": split, "label": label}
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_JSON, "w") as f:
        json.dump(split_map, f, indent=2)
    print(f"[OK] Split map saved: {SPLIT_JSON}")
    print(f"Total files: {len(files)}")

# === Audio processing ===
def secs_to_samples(secs: float, sr: int) -> int:
    return int(round(secs * sr))

def chunk_audio(y, sr, chunk_secs):
    L = secs_to_samples(chunk_secs, sr)
    n = len(y) // L
    return [y[i*L:(i+1)*L] for i in range(n)]

def compute_logmel(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                       hop_length=HOP, n_mels=N_MELS, power=2.0)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)

def process_audio():
    if not SPLIT_JSON.exists():
        print("[ERROR] Split map not found. Run create_split() first.")
        return

    with open(SPLIT_JSON, "r") as f:
        split_map = json.load(f)

    total_segments = 0
    for f_str, meta in split_map.items():
        path = Path(f_str)
        if not path.exists():
            print(f"[WARN] Missing: {path}")
            continue
        split = meta["split"]
        label = meta["label"]

        try:
            y, sr_file = sf.read(str(path))
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr_file != SR:
                y = librosa.resample(y, orig_sr=sr_file, target_sr=SR)
            y = y.astype(np.float32)
            if np.max(np.abs(y)) > 1e-8:
                y /= np.max(np.abs(y))

            L = secs_to_samples(CHUNK_SECS, SR)
            if len(y) >= L:
                segments = chunk_audio(y, SR, CHUNK_SECS)
            elif PAD_SHORT:
                seg = np.zeros(L, dtype=np.float32)
                seg[:len(y)] = y
                segments = [seg]
            else:
                continue

            out_dir = PROCESSED_DIR / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx, seg in enumerate(segments):
                out_path = out_dir / f"{path.stem}_{int(CHUNK_SECS*1000)}ms_seg{idx:04d}.npy" # determine the file name
               
                if SKIP_IF_EXISTS and out_path.exists():
                # print(f"[INFO] Exists, skipping: {out_path.name}")
                    continue
                mel = compute_logmel(seg, SR)
                np.save(out_path, mel)
                total_segments += 1

        except Exception as e:
            print(f"[WARN] Failed on {path}: {e}")

    print(f"[DONE] Total segments saved: {total_segments}")

# === Choose what to run ===
if __name__ == "__main__":
    # Step 1: run once
    # create_split()

    # Step 2: then run this
    process_audio()
