#!/usr/bin/env python3
"""
Training script for SmallCNN on log-mel .npy files.
- Loads processed data from processed/<split>/<label>/... .npy
- Discovers classes from train/ subfolders
- Computes/loads dataset-wide mean/std for normalization
- Trains, validates, saves best checkpoint, evaluates on test

Run:
  pip install torch torchvision torchaudio numpy
  python train.py
"""
from pathlib import Path
from typing import List, Tuple, Dict
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import SmallCNN

# =====================
# CONFIG
# =====================
PROCESSED_DIR = Path(r"C:\final_project\uav_acoustic\data\processed")
CHECKPOINT_DIR = Path(r"C:\final_project\uav_acoustic\models")

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2  # keep small on Windows

# Normalization stats file (computed once on a subset of train)
NORM_STATS_JSON = CHECKPOINT_DIR / "norm_stats.json"
MAX_FILES_FOR_NORM = 2000

# =====================
# Utils
# =====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_label_dirs(split_dir: Path) -> List[Path]:
    out = []
    if not split_dir.exists():
        return out
    for d in split_dir.iterdir():
        if d.is_dir() and any(d.rglob("*.npy")):
            out.append(d)
    return sorted(out)


def discover_classes(root: Path) -> Tuple[List[str], Dict[str, int]]:
    train_dir = root / "train"
    labels = [p.name for p in list_label_dirs(train_dir)]
    labels.sort()
    class_to_idx = {c: i for i, c in enumerate(labels)}
    return labels, class_to_idx


def collect_files(split_root: Path, class_to_idx: Dict[str, int]):
    items: List[Tuple[Path, int]] = []
    for cls, idx in class_to_idx.items():
        cdir = split_root / cls
        if not cdir.exists():
            continue
        for f in cdir.rglob("*.npy"):
            items.append((f, idx))
    return items

class MelNPYDataset(Dataset):
    def __init__(self, items: List[Tuple[Path,int]], mean: float, std: float):
        self.items = items
        self.mean = float(mean)
        self.std = float(std if std > 1e-8 else 1.0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        path, y = self.items[i]
        mel = np.load(path)  # [MELS, TIME]
        x = torch.from_numpy(mel).float()
        x = (x - self.mean) / self.std
        x = x.unsqueeze(0)  # [1, MELS, TIME]
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# =====================
# Norm stats
# =====================

def compute_mean_std(files: List[Path], max_files: int = 2000):
    if len(files) == 0:
        return 0.0, 1.0
    sel = files if len(files) <= max_files else random.sample(files, max_files)
    vals = []
    for p in sel:
        a = np.load(p).astype(np.float32)
        vals.append(a)
    flat = np.concatenate([v.reshape(-1) for v in vals], axis=0)
    mean = float(flat.mean())
    std = float(flat.std() + 1e-8)
    return mean, std

# =====================
# Train / Eval
# =====================

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        bsz = x.size(0)
        total_loss += float(loss.item()) * bsz
        total_acc  += accuracy_from_logits(logits, y) * bsz
        n += bsz
    return total_loss / max(n,1), total_acc / max(n,1)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            bsz = x.size(0)
            total_loss += float(loss.item()) * bsz
            total_acc  += accuracy_from_logits(logits, y) * bsz
            n += bsz
    return total_loss / max(n,1), total_acc / max(n,1)

# =====================
# Main
# =====================

def main():
    print("Setting seed...")
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Discovering classes from train/ ...")
    classes, class_to_idx = discover_classes(PROCESSED_DIR)
    if not classes:
        print("[ERROR] No classes found under train/. Did you run preprocessing?")
        return
    print(f"Classes ({len(classes)}): {classes}")

    train_items = collect_files(PROCESSED_DIR / "train", class_to_idx)
    val_items   = collect_files(PROCESSED_DIR / "val", class_to_idx)
    test_items  = collect_files(PROCESSED_DIR / "test", class_to_idx)
    print(f"Train files: {len(train_items)} | Val files: {len(val_items)} | Test files: {len(test_items)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if NORM_STATS_JSON.exists():
        stats = json.loads(NORM_STATS_JSON.read_text(encoding="utf-8"))
        mean, std = stats.get("mean", 0.0), stats.get("std", 1.0)
        print(f"Loaded norm stats: mean={mean:.4f}, std={std:.4f}")
    else:
        print("Computing normalization stats from train files (subset)...")
        train_paths = [p for (p, _) in train_items]
        mean, std = compute_mean_std(train_paths, MAX_FILES_FOR_NORM)
        NORM_STATS_JSON.write_text(json.dumps({"mean": mean, "std": std}, indent=2), encoding="utf-8")
        print(f"Saved norm stats: mean={mean:.4f}, std={std:.4f}")

    train_ds = MelNPYDataset(train_items, mean=mean, std=std)
    val_ds   = MelNPYDataset(val_items,   mean=mean, std=std)
    test_ds  = MelNPYDataset(test_items,  mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = SmallCNN(n_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_path = CHECKPOINT_DIR / "best_cnn.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "mean": mean,
                "std": std
            }, best_path)
            print(f"[CKPT] Saved best model -> {best_path}")

    # Final test
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print("Loaded best checkpoint for test eval.")
    te_loss, te_acc = evaluate(model, test_loader, device)
    print(f"[TEST] loss {te_loss:.4f} acc {te_acc:.3f}")

if __name__ == "__main__":
    main()
