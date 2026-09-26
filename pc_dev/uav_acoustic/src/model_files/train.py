import sys
# path setup for local execution
sys.path.append(r"C:\final_project")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json
from datetime import datetime
from typing import List, Tuple, Dict

# Metrics imports
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import itertools

# import the model architecture
from uav_acoustic.src.model_files.model import SmallCNN

# ==========================================
#       Configuration and Hyperparameters
# ==========================================
PROCESSED_DIR = Path(r"C:\final_project\uav_acoustic\data\processed")
CHECKPOINT_BASE_DIR = Path(r"C:\final_project\uav_acoustic\models")

hyperparams = {
    "RANDOM_SEED": 42,
    "BATCH_SIZE": 64,
    "EPOCHS": 15,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "NUM_WORKERS": 0,    # 0 for Windows compatibility
    "MAX_FILES_FOR_NORM": 2000,
    "FIXED_LENGTH": 63,   # size adjustment for mel spectrograms
    "EVAL_SNR_LEVEL": "SNR_10" # Representative SNR level used for unbiased validation
}

# ==========================================
#    training and evaluation functions
# ==========================================

def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        bsz = x.size(0)
        total_loss += float(loss.item()) * bsz
        total_acc += accuracy_from_logits(logits, y) * bsz
        n += bsz
    return total_loss / max(n, 1), total_acc / max(n, 1)

def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            bsz = x.size(0)
            total_loss += float(loss.item()) * bsz
            total_acc += accuracy_from_logits(logits, y) * bsz
            n += bsz
    return total_loss / max(n, 1), total_acc / max(n, 1)

def get_predictions_and_targets(model, loader, device):
    """
    Extracts all raw predictions and ground-truth targets from a dataloader.
    Used for comprehensive matrix evaluations.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
    return np.array(all_preds), np.array(all_targets)

# ==========================================
#    dataset class and data collection
# ==========================================

def collect_train_items(processed_root: Path, class_to_idx: Dict[str, int]) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    nd_dir = processed_root / "common" / "train" / "not_drone"
    if nd_dir.exists():
        items += [(f, class_to_idx["not_drone"]) for f in nd_dir.glob("*.npy")]
        
    snr_dirs = [d for d in processed_root.iterdir() if d.is_dir() and d.name.startswith("SNR_")]
    for sd in snr_dirs:
        d_dir = sd / "train" / "drone"
        if d_dir.exists():
            items += [(f, class_to_idx["drone"]) for f in d_dir.glob("*.npy")]
    return items

def collect_eval_items(processed_root: Path, split: str, target_snr: str, class_to_idx: Dict[str, int]) -> List[Tuple[Path, int]]:
    items: List[Tuple[Path, int]] = []
    nd_dir = processed_root / "common" / split / "not_drone"
    if nd_dir.exists():
        items += [(f, class_to_idx["not_drone"]) for f in nd_dir.glob("*.npy")]
        
    d_dir = processed_root / target_snr / split / "drone"
    if d_dir.exists():
        items += [(f, class_to_idx["drone"]) for f in d_dir.glob("*.npy")]
    return items

class MelNPYDataset(Dataset):
    def __init__(self, items, mean, std, fixed_length=63):
        self.items = items
        self.mean, self.std = mean, std
        self.fixed_length = fixed_length

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        mel = np.load(path)
        x = torch.from_numpy(mel).float()
        if x.shape[1] < self.fixed_length:
            x = F.pad(x, (0, self.fixed_length - x.shape[1]))
        else:
            x = x[:, :self.fixed_length]
        x = (x - self.mean) / self.std
        return x.unsqueeze(0), torch.tensor(label, dtype=torch.long)

# ==========================================
#    graph and visualization functions
# ==========================================

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train'); plt.plot(epochs, val_losses, label='Val')
    plt.title('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train'); plt.plot(epochs, val_accs, label='Val')
    plt.title('Accuracy'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path); plt.close()

def plot_advanced_snr_performance(snr_results_df, run_name, save_path):
    """
    Plots Accuracy, Precision, and Recall across all evaluated SNR levels.
    """
    plt.figure(figsize=(12, 6))
    x_labels = snr_results_df['SNR_Level'].tolist()
    
    plt.plot(x_labels, snr_results_df['Accuracy'] * 100, marker='o', linewidth=2, label='Accuracy', color='teal')
    plt.plot(x_labels, snr_results_df['Precision'] * 100, marker='s', linestyle='--', linewidth=2, label='Precision (Low False Alarms)', color='darkorange')
    plt.plot(x_labels, snr_results_df['Recall'] * 100, marker='^', linestyle=':', linewidth=2, label='Recall (Sensitivity)', color='crimson')
    
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% Target')
    plt.title(f'Multi-Metric Performance vs. SNR ({run_name})')
    plt.ylabel('Score (%)')
    plt.xlabel('Noise Level')
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path); plt.close()

def plot_confusion_matrix(cm, classes, title, save_path, cmap=plt.cm.Blues):
    """
    Generates and saves a clean, formatted confusion matrix visualization plot.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path); plt.close()

# ==========================================
#                  Main Function
# ==========================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    cur_run_dir = CHECKPOINT_BASE_DIR / run_name
    cur_run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Experiment: {run_name} ---")
    
    with open(cur_run_dir / "config.json", 'w') as f:
        json.dump(hyperparams, f, indent=4)

    random.seed(hyperparams["RANDOM_SEED"])
    np.random.seed(hyperparams["RANDOM_SEED"])
    torch.manual_seed(hyperparams["RANDOM_SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_to_idx = {"not_drone": 0, "drone": 1}
    classes_list = ["not_drone", "drone"]
    
    train_items = collect_train_items(PROCESSED_DIR, class_to_idx)
    val_items   = collect_eval_items(PROCESSED_DIR, "val", hyperparams["EVAL_SNR_LEVEL"], class_to_idx)
    
    print(f"[*] Total training chunks loaded (All SNRs): {len(train_items)}")
    print(f"[*] Total validation chunks loaded ({hyperparams['EVAL_SNR_LEVEL']} only): {len(val_items)}")
    
    train_paths = [p for p, _ in train_items]
    sel = random.sample(train_paths, min(len(train_paths), hyperparams["MAX_FILES_FOR_NORM"]))
    vals = [np.load(p).astype(np.float32) for p in sel]
    flat = np.concatenate([v.reshape(-1) for v in vals])
    mean, std = float(flat.mean()), float(flat.std() + 1e-8)
    with open(cur_run_dir / "norm_stats.json", 'w') as f:
        json.dump({"mean": mean, "std": std}, f)

    train_loader = DataLoader(MelNPYDataset(train_items, mean, std), batch_size=hyperparams["BATCH_SIZE"], shuffle=True, num_workers=hyperparams["NUM_WORKERS"])
    val_loader   = DataLoader(MelNPYDataset(val_items, mean, std), batch_size=hyperparams["BATCH_SIZE"], num_workers=hyperparams["NUM_WORKERS"])

    model = SmallCNN(n_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hyperparams["LEARNING_RATE"], weight_decay=hyperparams["WEIGHT_DECAY"])
    
    best_val_loss = float("inf")
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, hyperparams["EPOCHS"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc); val_accs.append(va_acc)
        
        print(f"Epoch {epoch:02d} | Train: {tr_acc:.3f} | Val: {va_acc:.3f}")
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std}, cur_run_dir / "best_model.pt")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, cur_run_dir / "curves.png")

    # --- Comprehensive Final Evaluation Section ---
    print("\n--- Final Analysis Per SNR Level ---")
    snr_records = []
    snr_dirs = sorted([d for d in PROCESSED_DIR.iterdir() if d.is_dir() and d.name.startswith("SNR_")], key=lambda x: int(x.name.split('_')[1]), reverse=True)
    model.load_state_dict(torch.load(cur_run_dir / "best_model.pt", map_location=device, weights_only=True)["state_dict"])

    for sd in snr_dirs:
        t_items = []
        nd_test = PROCESSED_DIR / "common" / "test" / "not_drone"
        t_items += [(f, class_to_idx["not_drone"]) for f in nd_test.glob("*.npy")]
        d_test = sd / "test" / "drone"
        t_items += [(f, class_to_idx["drone"]) for f in d_test.glob("*.npy")]
        
        if not t_items: continue
        
        test_loader = DataLoader(MelNPYDataset(t_items, mean, std), batch_size=hyperparams["BATCH_SIZE"], num_workers=hyperparams["NUM_WORKERS"])
        preds, targets = get_predictions_and_targets(model, test_loader, device)
        
        # Calculate multiple distinct validation metrics
        acc = float((preds == targets).mean())
        prec = precision_score(targets, preds, zero_division=0)
        rec = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        
        print(f"\n[{sd.name}] Acc: {acc:.2%} | Precision: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.2%}")
        
        # Compute Confusion Matrix
        cm = confusion_matrix(targets, preds)
        plot_confusion_matrix(cm, classes_list, f"Confusion Matrix: {sd.name}", cur_run_dir / f"confusion_matrix_{sd.name}.png")
        
        snr_records.append({
            "SNR_Level": sd.name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        })

    # Save detailed telemetry to CSV
    df_results = pd.DataFrame(snr_records)
    df_results.to_csv(cur_run_dir / "detailed_snr_results.csv", index=False)
    
    # Render advanced trends plots
    plot_advanced_snr_performance(df_results, run_name, cur_run_dir / "advanced_snr_analysis.png")
    print(f"\n[DONE] Advanced metrics, curves, and confusion matrices saved in: {cur_run_dir}")

if __name__ == "__main__":
    main()