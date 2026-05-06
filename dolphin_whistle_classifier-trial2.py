"""
Bottlenose Dolphin Whistle Detector
====================================
Supports multiple WAV files — each label row is automatically routed
to the correct WAV based on date/dolphin/task rules defined in WAV_ROUTING.

Pipeline:
  1. Load cleaned CSV labels -> route each row to its WAV file
  2. Build training dataset from all WAV files combined
  3. Train a binary CNN (whistle vs. noise) on mel-spectrogram patches
  4. Run inference on each WAV file
  5. Save detected whistles as individual WAV clips

Usage:
    python dolphin_whistle_classifier.py \
        --wav_dir  "/path/to/wav/folder" \
        --labels   "whistles_cleaned.csv" \
        --output_dir "whistles_out/" \
        [--label_duration 1.5] [--epochs 20] [--threshold 0.5] \
        [--min_gap 0.1] [--min_duration 0.2] [--model_out model.pt]
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# CONFIG DEFAULTS
# ─────────────────────────────────────────────
SR           = 22050
N_MELS       = 64
HOP_LENGTH   = 256
WIN_DURATION = 1.0
HOP_DURATION = 0.1
MIN_GAP      = 0.1
MIN_DURATION = 0.2


# ─────────────────────────────────────────────
# WAV ROUTING RULES
# Each rule is a dict with:
#   "filename"  : WAV filename (without path)
#   "dates"     : list of date strings "YYYY-MM-DD", or None = any date
#   "dolphins"  : list of dolphin codes to match, or None = any
#   "tasks"     : list of task codes to match, or None = any
#
# Rules are checked in order — first match wins.
# ─────────────────────────────────────────────
WAV_ROUTING = [
    {
        "filename": "PRS_25-02-24_BAI-NAI_GL-EM_yse-bs_CtoC.wav",
        "dates":    ["2025-02-24"],
        "dolphins": ["BAI", "NAI"],
        "tasks":    ["PRS", "PRS-I"],
    },
    {
        "filename": "PRS_25-02-24_HRT_GL_yne-bse_CtoC.wav",
        "dates":    ["2025-02-24"],
        "dolphins": ["HRT"],
        "tasks":    ["PRS", "PRS-I"],
    },
    # ── Add more rules here as you get more WAV files ──
    # {
    #     "filename": "PRS_25-03-07_NAI-HRT_xxx.wav",
    #     "dates":    ["2025-03-07"],
    #     "dolphins": ["NAI", "HRT"],
    #     "tasks":    ["PRS"],
    # },
]


def route_row(row):
    """Return the WAV filename for a label row, or None if no rule matches."""
    for rule in WAV_ROUTING:
        date_match     = (rule["dates"]    is None) or (row["Date"]    in rule["dates"])
        dolphin_match  = (rule["dolphins"] is None) or (row["Dolphin"] in rule["dolphins"])
        task_match     = (rule["tasks"]    is None) or (row["Task"]    in rule["tasks"])
        if date_match and dolphin_match and task_match:
            return rule["filename"]
    return None


# ─────────────────────────────────────────────
# 1. FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_patch(y, sr, start_sec, duration=WIN_DURATION):
    start_sample = int(start_sec * sr)
    chunk = y[start_sample: start_sample + int(duration * sr)]
    target_len = int(duration * sr)
    if len(chunk) < target_len:
        chunk = np.pad(chunk, (0, target_len - len(chunk)))
    S    = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=N_MELS,
                                           hop_length=HOP_LENGTH, fmax=sr // 2)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_db.astype(np.float32)


# ─────────────────────────────────────────────
# 2. LABEL LOADING & ROUTING
# ─────────────────────────────────────────────
def load_labels(csv_path, wav_dir, label_duration=1.5):
    """
    Reads the cleaned CSV (Date, Dolphin, Task, Whistle_start_sec).
    Routes each row to a WAV file using WAV_ROUTING rules.
    Returns a dict: { wav_filepath -> DataFrame(start, end) }
    """
    df = pd.read_csv(csv_path)

    # Ensure expected columns exist
    required = {"Date", "Dolphin", "Task", "Whistle_start_sec"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}. Found: {list(df.columns)}")

    df["Dolphin"] = df["Dolphin"].fillna("").str.strip()
    df["Task"]    = df["Task"].fillna("").str.strip()
    df["Date"]    = df["Date"].astype(str).str.strip()

    df["_wav"] = df.apply(route_row, axis=1)

    unmatched = df[df["_wav"].isna()]
    if len(unmatched):
        print(f"\n  WARNING: {len(unmatched)} label row(s) did not match any WAV routing rule and will be skipped:")
        print(unmatched[["Date","Dolphin","Task","Whistle_start_sec"]].to_string(index=False))

    matched = df.dropna(subset=["_wav"]).copy()
    matched["end"] = matched["Whistle_start_sec"] + label_duration
    matched.rename(columns={"Whistle_start_sec": "start"}, inplace=True)

    # Group by WAV file
    wav_labels = {}
    for wav_name, group in matched.groupby("_wav"):
        wav_path = os.path.join(wav_dir, wav_name)
        if not os.path.exists(wav_path):
            print(f"\n  WARNING: WAV file not found, skipping: {wav_path}")
            continue
        wav_labels[wav_path] = group[["start", "end"]].sort_values("start").reset_index(drop=True)

    return wav_labels


# ─────────────────────────────────────────────
# 3. DATASET BUILDER
# ─────────────────────────────────────────────
def is_whistle(t_start, t_end, labels_df, iou_thresh=0.3):
    for _, row in labels_df.iterrows():
        overlap = max(0, min(t_end, row.end) - max(t_start, row.start))
        union   = max(t_end, row.end) - min(t_start, row.start)
        if union > 0 and overlap / union >= iou_thresh:
            return True
    return False


def build_dataset_from_wav(y, sr, labels_df):
    total_duration = len(y) / sr
    patches, targets = [], []
    t = 0.0
    while t + WIN_DURATION <= total_duration:
        patches.append(extract_patch(y, sr, t))
        targets.append(1 if is_whistle(t, t + WIN_DURATION, labels_df) else 0)
        t += HOP_DURATION
    return patches, targets


# ─────────────────────────────────────────────
# 4. CNN MODEL
# ─────────────────────────────────────────────
class WhistleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1))).squeeze(1)


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
class SpectrogramDataset(Dataset):
    def __init__(self, patches, targets):
        self.X = torch.tensor(patches, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_model(patches, targets, epochs=20, device="cpu"):
    X_tr, X_val, y_tr, y_val = train_test_split(
        patches, targets, test_size=0.15, stratify=targets, random_state=42)

    train_dl = DataLoader(SpectrogramDataset(X_tr,  y_tr),  batch_size=32, shuffle=True)
    val_dl   = DataLoader(SpectrogramDataset(X_val, y_val), batch_size=32)

    pos_weight = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model     = WhistleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_dl.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                val_loss += criterion(model(xb.to(device)), yb.to(device)).item() * len(yb)
        val_loss /= len(val_dl.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

    model.load_state_dict(best_state)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            logits = model(xb.to(device))
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int))
            trues.extend(yb.numpy().astype(int))
    print("\nValidation Report:")
    print(classification_report(trues, preds, target_names=["noise", "whistle"]))

    return model


# ─────────────────────────────────────────────
# 6. INFERENCE
# ─────────────────────────────────────────────
def predict_probs(model, y, sr, device="cpu"):
    total_duration = len(y) / sr
    times, patches = [], []
    t = 0.0
    while t + WIN_DURATION <= total_duration:
        patches.append(extract_patch(y, sr, t))
        times.append(t)
        t += HOP_DURATION

    patches = np.array(patches)
    probs   = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), 64):
            batch  = torch.tensor(patches[i:i+64], dtype=torch.float32).to(device)
            probs.extend(torch.sigmoid(model(batch)).cpu().numpy())

    return np.array(times), np.array(probs)


# ─────────────────────────────────────────────
# 7. POST-PROCESSING
# ─────────────────────────────────────────────
def probs_to_segments(times, probs, threshold=0.5, min_gap=MIN_GAP, min_duration=MIN_DURATION):
    detections = [(t, t + WIN_DURATION) for t, p in zip(times, probs) if p >= threshold]
    if not detections:
        return []
    merged = [list(detections[0])]
    for start, end in detections[1:]:
        if start - merged[-1][1] <= min_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if (e - s) >= min_duration]


# ─────────────────────────────────────────────
# 8. SAVE CLIPS
# ─────────────────────────────────────────────
def save_segments(y, sr, segments, output_dir, wav_basename, padding=0.05):
    os.makedirs(output_dir, exist_ok=True)
    total_dur = len(y) / sr
    saved = []
    stem = os.path.splitext(wav_basename)[0]
    for i, (start, end) in enumerate(segments):
        s     = max(0.0, start - padding)
        e     = min(total_dur, end + padding)
        chunk = y[int(s * sr): int(e * sr)]
        fname = os.path.join(output_dir, f"{stem}__whistle_{i+1:04d}_{s:.3f}-{e:.3f}s.wav")
        sf.write(fname, chunk, sr)
        saved.append(fname)
        print(f"  Saved: {os.path.basename(fname)}  ({e-s:.2f}s)")
    return saved


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bottlenose dolphin whistle detector — multi-WAV")
    parser.add_argument("--wav_dir",        required=True,  help="Folder containing all WAV files")
    parser.add_argument("--labels",         required=True,  help="Path to cleaned CSV (Date, Dolphin, Task, Whistle_start_sec)")
    parser.add_argument("--output_dir",     default="whistles_out", help="Directory to save detected whistle clips")
    parser.add_argument("--label_duration", type=float, default=1.5,  help="Assumed whistle length in seconds (default: 1.5)")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--threshold",      type=float, default=0.5)
    parser.add_argument("--min_gap",        type=float, default=MIN_GAP)
    parser.add_argument("--min_duration",   type=float, default=MIN_DURATION)
    parser.add_argument("--model_out",      default=None,   help="Optional path to save trained model weights")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load & route labels ──
    print(f"\n[1/5] Loading labels: {args.labels}")
    wav_labels = load_labels(args.labels, args.wav_dir, label_duration=args.label_duration)

    if not wav_labels:
        print("ERROR: No matching WAV files found. Check --wav_dir and WAV_ROUTING rules.")
        return

    print(f"\n  Matched {len(wav_labels)} WAV file(s):")
    for wp, ldf in wav_labels.items():
        print(f"    {os.path.basename(wp)}  ->  {len(ldf)} whistle labels")

    # ── 2. Load audio & build combined dataset ──
    print(f"\n[2/5] Loading audio and building dataset...")
    all_patches, all_targets = [], []
    wav_audio = {}  # cache loaded audio for inference later

    for wav_path, labels_df in wav_labels.items():
        print(f"\n  Loading: {os.path.basename(wav_path)}")
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        print(f"    Duration: {len(y)/sr:.1f}s")
        wav_audio[wav_path] = (y, sr)

        patches, targets = build_dataset_from_wav(y, sr, labels_df)
        all_patches.extend(patches)
        all_targets.extend(targets)
        n_pos = sum(targets)
        print(f"    Windows: {len(targets)}  (whistle={n_pos}, noise={len(targets)-n_pos})")

    all_patches = np.array(all_patches)
    all_targets = np.array(all_targets)
    n_pos = all_targets.sum()
    n_neg = len(all_targets) - n_pos
    print(f"\n  Combined: {len(all_targets)} windows  (whistle={n_pos}, noise={n_neg})")

    if n_pos == 0:
        print("\nERROR: No whistle windows found. Check that timestamps match your WAV file durations.")
        return

    # ── 3. Train ──
    print(f"\n[3/5] Training CNN ({args.epochs} epochs)...")
    model = train_model(all_patches, all_targets, epochs=args.epochs, device=device)
    if args.model_out:
        torch.save(model.state_dict(), args.model_out)
        print(f"  Model saved -> {args.model_out}")

    # ── 4 & 5. Inference + save per WAV ──
    all_saved = []
    all_segments_rows = []

    for wav_path, (y, sr) in wav_audio.items():
        wav_name = os.path.basename(wav_path)
        print(f"\n[4-5/5] Inference on: {wav_name}")
        times, probs = predict_probs(model, y, sr, device=device)
        segments     = probs_to_segments(times, probs,
                                          threshold=args.threshold,
                                          min_gap=args.min_gap,
                                          min_duration=args.min_duration)
        print(f"  Detected {len(segments)} whistle segments")

        saved = save_segments(y, sr, segments, args.output_dir, wav_name)
        all_saved.extend(saved)

        for s, e in segments:
            all_segments_rows.append({"wav": wav_name, "start": s, "end": e})

    # Summary CSV
    summary_path = os.path.join(args.output_dir, "detections.csv")
    pd.DataFrame(all_segments_rows).to_csv(summary_path, index=False)
    print(f"\nDetection summary -> {summary_path}")
    print(f"Done! {len(all_saved)} total whistle clips saved.")


if __name__ == "__main__":
    main()
