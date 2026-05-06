"""
Bottlenose Dolphin Whistle Detector
====================================
Pipeline:
  1. Load WAV + CSV labels -> build training dataset
  2. Train a binary CNN (whistle vs. noise) on mel-spectrogram patches
  3. Slide window across full recording -> predict whistle probability
  4. Post-process predictions -> merge into start/end timestamps
  5. Save each detected whistle as its own WAV file

Supports your specific CSV format:
    NLP Whistle 2025 Data - Raw Data       <- ignored
    NLP Whistle 2025 Data - Raw Data,,     <- ignored
    Date,Dolphin,Whistle start time        <- header
    25-02-24,NAI,                          <- blank start = no whistle, skipped
    25-02-24,BAI,2:45.35                   <- M:SS.ss timestamp -> converted to seconds

Usage:
    python dolphin_whistle_classifier.py \
        --wav recording.wav \
        --labels labels.csv \
        --output_dir whistles/ \
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
WIN_DURATION = 1.0    # sliding window length (seconds)
HOP_DURATION = 0.1    # sliding window step   (seconds)
MIN_GAP      = 0.1    # merge detections closer than this (seconds)
MIN_DURATION = 0.2    # discard segments shorter than this (seconds)


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
    return S_db.astype(np.float32)   # (N_MELS, time_frames)


# ─────────────────────────────────────────────
# 2. LABEL LOADING
# ─────────────────────────────────────────────
def parse_timestamp(ts):
    """
    Convert timestamp string to seconds.
    Handles:
      M:SS.ss   e.g. '2:45.35'  ->  165.35
      SS.ss     e.g. '45.35'    ->  45.35
      plain int/float strings
    Returns None if unparseable.
    """
    if not isinstance(ts, str):
        try:
            return float(ts)
        except (TypeError, ValueError):
            return None
    ts = ts.strip()
    if not ts:
        return None
    # M:SS.ss or M:SS
    m = re.match(r'^(\d+):(\d+(?:\.\d+)?)$', ts)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    # plain number
    try:
        return float(ts)
    except ValueError:
        return None


def load_labels(csv_path, label_duration=1.5):
    """
    Reads the NLP Whistle CSV which has:
      - 2 junk rows at the top
      - Header row: Date, Dolphin, Whistle start time
      - Timestamps in M:SS.ss format
      - Many rows with blank start times (skipped)

    Returns a DataFrame with columns [start, end] in seconds.
    """
    # Skip the first 2 junk rows, use row 3 as header
    df = pd.read_csv(csv_path, skiprows=2, header=0)

    # Find the whistle time column (3rd column regardless of exact name)
    time_col = df.columns[2]
    print(f"      Reading timestamps from column: '{time_col}'")

    starts = df[time_col].apply(parse_timestamp).dropna()
    starts = starts[starts > 0]  # drop zeros/negatives if any

    result = pd.DataFrame({
        "start": starts.values,
        "end":   starts.values + label_duration
    }).sort_values("start").reset_index(drop=True)

    print(f"      Found {len(result)} whistle timestamps "
          f"(range: {result.start.min():.1f}s - {result.start.max():.1f}s)")
    print(f"      Inferring end = start + {label_duration}s")
    return result


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


def build_dataset(y, sr, labels_df):
    total_duration = len(y) / sr
    patches, targets = [], []
    t = 0.0
    while t + WIN_DURATION <= total_duration:
        patches.append(extract_patch(y, sr, t))
        targets.append(1 if is_whistle(t, t + WIN_DURATION, labels_df) else 0)
        t += HOP_DURATION
    return np.array(patches), np.array(targets)


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

    # Validation report
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
# 6. INFERENCE — SLIDING WINDOW
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
def save_segments(y, sr, segments, output_dir, padding=0.05):
    os.makedirs(output_dir, exist_ok=True)
    total_dur = len(y) / sr
    saved = []
    for i, (start, end) in enumerate(segments):
        s     = max(0.0, start - padding)
        e     = min(total_dur, end + padding)
        chunk = y[int(s * sr): int(e * sr)]
        fname = os.path.join(output_dir, f"whistle_{i+1:04d}_{s:.3f}-{e:.3f}s.wav")
        sf.write(fname, chunk, sr)
        saved.append(fname)
        print(f"  Saved: {fname}  ({e-s:.2f}s)")
    return saved


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bottlenose dolphin whistle detector")
    parser.add_argument("--wav",            required=True,  help="Path to input WAV file")
    parser.add_argument("--labels",         required=True,  help="Path to your NLP Whistle CSV")
    parser.add_argument("--output_dir",     default="whistles", help="Directory to save clipped whistles")
    parser.add_argument("--label_duration", type=float, default=1.5,
                        help="Assumed whistle length in seconds (default: 1.5)")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--threshold",      type=float, default=0.5)
    parser.add_argument("--min_gap",        type=float, default=MIN_GAP)
    parser.add_argument("--min_duration",   type=float, default=MIN_DURATION)
    parser.add_argument("--model_out",      default=None, help="Optional path to save trained model weights")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load audio
    print(f"\n[1/5] Loading audio: {args.wav}")
    y, sr = librosa.load(args.wav, sr=SR, mono=True)
    print(f"      Duration: {len(y)/sr:.1f}s  SR: {sr}")

    # 2. Load labels
    print(f"\n[2/5] Loading labels: {args.labels}")
    labels_df = load_labels(args.labels, label_duration=args.label_duration)
    print(labels_df.head(10).to_string(index=False))

    # Warn if any labels fall outside the audio duration
    audio_dur = len(y) / sr
    out_of_range = labels_df[labels_df["start"] > audio_dur]
    if len(out_of_range):
        print(f"\n  WARNING: {len(out_of_range)} label(s) start after audio end "
              f"({audio_dur:.1f}s) and will be ignored:")
        print(out_of_range.to_string(index=False))
        labels_df = labels_df[labels_df["start"] <= audio_dur].reset_index(drop=True)

    # 3. Build dataset
    print("\n[3/5] Building spectrogram dataset...")
    patches, targets = build_dataset(y, sr, labels_df)
    n_pos = targets.sum()
    n_neg = len(targets) - n_pos
    print(f"      Windows: {len(targets)}  (whistle={n_pos}, noise={n_neg})")

    if n_pos == 0:
        print("\nERROR: No whistle windows found. Check that your WAV file covers "
              "the same time range as the timestamps in your CSV.")
        return

    # 4. Train
    print(f"\n[4/5] Training CNN ({args.epochs} epochs)...")
    model = train_model(patches, targets, epochs=args.epochs, device=device)
    if args.model_out:
        torch.save(model.state_dict(), args.model_out)
        print(f"      Model saved -> {args.model_out}")

    # 5. Inference
    print("\n[5/5] Running inference on full recording...")
    times, probs = predict_probs(model, y, sr, device=device)
    segments     = probs_to_segments(times, probs,
                                      threshold=args.threshold,
                                      min_gap=args.min_gap,
                                      min_duration=args.min_duration)
    print(f"      Detected {len(segments)} whistle segments")

    # Save clips
    print(f"\nSaving WAV clips to: {args.output_dir}/")
    saved = save_segments(y, sr, segments, args.output_dir)

    # Summary CSV
    summary_path = os.path.join(args.output_dir, "detections.csv")
    pd.DataFrame(segments, columns=["start", "end"]).to_csv(summary_path, index=False)
    print(f"\nDetection summary -> {summary_path}")
    print(f"Done! {len(saved)} whistle clips saved.")


if __name__ == "__main__":
    main()
