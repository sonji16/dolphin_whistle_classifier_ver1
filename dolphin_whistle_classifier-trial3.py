"""
Bottlenose Dolphin Whistle Detector
====================================
Pipeline:
  1. Load cleaned CSV labels -> route each row to its WAV file
  2. Build training dataset (0.5s frames, whistle vs noise)
  3. Train binary CNN
  4. Evaluate every 0.5s frame across each WAV -> save predictions CSV
  5. Detect whistle start/end from state flips
  6. Chop WAV into individual whistle clips

Output per WAV file:
  - <wav_stem>_predictions.csv   : timestamp + probability + label for every 0.5s frame
  - whistle_NNNN_Xs-Ys.wav       : one clip per detected whistle

Usage:
    python dolphin_whistle_classifier.py \
        --wav_dir    "/path/to/wavs/" \
        --labels     "whistles_cleaned_all.csv" \
        --output_dir "whistles_out/" \
        [--label_duration 1.5] [--epochs 20] [--threshold 0.5] \
        [--min_gap 0.2] [--min_duration 0.2] [--model_out model.pt]
"""

import os
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
# CONFIG
# ─────────────────────────────────────────────
SR           = 22050
N_MELS       = 64
HOP_LENGTH   = 256
WIN_DURATION = 1.0    # spectrogram context window (seconds)
FRAME_HOP    = 0.5    # evaluate every 0.5 seconds
MIN_GAP      = 0.2    # merge whistle detections closer than this (seconds)
MIN_DURATION = 0.2    # discard detections shorter than this (seconds)


# ─────────────────────────────────────────────
# WAV ROUTING RULES
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
]


def route_row(row):
    for rule in WAV_ROUTING:
        date_match    = (rule["dates"]    is None) or (row["Date"]    in rule["dates"])
        dolphin_match = (rule["dolphins"] is None) or (row["Dolphin"] in rule["dolphins"])
        task_match    = (rule["tasks"]    is None) or (row["Task"]    in rule["tasks"])
        if date_match and dolphin_match and task_match:
            return rule["filename"]
    return None


# ─────────────────────────────────────────────
# 1. FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_patch(y, sr, center_sec, duration=WIN_DURATION):
    """
    Extract a mel-spectrogram patch centred on center_sec.
    Using a 1s context window centred on each 0.5s frame gives
    the CNN left+right context around the frame being classified.
    """
    start_sec = center_sec - duration / 2
    start_sample = max(0, int(start_sec * sr))
    end_sample   = start_sample + int(duration * sr)
    chunk = y[start_sample:end_sample]

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
    df = pd.read_csv(csv_path)
    required = {"Date", "Dolphin", "Task", "Whistle_start_sec"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}. Found: {list(df.columns)}")

    df["Dolphin"] = df["Dolphin"].fillna("").str.strip()
    df["Task"]    = df["Task"].fillna("").str.strip()
    df["Date"]    = df["Date"].astype(str).str.strip()
    df["_wav"]    = df.apply(route_row, axis=1)

    unmatched = df[df["_wav"].isna()]
    if len(unmatched):
        print(f"  WARNING: {len(unmatched)} row(s) matched no routing rule and are skipped.")

    matched = df.dropna(subset=["_wav"]).copy()
    matched["end"] = matched["Whistle_start_sec"] + label_duration
    matched.rename(columns={"Whistle_start_sec": "start"}, inplace=True)

    wav_labels = {}
    for wav_name, group in matched.groupby("_wav"):
        wav_path = os.path.join(wav_dir, wav_name)
        if not os.path.exists(wav_path):
            print(f"  WARNING: WAV not found, skipping: {wav_path}")
            continue
        wav_labels[wav_path] = group[["start","end"]].sort_values("start").reset_index(drop=True)

    return wav_labels


# ─────────────────────────────────────────────
# 3. DATASET BUILDER  (0.5s frames)
# ─────────────────────────────────────────────
def frame_is_whistle(t_center, labels_df, iou_thresh=0.3):
    """True if the 0.5s frame centred on t_center overlaps a whistle label."""
    t_start = t_center - FRAME_HOP / 2
    t_end   = t_center + FRAME_HOP / 2
    for _, row in labels_df.iterrows():
        overlap = max(0, min(t_end, row.end) - max(t_start, row.start))
        union   = max(t_end, row.end) - min(t_start, row.start)
        if union > 0 and overlap / union >= iou_thresh:
            return True
    return False


def build_dataset_from_wav(y, sr, labels_df):
    total_duration = len(y) / sr
    patches, targets, frame_times = [], [], []
    t = FRAME_HOP / 2   # start at centre of first frame
    while t < total_duration:
        patches.append(extract_patch(y, sr, t))
        targets.append(1 if frame_is_whistle(t, labels_df) else 0)
        frame_times.append(t)
        t += FRAME_HOP
    return patches, targets, frame_times


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
        self.X = torch.tensor(np.array(patches), dtype=torch.float32)
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
# 6. INFERENCE — every 0.5s frame
# ─────────────────────────────────────────────
def predict_frames(model, y, sr, device="cpu"):
    """Returns frame_times, probs, labels arrays — one entry per 0.5s frame."""
    total_duration = len(y) / sr
    frame_times, patches = [], []

    t = FRAME_HOP / 2
    while t < total_duration:
        patches.append(extract_patch(y, sr, t))
        frame_times.append(round(t, 3))
        t += FRAME_HOP

    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), 64):
            batch  = torch.tensor(np.array(patches[i:i+64]), dtype=torch.float32).to(device)
            probs.extend(torch.sigmoid(model(batch)).cpu().numpy())

    frame_times = np.array(frame_times)
    probs       = np.array(probs)
    return frame_times, probs


# ─────────────────────────────────────────────
# 7. FRAME CSV + SEGMENT DETECTION
# ─────────────────────────────────────────────
def save_frame_csv(frame_times, probs, threshold, output_dir, wav_stem):
    """Save per-frame predictions to CSV."""
    labels = (probs >= threshold).astype(int)
    df = pd.DataFrame({
        "time_sec":    frame_times,
        "probability": np.round(probs, 4),
        "whistle":     labels,
    })
    # Add readable timestamp column MM:SS.ss
    def to_mmss(s):
        m = int(s) // 60
        sec = s - m * 60
        return f"{m}:{sec:05.2f}"
    df.insert(1, "timestamp", df["time_sec"].apply(to_mmss))

    csv_path = os.path.join(output_dir, f"{wav_stem}_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Frame predictions saved -> {os.path.basename(csv_path)}")
    return df


def frames_to_segments(frame_times, probs, threshold=0.5,
                        min_gap=MIN_GAP, min_duration=MIN_DURATION):
    """
    Convert per-frame probabilities to (start, end) segments.
    A whistle starts when a frame crosses threshold and ends when it drops below.
    Adjacent detections within min_gap are merged.
    """
    labels = probs >= threshold

    # Build raw on/off segments from frame state flips
    segments = []
    in_whistle = False
    seg_start  = None

    for t, is_on in zip(frame_times, labels):
        if is_on and not in_whistle:
            seg_start  = t - FRAME_HOP / 2   # start = left edge of this frame
            in_whistle = True
        elif not is_on and in_whistle:
            seg_end    = t - FRAME_HOP / 2   # end = left edge of first OFF frame
            segments.append([seg_start, seg_end])
            in_whistle = False

    # Close any open segment at end of file
    if in_whistle:
        segments.append([seg_start, frame_times[-1] + FRAME_HOP / 2])

    if not segments:
        return []

    # Merge segments closer than min_gap
    merged = [segments[0]]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= min_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # Filter too-short segments
    return [(s, e) for s, e in merged if (e - s) >= min_duration]


# ─────────────────────────────────────────────
# 8. SAVE WHISTLE CLIPS
# ─────────────────────────────────────────────
def save_whistle_clips(y, sr, segments, output_dir, wav_stem, padding=0.05):
    total_dur = len(y) / sr
    saved = []
    for i, (start, end) in enumerate(segments):
        s     = max(0.0, start - padding)
        e     = min(total_dur, end + padding)
        chunk = y[int(s * sr): int(e * sr)]
        fname = os.path.join(output_dir, f"{wav_stem}__whistle_{i+1:04d}_{s:.3f}-{e:.3f}s.wav")
        sf.write(fname, chunk, sr)
        saved.append(fname)
        print(f"  [{i+1:04d}] {s:.2f}s - {e:.2f}s  ({e-s:.2f}s)")
    return saved


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dolphin whistle detector — 0.5s frame classification")
    parser.add_argument("--wav_dir",        required=True)
    parser.add_argument("--labels",         required=True)
    parser.add_argument("--output_dir",     default="whistles_out")
    parser.add_argument("--label_duration", type=float, default=1.5)
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--threshold",      type=float, default=0.5)
    parser.add_argument("--min_gap",        type=float, default=MIN_GAP)
    parser.add_argument("--min_duration",   type=float, default=MIN_DURATION)
    parser.add_argument("--model_out",      default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
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

    # ── 2. Load audio + build combined dataset ──
    print(f"\n[2/5] Loading audio and building 0.5s frame dataset...")
    all_patches, all_targets = [], []
    wav_audio = {}

    for wav_path, labels_df in wav_labels.items():
        print(f"\n  {os.path.basename(wav_path)}")
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        print(f"    Duration: {len(y)/sr:.1f}s")
        wav_audio[wav_path] = (y, sr)

        patches, targets, _ = build_dataset_from_wav(y, sr, labels_df)
        all_patches.extend(patches)
        all_targets.extend(targets)
        n_pos = sum(targets)
        print(f"    Frames: {len(targets)}  (whistle={n_pos}, noise={len(targets)-n_pos})")

    all_targets = np.array(all_targets)
    n_pos = all_targets.sum()
    n_neg = len(all_targets) - n_pos
    print(f"\n  Combined: {len(all_targets)} frames  (whistle={n_pos}, noise={n_neg})")

    if n_pos == 0:
        print("\nERROR: No whistle frames found. Check timestamps match WAV durations.")
        return

    # ── 3. Train ──
    print(f"\n[3/5] Training CNN ({args.epochs} epochs)...")
    model = train_model(all_patches, all_targets, epochs=args.epochs, device=device)
    if args.model_out:
        torch.save(model.state_dict(), args.model_out)
        print(f"  Model saved -> {args.model_out}")

    # ── 4 & 5. Inference + save per WAV ──
    all_saved = []
    all_detection_rows = []

    for wav_path, (y, sr) in wav_audio.items():
        wav_stem = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\n[4-5/5] Inference: {wav_stem}")

        # Predict every 0.5s frame
        frame_times, probs = predict_frames(model, y, sr, device=device)

        # Save frame CSV
        save_frame_csv(frame_times, probs, args.threshold, args.output_dir, wav_stem)

        # Detect segments from state flips
        segments = frames_to_segments(frame_times, probs,
                                       threshold=args.threshold,
                                       min_gap=args.min_gap,
                                       min_duration=args.min_duration)
        print(f"  Detected {len(segments)} whistle(s)")

        # Save individual clips
        saved = save_whistle_clips(y, sr, segments, args.output_dir, wav_stem)
        all_saved.extend(saved)

        for s, e in segments:
            all_detection_rows.append({"wav": os.path.basename(wav_path),
                                        "start": round(s, 3), "end": round(e, 3),
                                        "duration": round(e - s, 3)})

    # Overall detections CSV
    det_path = os.path.join(args.output_dir, "detections.csv")
    pd.DataFrame(all_detection_rows).to_csv(det_path, index=False)
    print(f"\nAll detections -> {det_path}")
    print(f"Done! {len(all_saved)} whistle clips saved.")


if __name__ == "__main__":
    main()


'''
python "/Users/sonji16/Desktop/claude code/dolphin_whistle_classifier.py" \
    --wav_dir  "/Users/sonji16/Desktop/claude code/" \
    --labels   "/Users/sonji16/Desktop/claude code/whistles_cleaned_all.csv" \
    --output_dir "/Users/sonji16/Desktop/claude code/whistles_out/"
'''