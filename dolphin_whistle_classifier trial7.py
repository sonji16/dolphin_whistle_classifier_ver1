"""
Bottlenose Dolphin Whistle Detector
=====================================
Training data comes from TWO sources:
  A) SEANOE dataset — Whistle_Signals/ (pre-cut whistles) +
                      Vocalization_Labels/ TXT files (NOISE segments)
  B) Your own recordings — WAV files matched via WAV_ROUTING rules
                           using whistles_cleaned_all.csv

Inference runs on your own recordings only.

Folder structure expected:
  seanoe dataset/
    Whistle_Signals/
      20211120_102109_192/
        20211120_102109_192-W-52.07.wav
        ...
    Vocalization_Labels/
      20211120_102109_192.txt     (start  end  label)
    122802/
      Raw_recordings_Day1_pt1/
        20211120_102109_192.wav
      ...
    122803/ ...

Usage:
    python dolphin_whistle_classifier.py \
        --wav_dir    "/path/to/your/wavs/" \
        --labels     "whistles_cleaned_all.csv" \
        --seanoe_dir "/path/to/seanoe dataset/" \
        --output_dir "whistles_out/" \
        [--label_duration 1.5] [--epochs 30] [--threshold 0.5] \
        [--min_gap 0.2] [--min_duration 0.2] [--model_out model.pt]
"""

import os
import glob
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
WIN_DURATION = 0.5    # context window fed to CNN (seconds)
FRAME_HOP    = 0.1    # evaluate every 0.1s (50% overlap with 0.2s, 20% with 0.5s)
MIN_GAP      = 0.2    # merge detections closer than this (seconds)
MIN_DURATION = 0.2    # discard detections shorter than this (seconds)

# SEANOE label types
WHISTLE_LABELS = {"W", "W+NOISE"}
NOISE_LABELS   = {"NOISE"}
# ECT, BPS, FB etc. are ignored


# ─────────────────────────────────────────────
# WAV ROUTING — your own recordings
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
    # ── Add more entries as you get more recordings ──
]


def route_row(row):
    for rule in WAV_ROUTING:
        if (rule["dates"]    is None or row["Date"]    in rule["dates"]) and \
           (rule["dolphins"] is None or row["Dolphin"] in rule["dolphins"]) and \
           (rule["tasks"]    is None or row["Task"]    in rule["tasks"]):
            return rule["filename"]
    return None


# ─────────────────────────────────────────────
# 1. FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_patch(y, sr, center_sec, duration=WIN_DURATION):
    """Extract mel-spectrogram patch centred on center_sec."""
    start_sec    = center_sec - duration / 2
    start_sample = max(0, int(start_sec * sr))
    chunk        = y[start_sample: start_sample + int(duration * sr)]
    target_len   = int(duration * sr)
    if len(chunk) < target_len:
        chunk = np.pad(chunk, (0, target_len - len(chunk)))
    S    = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=N_MELS,
                                           hop_length=HOP_LENGTH, fmax=sr // 2)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_db.astype(np.float32)


def extract_patch_from_clip(y, sr, duration=WIN_DURATION):
    """Extract a single centred patch from a short pre-cut clip."""
    center = len(y) / sr / 2
    return extract_patch(y, sr, center, duration)


# ─────────────────────────────────────────────
# 2A. SEANOE TRAINING DATA
# ─────────────────────────────────────────────
def load_seanoe_label_file(txt_path):
    """Parse a Vocalization_Labels TXT file into a DataFrame."""
    rows = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end   = float(parts[1])
                label = parts[2].upper()
                rows.append({"start": start, "end": end, "label": label})
            except ValueError:
                continue
    return pd.DataFrame(rows)


def build_seanoe_dataset(seanoe_dir, max_noise_per_file=50):
    """
    Build patches from SEANOE data:
      Positives — pre-cut WAVs in Whistle_Signals/
      Negatives — NOISE segments from Vocalization_Labels TXTs
                  (extracted from raw WAVs in numbered subfolders)
    """
    patches, targets = [], []

    whistle_signals_dir   = os.path.join(seanoe_dir, "Whistle_Signals")
    vocalization_labels_dir = os.path.join(seanoe_dir, "Vocalization_Labels")

    # ── Positives: pre-cut whistle clips ──
    whistle_wavs = glob.glob(os.path.join(whistle_signals_dir, "**", "*.wav"),
                              recursive=True)
    print(f"  SEANOE whistle clips found: {len(whistle_wavs)}")

    for wav_path in whistle_wavs:
        try:
            y, sr = librosa.load(wav_path, sr=SR, mono=True)
            if len(y) < int(WIN_DURATION * sr / 2):
                continue   # clip too short
            patch = extract_patch_from_clip(y, sr)
            patches.append(patch)
            targets.append(1)
        except Exception:
            continue

    # ── Negatives: NOISE segments from label files ──
    # Find all raw WAV files across numbered subdirectories
    raw_wav_index = {}   # stem -> full path
    for entry in os.scandir(seanoe_dir):
        if not entry.is_dir():
            continue
        for wav_path in glob.glob(os.path.join(entry.path, "**", "*.wav"),
                                   recursive=True):
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            raw_wav_index[stem] = wav_path

    txt_files = glob.glob(os.path.join(vocalization_labels_dir, "*.txt"))
    print(f"  SEANOE label files found: {len(txt_files)}")

    noise_count = 0
    for txt_path in txt_files:
        stem     = os.path.splitext(os.path.basename(txt_path))[0]
        wav_path = raw_wav_index.get(stem)
        if wav_path is None:
            continue

        label_df = load_seanoe_label_file(txt_path)
        if label_df.empty or "label" not in label_df.columns:
            continue
        noise_rows = label_df[label_df["label"].isin(NOISE_LABELS)]
        if noise_rows.empty:
            continue

        try:
            y, sr = librosa.load(wav_path, sr=SR, mono=True)
        except Exception:
            continue

        total_dur = len(y) / sr
        count = 0
        for _, row in noise_rows.iterrows():
            if count >= max_noise_per_file:
                break
            # Sample multiple 0.5s windows from each NOISE segment
            t = row.start + WIN_DURATION / 2
            while t < min(row.end, total_dur) - WIN_DURATION / 2:
                patch = extract_patch(y, sr, t)
                patches.append(patch)
                targets.append(0)
                t += WIN_DURATION   # non-overlapping noise windows
                count += 1
                noise_count += 1

    print(f"  SEANOE noise patches extracted: {noise_count}")
    return patches, targets


# ─────────────────────────────────────────────
# 2B. YOUR OWN RECORDINGS TRAINING DATA
# ─────────────────────────────────────────────
def load_own_labels(csv_path, wav_dir, label_duration=1.5):
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
        print(f"  NOTE: {len(unmatched)} row(s) skipped (no matching WAV rule).")

    matched = df.dropna(subset=["_wav"]).copy()
    matched["end"] = matched["Whistle_start_sec"] + label_duration
    matched.rename(columns={"Whistle_start_sec": "start"}, inplace=True)

    wav_labels = {}
    for wav_name, group in matched.groupby("_wav"):
        wav_path = os.path.join(wav_dir, wav_name)
        if not os.path.exists(wav_path):
            print(f"  WARNING: WAV not found: {wav_path}")
            continue
        wav_labels[wav_path] = group[["start","end"]].sort_values("start").reset_index(drop=True)

    return wav_labels


def frame_is_whistle(t_center, labels_df):
    for _, row in labels_df.iterrows():
        if row.start <= t_center <= row.end:
            return True
    return False


def build_own_dataset(wav_labels):
    patches, targets = [], []
    for wav_path, labels_df in wav_labels.items():
        print(f"  {os.path.basename(wav_path)}")
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        total_dur = len(y) / sr
        t = FRAME_HOP / 2
        while t < total_dur:
            patches.append(extract_patch(y, sr, t))
            targets.append(1 if frame_is_whistle(t, labels_df) else 0)
            t += FRAME_HOP
        n_pos = sum(targets[-int(total_dur / FRAME_HOP):])
        print(f"    Frames: whistle={n_pos}")
    return patches, targets


# ─────────────────────────────────────────────
# 3. OVERSAMPLE MINORITY CLASS
# ─────────────────────────────────────────────
def oversample_minority(patches, targets, target_ratio=5.0):
    patches = np.array(patches)
    targets = np.array(targets)
    pos_idx = np.where(targets == 1)[0]
    neg_idx = np.where(targets == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0:
        return patches, targets

    desired_pos = max(n_pos, int(n_neg / target_ratio))
    rng = np.random.default_rng(42)

    if desired_pos > n_pos:
        extra_idx     = rng.choice(pos_idx, size=desired_pos - n_pos, replace=True)
        extra_patches = patches[extra_idx].copy()
        extra_patches += rng.normal(0, 0.01, extra_patches.shape).astype(np.float32)
        extra_patches  = np.clip(extra_patches, 0, 1)
        patches = np.concatenate([patches, extra_patches], axis=0)
        targets = np.concatenate([targets, np.ones(len(extra_idx))], axis=0)

    shuffle = rng.permutation(len(targets))
    return patches[shuffle], targets[shuffle]


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
        self.y = torch.tensor(np.array(targets), dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_model(patches, targets, epochs=30, device="cpu"):
    patches = np.array(patches)
    targets = np.array(targets)

    n_pos = int(targets.sum())
    n_neg = int(len(targets) - n_pos)
    print(f"  Before oversampling: whistle={n_pos}, noise={n_neg}  ratio=1:{n_neg//max(n_pos,1)}")

    patches, targets = oversample_minority(patches, targets, target_ratio=5.0)

    n_pos2 = int(targets.sum())
    n_neg2 = int(len(targets) - n_pos2)
    print(f"  After  oversampling: whistle={n_pos2}, noise={n_neg2}  ratio=1:{n_neg2//max(n_pos2,1)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        patches, targets, test_size=0.15, stratify=targets, random_state=42)

    train_dl = DataLoader(SpectrogramDataset(X_tr,  y_tr),  batch_size=64, shuffle=True)
    val_dl   = DataLoader(SpectrogramDataset(X_val, y_val), batch_size=64)

    criterion = nn.BCEWithLogitsLoss()
    model     = WhistleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

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
    print(classification_report(trues, preds, target_names=["noise", "whistle"], zero_division=0))
    return model


# ─────────────────────────────────────────────
# 6. INFERENCE — every 0.1s frame
# ─────────────────────────────────────────────
def predict_frames(model, y, sr, device="cpu"):
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
        for i in range(0, len(patches), 128):
            batch = torch.tensor(np.array(patches[i:i+128]), dtype=torch.float32).to(device)
            probs.extend(torch.sigmoid(model(batch)).cpu().numpy())

    return np.array(frame_times), np.array(probs)


# ─────────────────────────────────────────────
# 7. FRAME CSV
# ─────────────────────────────────────────────
def save_frame_csv(frame_times, probs, threshold, output_dir, wav_stem):
    def to_mmss(s):
        m = int(s) // 60
        return f"{m}:{s - m*60:05.2f}"

    df = pd.DataFrame({
        "time_sec":    frame_times,
        "timestamp":   [to_mmss(t) for t in frame_times],
        "probability": np.round(probs, 4),
        "whistle":     (probs >= threshold).astype(int),
    })
    csv_path = os.path.join(output_dir, f"{wav_stem}_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Frame CSV -> {os.path.basename(csv_path)}")
    return df


# ─────────────────────────────────────────────
# 8. SEGMENT DETECTION FROM STATE FLIPS
# ─────────────────────────────────────────────
def frames_to_segments(frame_times, probs, threshold=0.5,
                        min_gap=MIN_GAP, min_duration=MIN_DURATION):
    labels     = probs >= threshold
    segments   = []
    in_whistle = False
    seg_start  = None

    for t, is_on in zip(frame_times, labels):
        if is_on and not in_whistle:
            seg_start  = t - FRAME_HOP / 2
            in_whistle = True
        elif not is_on and in_whistle:
            segments.append([seg_start, t - FRAME_HOP / 2])
            in_whistle = False

    if in_whistle:
        segments.append([seg_start, frame_times[-1] + FRAME_HOP / 2])

    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        if start - merged[-1][1] <= min_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged if (e - s) >= min_duration]


# ─────────────────────────────────────────────
# 9. SAVE WHISTLE CLIPS
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
        print(f"  [{i+1:04d}] {s:.2f}s – {e:.2f}s  ({e-s:.2f}s)")
    return saved


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Dolphin whistle detector with SEANOE pretraining")
    parser.add_argument("--wav_dir",        required=True,  help="Folder with your own WAV recordings")
    parser.add_argument("--labels",         required=True,  help="Your whistles_cleaned_all.csv")
    parser.add_argument("--seanoe_dir",     required=True,  help="Path to seanoe dataset folder")
    parser.add_argument("--output_dir",     default="whistles_out")
    parser.add_argument("--label_duration", type=float, default=1.5,
                        help="Assumed whistle duration for your CSV (default: 1.5s)")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--threshold",      type=float, default=0.5)
    parser.add_argument("--min_gap",        type=float, default=MIN_GAP)
    parser.add_argument("--min_duration",   type=float, default=MIN_DURATION)
    parser.add_argument("--model_out",      default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Build training data ──
    print(f"\n[1/5] Loading SEANOE training data from: {args.seanoe_dir}")
    seanoe_patches, seanoe_targets = build_seanoe_dataset(args.seanoe_dir)
    print(f"  SEANOE total: {len(seanoe_targets)} patches "
          f"(whistle={sum(seanoe_targets)}, noise={len(seanoe_targets)-sum(seanoe_targets)})")

    print(f"\n[2/5] Loading your own recordings: {args.labels}")
    wav_labels = load_own_labels(args.labels, args.wav_dir, args.label_duration)
    if not wav_labels:
        print("  WARNING: No own recordings matched. Training on SEANOE data only.")
        own_patches, own_targets = [], []
    else:
        print(f"  Matched {len(wav_labels)} WAV file(s):")
        for wp, ldf in wav_labels.items():
            print(f"    {os.path.basename(wp)}  ->  {len(ldf)} labels")
        own_patches, own_targets = build_own_dataset(wav_labels)

    # Combine
    all_patches = seanoe_patches + own_patches
    all_targets = seanoe_targets + own_targets
    print(f"\n  Combined training set: {len(all_targets)} patches "
          f"(whistle={sum(all_targets)}, noise={len(all_targets)-sum(all_targets)})")

    if sum(all_targets) == 0:
        print("ERROR: No whistle patches found.")
        return

    # ── 3. Train ──
    print(f"\n[3/5] Training CNN ({args.epochs} epochs)...")
    model = train_model(all_patches, all_targets, epochs=args.epochs, device=device)
    if args.model_out:
        torch.save(model.state_dict(), args.model_out)
        print(f"  Model saved -> {args.model_out}")

    # ── 4 & 5. Inference on your own recordings ──
    all_saved, all_det_rows = [], []

    for wav_path, _ in wav_labels.items():
        wav_stem = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\n[4-5/5] Inference: {wav_stem}")
        y, sr = librosa.load(wav_path, sr=SR, mono=True)

        frame_times, probs = predict_frames(model, y, sr, device=device)
        save_frame_csv(frame_times, probs, args.threshold, args.output_dir, wav_stem)

        segments = frames_to_segments(frame_times, probs,
                                       threshold=args.threshold,
                                       min_gap=args.min_gap,
                                       min_duration=args.min_duration)
        print(f"  Detected {len(segments)} whistle(s)")

        saved = save_whistle_clips(y, sr, segments, args.output_dir, wav_stem)
        all_saved.extend(saved)
        for s, e in segments:
            all_det_rows.append({"wav": os.path.basename(wav_path),
                                  "start": round(s,3), "end": round(e,3),
                                  "duration": round(e-s,3)})

    det_path = os.path.join(args.output_dir, "detections.csv")
    pd.DataFrame(all_det_rows).to_csv(det_path, index=False)
    print(f"\nAll detections -> {det_path}")
    print(f"Done! {len(all_saved)} whistle clips saved.")


if __name__ == "__main__":
    main()
