# -----------------------------
# 📘 TEXT: Tokenize and Encode (Pandas)
# -----------------------------
from transformers import DistilBertTokenizerFast
import pandas as pd
from pathlib import Path
import librosa
import numpy as np
import os
import cv2
# Load data
text_df = pd.read_parquet("data/silver/goemotions.parquet").head(100)

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize
tokenized = tokenizer(
    list(text_df["raw_text"]),
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="np"
)

# Combine with IDs
text_encoded = pd.DataFrame({
    "id": text_df["id"],
    "input_ids": list(tokenized["input_ids"]),
    "attention_mask": list(tokenized["attention_mask"]),
    "label": text_df["emotion_idx"]
})

text_encoded.to_parquet("data/silver/text_tokenized_100.parquet", index=False)
print("✅ Text encoding complete.")

# -----------------------------
# 🎵 AUDIO: MFCC Feature Extraction (Librosa)
# -----------------------------
# Dummy sample audio paths (update with real)
root_audio_dir = Path("data/raw/Audio_Speech_Actors_01-24")
audio_paths = list(root_audio_dir.rglob("*.wav"))[:100]

rows = []
for path in audio_paths:
    try:
        y, sr = librosa.load(path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        rows.append({
            "id": path.name,
            "actor": path.parent.name,
            "mfcc_features": mfcc.flatten().tolist(),
            "duration_sec": librosa.get_duration(y=y, sr=sr)
        })
    except Exception as e:
        print(f"⚠️ Failed to process {path}: {e}")

audio_df = pd.DataFrame(rows)
audio_df.to_parquet("data/silver/audio_mfcc_100.parquet", index=False)
print("✅ Audio MFCC extraction complete.")

# -----------------------------
# 🖼️ IMAGE: Face Crop with OpenCV
# -----------------------------

label_folder = Path("data/raw/YOLO_format/train/labels")
image_folder = Path("data/raw/YOLO_format/train/images")
face_data = []

# Process first 100 images
for img_path in sorted(list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
)[:100]:
    label_path = label_folder / (img_path.stem + ".txt")
    if not label_path.exists():
        continue

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    found = False
    cropped = None

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_center, y_center, bw, bh = map(float, parts)
            # Convert YOLO normalized coords to pixel values
            cx, cy = int(x_center * w), int(y_center * h)
            bw, bh = int(bw * w), int(bh * h)
            x1, y1 = max(cx - bw // 2, 0), max(cy - bh // 2, 0)
            x2, y2 = min(cx + bw // 2, w), min(cy + bh // 2, h)
            cropped = img[y1:y2, x1:x2]
            found = True
            break  # only crop one face per image

    face_data.append({
        "id": img_path.name,
        "label_path": str(label_path),
        "face_detected": found,
        "image_shape": [h, w, img.shape[2]],
        "cropped": cropped.tolist() if found else None
    })

face_df = pd.DataFrame(face_data)
face_df.to_parquet("data/silver/image_faces_100.parquet", index=False)
print("✅ YOLO-based face cropping complete.")
