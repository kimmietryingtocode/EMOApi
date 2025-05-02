import pandas as pd
from datetime import datetime
from pathlib import Path

# Load CSV file
df = pd.read_csv("data/raw/goemotions/goemotions_1.csv")

# List of emotion labels in dataset
emotion_cols = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

rows = []

for i, row in df.iterrows():
    # Pick the first emotion with value 1 (one-hot encoded)
    found = [e for e in emotion_cols if row[e] == 1]
    if not found:
        continue
    label = found[0]

    rows.append({
        "id": f"text_{i:05d}",
        "modality": "text",
        "emotion_label": label,
        "emotion_idx": emotion_cols.index(label),
        "source_dataset": "GoEmotions_CSV",
        "split": "train",  # No split info in this CSV, default to train
        "timestamp": datetime.now(),
        "raw_text": row["text"]
    })

# Output as Parquet
parsed_df = pd.DataFrame(rows)
Path("data/silver").mkdir(parents=True, exist_ok=True)
parsed_df.to_parquet("data/silver/goemotions.parquet", index=False)
print(f"✅ Saved {len(parsed_df)} rows to data/silver/goemotions.parquet")
