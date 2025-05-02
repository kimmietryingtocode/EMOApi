# ✅ tokenize_goemotions.py — Week 2 Tokenizer
# Updated to work with GoEmotions CSV-based Parquet format

from transformers import DistilBertTokenizerFast
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

# Load cleaned Parquet
df = pd.read_parquet("data/silver/goemotions.parquet")
print(f"✅ Loaded {len(df)} rows from goemotions.parquet")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize text column
tokenized = tokenizer(
    list(df["raw_text"]),
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="np"
)

# Assemble tokenized DataFrame
output = pd.DataFrame({
    "id": df["id"],
    "input_ids": list(tokenized["input_ids"]),
    "attention_mask": list(tokenized["attention_mask"]),
    "label": df["emotion_idx"]
})

# Save tokenized output
Path("data/silver").mkdir(parents=True, exist_ok=True)
output.to_parquet("data/silver/tokenized_goemotions.parquet", index=False)
print("✅ Tokenized GoEmotions and saved to data/silver/tokenized_goemotions.parquet")
