from transformers import DistilBertTokenizerFast
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
input_df = pd.read_parquet("data/silver/goemotions.parquet")

tokenized = tokenizer(
    list(input_df["raw_text"]),
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="np"
)

# Store tokenized features + labels
output = pd.DataFrame({
    "id": input_df["id"],
    "input_ids": list(tokenized["input_ids"]),
    "attention_mask": list(tokenized["attention_mask"]),
    "label": input_df["emotion_idx"]
})
output.to_parquet("data/silver/tokenized_goemotions.parquet", index=False)
print("✅ Tokenized GoEmotions and saved.")
