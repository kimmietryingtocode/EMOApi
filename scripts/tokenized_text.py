import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("models/v1/text_model.onnx")

# Inference on first 5 samples
inference_rows = []
for _, row in text_encoded.head(5).iterrows():
    ort_inputs = {
        "input_ids": np.array([row["input_ids"]], dtype=np.int64),
        "attention_mask": np.array([row["attention_mask"]], dtype=np.int64)
    }
    logits = session.run(None, ort_inputs)[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred_idx = int(np.argmax(probs))
    inference_rows.append({
        "id": row["id"],
        "predicted_label": pred_idx,
        "confidence": float(probs[pred_idx])
    })

inference_df = pd.DataFrame(inference_rows)
inference_df.to_parquet("data/silver/text_onnx_predictions.parquet", index=False)
print("✅ ONNX inference complete.")