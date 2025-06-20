# ✅ Feature Extraction Scripts for Text, Audio, and Image + Inference Endpoints

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
import time
import hashlib

app = FastAPI()

# Load ONNX model (placeholder for image)
image_session = ort.InferenceSession("models/v1/image_model.onnx")

# Emotion labels
label_map = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# In-memory cache for detection
image_cache = {}

# Request schema
class ImageRequest(BaseModel):
    image_base64: str

# Endpoint: /score/image
@app.post("/score/image")
def score_image(payload: ImageRequest):
    start_time = time.time()

    # Hash input for caching
    img_hash = hashlib.md5(payload.image_base64.encode()).hexdigest()
    if img_hash in image_cache:
        result = image_cache[img_hash]
        latency_ms = (time.time() - start_time) * 1000
        return {"top_emotions": result, "latency_ms": round(latency_ms, 2), "cached": True}

    try:
        img_bytes = base64.b64decode(payload.image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))  # adjust size as needed
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]  # CHW format
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

    ort_inputs = {image_session.get_inputs()[0].name: img_tensor.astype(np.float32)}
    logits = image_session.run(None, ort_inputs)[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    top_indices = probs.argsort()[::-1][:3]

    result = {label_map[i]: float(probs[i]) for i in top_indices}
    image_cache[img_hash] = result  # store in cache
    latency_ms = (time.time() - start_time) * 1000

    return {"top_emotions": result, "latency_ms": round(latency_ms, 2), "cached": False}
