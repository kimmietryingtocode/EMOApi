from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import DistilBertTokenizerFast
import onnxruntime as ort
import numpy as np
import time
from datetime import datetime
import hashlib
import logging

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging
logging.basicConfig(level=logging.INFO)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    process_time = round((time.time() - start) * 1000, 2)
    logging.info(f"{request.method} {request.url} - {process_time}ms")
    return response

# Load tokenizer and ONNX model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
session = ort.InferenceSession("models/v1/text_model.onnx")

# Emotion labels
label_map = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Cache
text_cache = {}

class TextRequest(BaseModel):
    text: str
    threshold: float = 0.1  # Optional confidence threshold

@app.post("/score/text")
def score_text(payload: TextRequest):
    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input is empty")

    # Cache key
    key = f"{text}-{payload.threshold}"
    hash_key = hashlib.md5(key.encode()).hexdigest()
    if hash_key in text_cache:
        result = text_cache[hash_key]
        latency = round((time.time() - start_time) * 1000, 2)
        return {
            "timestamp": timestamp,
            "top_emotions": result,
            "latency_ms": latency,
            "cached": True
        }

    # Tokenize + inference
    inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=64)
    ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    logits = session.run(None, ort_inputs)[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))

    # Filter by threshold and return top 3
    top_indices = probs.argsort()[::-1]
    top_emotions = {label_map[i]: float(probs[i]) for i in top_indices if probs[i] >= payload.threshold}
    top3 = dict(list(top_emotions.items())[:3])

    text_cache[hash_key] = top3
    latency = round((time.time() - start_time) * 1000, 2)

    return {
        "timestamp": timestamp,
        "top_emotions": top3,
        "latency_ms": latency,
        "cached": False
    }
