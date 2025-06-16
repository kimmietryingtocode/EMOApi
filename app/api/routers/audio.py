from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import librosa
import onnxruntime as ort
import numpy as np
import base64
import io
import soundfile as sf

app = FastAPI()

session = ort.InferenceSession("models/v1/text_model.onnx")

label_map = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
audio_cache = {}
class AudioRequest(BaseModel):
    audio_base64: str

@app.post("/score/audio")
def score_audio(payload: AudioRequest):
    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    audio_hash = hashlib.md5(payload.audio_base64.encode()).hexdigest()
    if audio_hash in audio_cache:
        result = audio_cache[audio_hash]
        latency_ms = (time.time() - start_time) * 1000
        return {
            "timestamp": timestamp,
            "top_emotions": result,
            "latency_ms": round(latency_ms, 2),
            "cached": True
        }

    try:
        audio_bytes = base64.b64decode(payload.audio_base64)
        audio_io = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_io)
        if y.ndim > 1:
            y = y[:, 0]  # convert to mono if stereo
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_flat = mfcc.flatten().astype(np.float32)[np.newaxis, :]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

    ort_inputs = {audio_session.get_inputs()[0].name: mfcc_flat}
    logits = audio_session.run(None, ort_inputs)[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    top_indices = probs.argsort()[::-1][:3]

    result = {label_map[i]: float(probs[i]) for i in top_indices}
    audio_cache[audio_hash] = result
    latency_ms = (time.time() - start_time) * 1000

    return {
        "timestamp": timestamp,
        "top_emotions": result,
        "latency_ms": round(latency_ms, 2),
        "cached": False
    }