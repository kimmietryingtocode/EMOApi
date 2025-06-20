import base64
import requests
import json
import time

API_URL = "http://localhost:8000/score/audio"

def encode_audio_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def test_audio_clip(path, label):
    audio_b64 = encode_audio_to_base64(path)
    payload = {"audio_base64": audio_b64}
    start = time.time()
    response = requests.post(API_URL, json=payload)
    elapsed = round((time.time() - start) * 1000, 2)

    if response.status_code == 200:
        print(f"\n✅ {label} CLIP RESULT:")
        print(json.dumps(response.json(), indent=2))
        print(f"Client-side elapsed: {elapsed} ms")
    else:
        print(f"\n❌ Error ({label}):", response.status_code, response.text)

if __name__ == "__main__":
    # Test both clips only when run as a script:
    test_audio_clip("samples/3s_clip.wav", "3-second")
    test_audio_clip("samples/5s_clip.wav", "5-second")

