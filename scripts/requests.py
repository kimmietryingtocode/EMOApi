# requests.py
from fastapi.testclient import TestClient
from app.api.routers.audio import app   # point to where you define `app = FastAPI()`

client = TestClient(app)

from urllib.parse import urlparse

def post(url, *args, **kwargs):
    """
    Redirect requests.post("http://localhost:8000/...") into TestClient.
    """
    parts = urlparse(url)
    path = parts.path
    if parts.query:
        path += "?" + parts.query
    return client.post(path, *args, **kwargs)
