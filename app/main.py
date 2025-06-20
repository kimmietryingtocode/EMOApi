from fastapi import FastAPI
from app.routers import status  # Import your router
from app.middleware.logging import LoggingMiddelware
from app.routers import text


app = FastAPI(
    title="EMOApi",
    description="Multi-modal Emotion Classification API",
    version="1.0.0",
)

app.add_middleware(LoggingMiddelware)
app.include_router(text.router)


# Register the router with the app
app.include_router(status.router)
app.include_router(text.router, prefix="/score", tags=["text"])


# Root endpoint for basic info
@app.get("/")
def read_root():
    return {"message": "Welcome to EMOApi!"}
