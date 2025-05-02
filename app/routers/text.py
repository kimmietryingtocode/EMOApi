from fastapi import APIRouter
from app.schemas.text import TextInput

router = APIRouter()


@router.post("/score/text")
def score_text(payload: TextInput):
    return {"received_text": payload.text}
