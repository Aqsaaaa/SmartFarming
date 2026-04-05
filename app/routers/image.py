import base64
from fastapi import APIRouter, File, UploadFile, HTTPException
from ..ollama_client import generate as ollama_generate

router = APIRouter()

LLAVA_MODEL = "qwen3.5:397b-cloud"

@router.post("", response_model=dict)
async def analyze_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"): # type: ignore
        raise HTTPException(status_code=400, detail="File must be an image")
    raw_bytes = await image.read()
    # Send to Ollama llava model
    description = await ollama_generate(
        prompt="jelaskan gambar ini secara detail",
        model=LLAVA_MODEL,
        images=[raw_bytes],
    )
    return {"description": description.strip()}
