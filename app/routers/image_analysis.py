from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from ..ollama_client import generate as ollama_generate

router = APIRouter()

VISION_MODEL = "qwen3.5:397b-cloud"

class ImageAnalysisResponse(BaseModel):
    description: str

@router.post("", response_model=ImageAnalysisResponse)
async def analyze_plant_image(image: UploadFile = File(...)):
    """
    Analyze uploaded plant image and provide detailed description
    """
    if not image.content_type.startswith("image/"): # type: ignore
        raise HTTPException(status_code=400, detail="File must be an image")

    raw_bytes = await image.read()

    # Send to Ollama VISION model
    description = await ollama_generate(
        prompt="Analyze this plant image in detail. Identify any signs of disease, pests, nutrient deficiencies, or other issues. Provide specific recommendations for treatment or care., jawab dalam bahasa indonesia yang mudah dipahami oleh petani.",
        model=VISION_MODEL,
        images=[raw_bytes],
    )

    return ImageAnalysisResponse(description=description.strip())
