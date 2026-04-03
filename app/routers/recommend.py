from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from ..ollama_client import generate as ollama_generate
from ..rag import rag_store

router = APIRouter()

GPT_MODEL = "gpt-oss:120b-cloud"

class RecommendResponse(BaseModel):
    recommendation: str

@router.post("/", response_model=RecommendResponse)
async def get_recommendation(
    prompt: str = Form(...),
    image: UploadFile = File(None),
):
    # Build context from recent records
    context = await rag_store.build_context(prompt)
    image_desc = ""
    if image is not None:
        if not image.content_type.startswith("image/"): # type: ignore
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        raw_bytes = await image.read()
        image_desc = await ollama_generate(
            prompt="Provide a concise description of the image for agronomic purposes.",
            model="llava:13b",
            images=[raw_bytes],
        )
        image_desc = f"Image description: {image_desc.strip()}\n"
    # Compose final prompt for the reasoning model
    final_prompt = (
        f"Context data (weather, sensor, recent):\n{context}\n\n"
        f"{image_desc}"
        f"User request: {prompt}\n"
        "Provide a concise fertilizer recommendation (just the recommendation, no extra explanation)."
    )
    answer = await ollama_generate(prompt=final_prompt, model=GPT_MODEL)
    return RecommendResponse(recommendation=answer.strip())
