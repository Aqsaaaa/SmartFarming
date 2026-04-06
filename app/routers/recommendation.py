from fastapi import APIRouter, Form
from pydantic import BaseModel
from ..ollama_client import generate as ollama_generate
from ..rag import rag_store

router = APIRouter()

GPT_MODEL = "gpt-oss:120b-cloud"

class RecommendResponse(BaseModel):
    recommendation: str

@router.post("", response_model=RecommendResponse)
async def get_text_recommendation(
    prompt: str = Form(...),
):
    # Build context from recent records
    context = await rag_store.build_context(prompt)

    # Compose final prompt for the reasoning model
    final_prompt = (
        f"Context data (weather, sensor, recent):\n{context}\n\n"
        f"User request: {prompt}\n"
        "kalau nanyanya pake bahasa sunda jawabnya juga pake bahasa sunda, kalau nanya pakai bahasa indonesia jawab dengan bahasa indonesia yang mudah dipahami oleh petani, berikan rekomendasi yang spesifik dan praktis untuk masalah pertanian yang dihadapi."
    )

    answer = await ollama_generate(prompt=final_prompt, model=GPT_MODEL)
    return RecommendResponse(recommendation=answer.strip())