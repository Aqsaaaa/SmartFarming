import base64
import json
import os
import re

from fastapi import APIRouter, Form, HTTPException
from pydantic import BaseModel

from ..ollama_client import generate as ollama_generate

router = APIRouter()

VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))


class ClassifyDiseaseResponse(BaseModel):
    disease_name: str
    confidence: float | None = None
    explanation: str = ""
    cause: str = ""
    treatment: str = ""
    prevention: str = ""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


@router.post("", response_model=ClassifyDiseaseResponse)
async def classify_disease(
    prompt: str = Form(...),
    image_base64: str = Form(...),
    mime_type: str = Form(...),
):
    try:
        raw_bytes = base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    llm_response = await ollama_generate(
        prompt=prompt,
        model=VISION_MODEL,
        images=[raw_bytes],
        temperature=TEMPERATURE,
    )

    parsed = _extract_json(llm_response)

    return ClassifyDiseaseResponse(
        disease_name=parsed.get("disease_name", parsed.get("disease", parsed.get("result", "Tidak dapat diidentifikasi"))),
        confidence=parsed.get("confidence"),
        explanation=parsed.get("explanation", parsed.get("description", parsed.get("recommendation", ""))),
        cause=parsed.get("cause", parsed.get("reason", "")),
        treatment=parsed.get("treatment", parsed.get("cure", "")),
        prevention=parsed.get("prevention", parsed.get("prevent", "")),
    )
