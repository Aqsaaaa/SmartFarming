import os
import base64
import json
from typing import List, Optional

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

async def _post(endpoint: str, payload: dict) -> dict:
    # Set a generous timeout because llava image generation can take >1 minute.
    # httpx Timeout can be a single float (applies to all phases) or a Timeout object.
    timeout_cfg = httpx.Timeout(300.0, read=300.0)  # 5 minutes total, 5 minutes read
    async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=timeout_cfg) as client:
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()

async def generate(prompt: str, model: str, images: Optional[List[bytes]] = None, temperature: Optional[float] = None) -> str:
    """Generate a response from an Ollama model.
    If ``images`` is provided, they are base64‑encoded and sent as ``image`` fields.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if temperature is not None:
        payload["temperature"] = temperature
    if images:
        # Ollama expects base64 strings
        payload["images"] = [base64.b64encode(img).decode("utf-8") for img in images]
    # Try the generate endpoint first; if it returns 404, fall back to the chat endpoint
    try:
        result = await _post("/api/generate", payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            # Convert payload to chat format
            chat_payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            if images:
                chat_payload["images"] = payload.get("images")
            result = await _post("/api/chat", chat_payload)
        else:
            raise

    # Ollama may return "response" (generate) or "message" (chat)
    if isinstance(result, dict):
        if "response" in result:
            return result["response"]
        if "message" in result and isinstance(result["message"], dict) and "content" in result["message"]:
            return result["message"]["content"]
    # Fallback to JSON dump for debugging
    return json.dumps(result)
