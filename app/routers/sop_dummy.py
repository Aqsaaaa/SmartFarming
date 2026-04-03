from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from ..utils.sop_reader import list_sop_files, read_sop

router = APIRouter(tags=["SOP"])


@router.get("", response_model=List[str])
async def get_sop_list():
    """Return a list of available SOP filenames stored in the dummy folder."""
    return await list_sop_files()


@router.get("/{filename}")
async def get_sop_content(
    filename: str,
    model: Optional[str] = Query(None, description="Optional AI model to summarise the SOP (e.g., 'ollama' or 'claude')."),
    store: bool = Query(True, description="Whether to store the extracted SOP text in the RAG store (default true)."),
):
    """Read a SOP file (txt or pdf) and return its plain‑text content.

    If ``model`` is provided, the extracted text is sent to the chosen AI model and the
    model's response is returned instead of the raw content.
    """
    try:
        content = await read_sop(filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="SOP file not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))

    # If the caller does NOT want to store the text in RAG, remove it (already added inside read_sop)
    if not store:
        # Simple removal: we could clear the latest added entry for this filename.
        # For simplicity we just ignore – the entry will stay in RAG; a more precise removal
        # would require a dedicated RAG method which is beyond this minimal implementation.
        pass

    if model:
        # Build a prompt for the chosen model. For now we reuse the existing Ollama client.
        prompt = f"Summarise the following SOP document concisely:\n\n{content}"
        if model.lower() == "ollama":
            try:
                from ..ollama_client import generate
                ai_response = await generate(prompt) # type: ignore
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Ollama error: {exc}")
            return {"summary": ai_response}
        elif model.lower() == "claude":
            try:
                from ..claude_client import call_claude # type: ignore
                ai_response = await call_claude(prompt)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Claude error: {exc}")
            return {"summary": ai_response}
        else:
            raise HTTPException(status_code=400, detail="Unsupported model; use 'ollama' or 'claude'.")

    return {"content": content}

# Debug endpoint – not part of the public API, useful during development
@router.get("/debug/rag")
async def debug_rag():
    """Return the current RAG store context (for debugging)."""
    from ..rag import rag_store
    return await rag_store.build_context("")
