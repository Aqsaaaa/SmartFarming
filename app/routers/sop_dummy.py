from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from ..utils.sop_reader import list_sop_files, read_sop
# Asumsi: Anda memiliki wrapper vector_db di dalam modul rag Anda
from ..rag import vector_db 

router = APIRouter(tags=["SOP"])

@router.get("", response_model=List[str])
async def get_sop_list():
    """Return a list of available SOP filenames stored in the dummy folder."""
    return await list_sop_files()


@router.get("/{filename}")
async def get_sop_content(
    filename: str,
    model: Optional[str] = Query(None, description="Optional AI model to summarise the SOP (e.g., 'ollama' or 'claude')."),
    store: bool = Query(True, description="Whether to store the extracted SOP text in the Vector DB (default true)."),
):
    """Read a SOP file (txt or pdf) and return its plain-text content.

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

    # VECTOR DB INTEGRATION: Menyimpan atau menghapus dari Vector DB
    if store:
        try:
            # Upsert akan melakukan chunking, embedding, dan menyimpan ke Vector DB
            await vector_db.upsert_document(
                doc_id=filename, 
                text=content, 
                metadata={"source": filename, "type": "SOP"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to store in Vector DB: {str(e)}")
    else:
        # Jika caller tidak ingin menyimpan (atau ingin menghapus jika sudah ada)
        try:
            await vector_db.delete_document(doc_id=filename)
        except Exception:
            pass # Abaikan jika dokumen memang tidak ada

    if model:
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

@router.get("/debug/rag")
async def debug_rag():
    """Return the current Vector DB stats (for debugging)."""
    # Mengambil statistik dari Vector DB (misal: jumlah chunks/dokumen)
    stats = await vector_db.get_stats() # type: ignore
    return {"vector_db_stats": stats}