import io
import os
import traceback

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..middleware.service_auth import verify_service_token
from ..rag import vector_db

router = APIRouter(tags=["sop_upload"])

LARAVEL_API_URL: str = os.getenv("LARAVEL_API_URL", "http://localhost")
RAG_SERVICE_TOKEN: str = os.getenv("RAG_SERVICE_TOKEN", "")

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


def _extract_text_from_pdf(raw: bytes) -> str:
    import fitz

    doc = fitz.open(stream=raw, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    if not text.strip():
        raise HTTPException(status_code=400, detail="Gagal mengekstrak teks dari PDF")
    return text


def _extract_text_from_docx(raw: bytes) -> str:
    try:
        import docx

        doc = docx.Document(io.BytesIO(raw))
        text = "\n".join(p.text for p in doc.paragraphs)
        if not text.strip():
            raise ValueError("Extracted text from DOCX is empty")
        return text
    except ImportError:
        raise RuntimeError(
            "python-docx is not installed. Install it to process .docx files."
        )


@router.post("")
async def upload_sop_file(
    file: UploadFile = File(...),
    _=Depends(verify_service_token),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipe file '{ext}' tidak didukung. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    raw = await file.read()

    try:
        if ext == ".txt":
            text = raw.decode("utf-8")
        elif ext == ".pdf":
            text = _extract_text_from_pdf(raw)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File teks harus menggunakan encoding UTF-8")

    doc_id = file.filename or "unknown"
    await vector_db.upsert_document(
        doc_id=doc_id,
        text=text,
        metadata={
            "source": doc_id,
            "file_name": doc_id,
            "type": "SOP",
        },
    )

    return {
        "message": "SOP berhasil diupload dan diproses",
        "doc_id": doc_id,
        "text_length": len(text),
    }


class ProcessSOPRequest(BaseModel):
    doc_id: str
    file_name: str
    text: str


class ProcessSOPResponse(BaseModel):
    message: str
    doc_id: str
    chunk_count: int


@router.post("/process", response_model=ProcessSOPResponse)
async def process_sop(
    request: ProcessSOPRequest,
    _=Depends(verify_service_token),
):
    try:
        await vector_db.upsert_document(
            doc_id=request.doc_id,
            text=request.text,
            metadata={
                "source": request.doc_id,
                "file_name": request.file_name,
                "type": "SOP",
            },
        )

        stats = await vector_db.get_stats()
        chunk_count = stats.get("document_count", 0)

        return ProcessSOPResponse(
            message="SOP berhasil diproses dan disimpan ke Vector DB",
            doc_id=request.doc_id,
            chunk_count=chunk_count,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")


class SearchSOPRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/search")
async def search_sop(
    request: SearchSOPRequest,
    _=Depends(verify_service_token),
):
    try:
        results = await vector_db.search_similar(request.query, request.top_k)
        return {
            "query": request.query,
            "results": [
                {"text": doc.text, "metadata": doc.metadata}
                for doc in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stats")
async def sop_stats(
    _=Depends(verify_service_token),
):
    try:
        stats = await vector_db.get_stats()
        return {"vector_db": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
