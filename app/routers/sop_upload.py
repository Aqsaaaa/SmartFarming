import traceback
import os
import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from ..rag import vector_db

router = APIRouter(tags=["sop_upload"])

EXPRESS_API_URL = os.getenv("EXPRESS_API_URL", "http://localhost:3000")

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@router.post("")
async def upload_sop_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipe file '{ext}' tidak didukung. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    raw = await file.read()

    try:
        if ext == ".txt":
            text = raw.decode("utf-8")
        elif ext == ".pdf":
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)  # type: ignore
            doc.close()
            if not text.strip():
                raise HTTPException(status_code=400, detail="Gagal mengekstrak teks dari PDF")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File teks harus menggunakan encoding UTF-8")

    doc_id = file.filename or "unknown"
    await vector_db.upsert_document(
        doc_id=doc_id,
        text=text,
        metadata={
            "source": doc_id,
            "file_name": doc_id,
            "type": "SOP"
        }
    )

    return {
        "message": "SOP berhasil diupload dan diproses",
        "doc_id": doc_id,
        "text_length": len(text)
    }

@router.post("/sync/{doc_id}")
async def sync_sop_from_express(doc_id: int):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(f"{EXPRESS_API_URL}/api/sop/{doc_id}")
            if resp.status_code == 404:
                raise HTTPException(404, "Dokumen tidak ditemukan di Express API")
            resp.raise_for_status()
            doc = resp.json()["data"]
        except httpx.RequestError:
            raise HTTPException(502, "Gagal terhubung ke Express API")

        try:
            file_resp = await client.get(f"{EXPRESS_API_URL}/api/sop/{doc_id}/file")
            if file_resp.status_code == 404:
                raise HTTPException(404, "File tidak ditemukan di Express API")
            file_resp.raise_for_status()
            raw = file_resp.content
        except httpx.RequestError:
            raise HTTPException(502, "Gagal mengunduh file dari Express API")

        ext = os.path.splitext(doc["file_name"])[1].lower()
        try:
            if ext == ".txt":
                text = raw.decode("utf-8")
            elif ext == ".pdf":
                import fitz
                pdf = fitz.open(stream=raw, filetype="pdf")
                text = "\n".join(page.get_text() for page in pdf) # type: ignore
                pdf.close()
            else:
                await client.patch(f"{EXPRESS_API_URL}/api/sop/{doc_id}", json={"status": "Failed", "chunk_count": 0})
                raise HTTPException(400, f"Tipe file '{ext}' tidak didukung")
        except UnicodeDecodeError:
            raise HTTPException(400, "File teks harus menggunakan encoding UTF-8")

        if not text.strip():
            await client.patch(f"{EXPRESS_API_URL}/api/sop/{doc_id}", json={"status": "Failed", "chunk_count": 0})
            raise HTTPException(400, "Teks kosong setelah ekstraksi")

        try:
            await vector_db.upsert_document(
                doc_id=str(doc_id),
                text=text,
                metadata={
                    "source": str(doc_id),
                    "file_name": doc["file_name"],
                    "type": "SOP"
                }
            )

            stats = await vector_db.get_stats()
            chunk_count = stats.get("document_count", 0)

            await client.patch(
                f"{EXPRESS_API_URL}/api/sop/{doc_id}",
                json={"status": "Processed", "chunk_count": chunk_count}
            )

            return {
                "message": "SOP berhasil disinkronisasi",
                "doc_id": doc_id,
                "chunk_count": chunk_count,
                "text_length": len(text)
            }
        except Exception as e:
            traceback.print_exc()
            await client.patch(
                f"{EXPRESS_API_URL}/api/sop/{doc_id}",
                json={"status": "Failed", "chunk_count": 0}
            )
            raise HTTPException(500, f"RAG processing failed: {str(e)}")


class ProcessSOPRequest(BaseModel):
    doc_id: str
    file_name: str
    text: str

class ProcessSOPResponse(BaseModel):
    message: str
    doc_id: str
    chunk_count: int

@router.post("/process", response_model=ProcessSOPResponse)
async def process_sop(request: ProcessSOPRequest):
    try:
        await vector_db.upsert_document(
            doc_id=request.doc_id,
            text=request.text,
            metadata={
                "source": request.doc_id,
                "file_name": request.file_name,
                "type": "SOP"
            }
        )

        stats = await vector_db.get_stats()
        chunk_count = stats.get("document_count", 0)

        return ProcessSOPResponse(
            message="SOP berhasil diproses dan disimpan ke Vector DB",
            doc_id=request.doc_id,
            chunk_count=chunk_count
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")

class SearchSOPRequest(BaseModel):
    query: str
    top_k: int = 3

@router.post("/search")
async def search_sop(request: SearchSOPRequest):
    try:
        results = await vector_db.search_similar(request.query, request.top_k)
        return {
            "query": request.query,
            "results": [
                {"text": doc.text, "metadata": doc.metadata}
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/stats")
async def sop_stats():
    try:
        stats = await vector_db.get_stats()
        return {"vector_db": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
