import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..rag import vector_db

router = APIRouter(tags=["sop_upload"])

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
