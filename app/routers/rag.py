from fastapi import APIRouter, Depends

from ..middleware.service_auth import verify_service_token
from ..rag import vector_db

router = APIRouter()


@router.delete("/documents/{doc_id}")
async def delete_document_vectors(
    doc_id: int,
    _=Depends(verify_service_token),
):
    await vector_db.delete_document(str(doc_id))
    return {"success": True, "message": f"Vector entries for document {doc_id} deleted"}
