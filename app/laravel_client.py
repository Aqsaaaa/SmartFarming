import os
from typing import Any, Optional

import httpx

LARAVEL_API_URL: str = os.getenv("LARAVEL_API_URL", "http://localhost:8001")
RAG_SERVICE_TOKEN: str = os.getenv("RAG_SERVICE_TOKEN", "")


class LaravelClient:
    """HTTP client for service-to-service communication with Laravel.

    Uses the shared RAG_SERVICE_TOKEN for authentication.
    Never forwards user credentials or bearer tokens.
    """

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None) -> None:
        self.base_url = (base_url or LARAVEL_API_URL).rstrip("/")
        self.token = token or RAG_SERVICE_TOKEN

        if not self.token:
            raise RuntimeError(
                "RAG_SERVICE_TOKEN is not configured. "
                "Set the RAG_SERVICE_TOKEN environment variable."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    async def get_pending_documents(self) -> dict[str, Any]:
        """GET /api/rag/documents/pending — Poll for unprocessed documents."""
        async with httpx.AsyncClient(headers=self._headers(), timeout=30.0) as client:
            resp = await client.get(f"{self.base_url}/api/rag/documents/pending")
            resp.raise_for_status()
            return resp.json()

    async def download_document(self, doc_id: int) -> tuple[bytes, dict[str, Any]]:
        """GET /api/rag/documents/{id}/download — Download the raw file.

        Returns ``(file_content_bytes, response_json)``.
        """
        async with httpx.AsyncClient(
            headers=self._headers(), timeout=30.0, follow_redirects=True
        ) as client:
            meta_resp = await client.get(
                f"{self.base_url}/api/rag/documents/{doc_id}/download"
            )
            meta_resp.raise_for_status()
            data: dict[str, Any] = meta_resp.json()

            signed_url: str = data["url"]
            file_resp = await client.get(signed_url, timeout=120.0)
            file_resp.raise_for_status()

            return file_resp.content, data

    async def update_status(
        self,
        doc_id: int,
        status: str,
        chunk_count: Optional[int] = None,
        token_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> dict[str, Any]:
        """PATCH /api/rag/documents/{id}/status — Update processing status."""
        payload: dict[str, Any] = {"status": status}

        if chunk_count is not None:
            payload["chunk_count"] = chunk_count
        if token_count is not None:
            payload["token_count"] = token_count
        if error_message is not None:
            payload["error_message"] = error_message

        async with httpx.AsyncClient(headers=self._headers(), timeout=30.0) as client:
            resp = await client.patch(
                f"{self.base_url}/api/rag/documents/{doc_id}/status",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
