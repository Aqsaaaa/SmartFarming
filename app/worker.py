import asyncio
import io
import os
import traceback
from typing import Any, Optional

from .laravel_client import LaravelClient
from .rag import vector_db

async def start_background_worker() -> None:
    print(">>> start_background_worker CALLED <<<")

    global _worker_task

    print("WORKER_ENABLED =", os.getenv("WORKER_ENABLED"))

    if os.getenv("WORKER_ENABLED", "true").lower() not in ("true", "1"):
        print("Worker disabled")
        return

POLL_INTERVAL: int = int(os.getenv("WORKER_POLL_INTERVAL", "15"))


class RagWorker:
    """Background worker that polls Laravel for pending RAG documents,
    downloads, chunks, embeds, stores in ChromaDB, and reports status.

    Designed to run as a standalone asyncio process.
    """

    def __init__(self) -> None:
        self.client: LaravelClient = LaravelClient()
        print("CLIENT URL =", self.client.base_url)

    async def _extract_text(self, content: bytes, filename: str) -> str:
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".txt":
            return content.decode("utf-8")

        if ext == ".pdf":
            import fitz

            doc = fitz.open(stream=content, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            if not text.strip():
                raise ValueError("Extracted text from PDF is empty")
            return text

        if ext == ".docx":
            try:
                import docx

                doc = docx.Document(io.BytesIO(content))
                text = "\n".join(p.text for p in doc.paragraphs)
                if not text.strip():
                    raise ValueError("Extracted text from DOCX is empty")
                return text
            except ImportError:
                raise RuntimeError(
                    "python-docx is not installed. Install it to process .docx files."
                )

        raise ValueError(f"Unsupported file extension: {ext}")

    async def process_document(self, doc: dict[str, Any]) -> None:
        doc_id: int = doc["id"]
        original_name: str = doc.get("original_filename", f"doc_{doc_id}")
        stored_filename: str = doc.get("stored_filename", original_name)

        try:
            print(f"[WORKER] Processing: {original_name} (ID: {doc_id})")
            await self.client.update_status(doc_id, "processing")

            content, _ = await self.client.download_document(doc_id)
            text = await self._extract_text(content, stored_filename)

            await vector_db.upsert_document(
                doc_id=str(doc_id),
                text=text,
                metadata={
                    "source": str(doc_id),
                    "file_name": original_name,
                    "type": "RAG",
                },
            )

            stats: dict[str, Any] = await vector_db.get_stats()

            await self.client.update_status(
                doc_id,
                "processed",
                chunk_count=stats.get("document_count", 0),
            )

            print(f"[WORKER] Done: {original_name} (ID: {doc_id})")

        except Exception as exc:
            traceback.print_exc()
            print(f"[WORKER] Failed: {original_name} (ID: {doc_id}): {exc}")
            try:
                await self.client.update_status(
                    doc_id,
                    "failed",
                    error_message=str(exc),
                )
            except Exception:
                pass

    async def run_once(self) -> None:
        try:
            result: dict[str, Any] = await self.client.get_pending_documents()
            docs: list[dict[str, Any]] = result.get("data", [])

            if docs:
                print(f"[WORKER] Found {len(docs)} pending document(s)")
                for doc in docs:
                    await self.process_document(doc)
            else:
                print("[WORKER] No pending documents")

        except Exception as exc:
            print(f"[WORKER] Poll error: {exc}")

    async def run_forever(self) -> None:
        print(f"[WORKER] Starting RAG worker (poll every {POLL_INTERVAL}s)")
        while True:
            await self.run_once()
            await asyncio.sleep(POLL_INTERVAL)


_worker_instance: RagWorker = RagWorker()
_worker_task: Optional[asyncio.Task[None]] = None


async def run_worker_once() -> None:
    await _worker_instance.run_once()


async def run_worker_forever() -> None:
    await _worker_instance.run_forever()


async def start_background_worker() -> None:
    global _worker_task

    if os.getenv("WORKER_ENABLED", "true").lower() not in ("true", "1"):
        print("[WORKER] Disabled via WORKER_ENABLED env var")
        return

    if _worker_task is not None and not _worker_task.done():
        print("[WORKER] Already running")
        return

    async def _wrapped() -> None:
        try:
            await run_worker_forever()
        except asyncio.CancelledError:
            print("[WORKER] Shutting down")
        except Exception:
            traceback.print_exc()

    _worker_task = asyncio.create_task(_wrapped(), name="rag-worker")
    print(f"[WORKER] Launched as background task (poll every {POLL_INTERVAL}s)")


async def stop_background_worker() -> None:
    global _worker_task

    if _worker_task is None or _worker_task.done():
        return

    _worker_task.cancel()
    try:
        await _worker_task
    except asyncio.CancelledError:
        pass
    _worker_task = None
    print("[WORKER] Stopped")
