import os

import json
import asyncio
import time
import traceback
from collections import deque
from typing import Deque, Dict, List, Optional

# --- IMPORT CHROMA DB UNTUK VECTOR STORE ---
import httpx
import chromadb
from chromadb import EmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. RAG STORE (In-Memory untuk Data Sensor)
# ==========================================
MAX_RECORDS = 5

class RAGStore:
    def __init__(self):
        # category -> deque of (timestamp, data)
        self.store: Dict[str, Deque[tuple[float, dict]]] = {}
        self.lock = asyncio.Lock()

    async def add(self, category: str, data: dict):
        async with self.lock:
            if category not in self.store:
                self.store[category] = deque(maxlen=MAX_RECORDS)
            self.store[category].append((time.time(), data))

    async def get_recent(self, category: str) -> List[dict]:
        async with self.lock:
            if category not in self.store:
                return []
            return [item[1] for item in list(self.store[category])]

    # Default argumen prompt="" ditambahkan agar tidak error jika dipanggil tanpa argumen
    async def build_context(self, prompt: str = "") -> str:
        """Create a simple concatenated context string.
        We pull recent records from all categories and dump them as JSON.
        """
        async with self.lock:
            parts: List[str] = []
            for cat, dq in self.store.items():
                recent = [json.dumps(rec) for _, rec in dq]
                if recent:
                    parts.append(f"Category {cat}: " + ", ".join(recent))
            context = "\n".join(parts)
            return context

# ==========================================
# 2. VECTOR DB STORE (Untuk Dokumen SOP)
# ==========================================
class RetrievedDoc:
    """Helper class untuk menstrukturkan output dari Vector DB agar mudah dibaca di Router"""
    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata

# ==========================================
# 2b. OLLAMA EMBEDDING FUNCTION (via /api/embed)
# ==========================================
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model_name = os.getenv(
            "EMBEDDING_MODEL",
            "nomic-embed-text"
        )

        self.ollama_url = os.getenv(
            "OLLAMA_URL",
            "http://127.0.0.1:11434"
        )

        self.client = httpx.Client(
            timeout=httpx.Timeout(
                connect=30,
                read=300,
                write=300,
                pool=300,
            )
        )

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []

        total = len(input)

        for i, text in enumerate(input):

            for retry in range(3):
                try:
                    print(f"[Embedding] {i+1}/{total}")

                    response = self.client.post(
                        f"{self.ollama_url}/api/embed",
                        json={
                            "model": self.model_name,
                            "input": text
                        }
                    )

                    response.raise_for_status()

                    embeddings.append(
                        response.json()["embeddings"][0]
                    )

                    break

                except Exception as e:

                    print(
                        f"[Embedding Retry {retry+1}] {e}"
                    )

                    if retry == 2:
                        raise

                    time.sleep(5)

        return embeddings


class VectorDBStore:

    def __init__(self):

        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "chroma_data"
        )

        os.makedirs(db_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=db_path
        )

        self.embedding_fn = OllamaEmbeddingFunction()

        collection_name = "smart_farming_sops"

        try:

            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )

        except Exception:

            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )


    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                ".",
                " ",
                ""
            ]
        )

        chunks = splitter.split_text(text)

        print("=" * 60)
        print(f"Total Chunk : {len(chunks)}")
        print("=" * 60)

        return chunks


    async def upsert_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
        batch_size: int = 5,
    ):

        chunks = self._chunk_text(text)

        if not chunks:
            return

        await self.delete_document(doc_id)

        ids = [
            f"{doc_id}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            metadata or {"source": doc_id}
            for _ in chunks
        ]

        total_batch = (
            len(chunks) + batch_size - 1
        ) // batch_size

        for batch in range(total_batch):

            start = batch * batch_size
            end = min(
                start + batch_size,
                len(chunks)
            )

            print(
                f"\n========== Batch {batch+1}/{total_batch} =========="
            )

            await asyncio.to_thread(

                self.collection.add,

                documents=chunks[start:end],

                ids=ids[start:end],

                metadatas=metadatas[start:end],

            )

            print(
                f"Chunk {start+1}-{end} selesai"
            )


    async def delete_document(
        self,
        doc_id: str
    ):

        try:

            await asyncio.to_thread(

                self.collection.delete,

                where={
                    "source": doc_id
                }

            )

        except Exception:

            pass


    async def search_similar(
        self,
        query: str,
        top_k: int = 3
    ):

        result = await asyncio.to_thread(

            self.collection.query,

            query_texts=[query],

            n_results=top_k,

        )

        docs = []

        if result["documents"]:

            for text, meta in zip(

                result["documents"][0],

                result["metadatas"][0],

            ):

                docs.append(
                    RetrievedDoc(
                        text=text,
                        metadata=meta
                    )
                )

        return docs

    async def get_stats(self):

        count = await asyncio.to_thread(
            self.collection.count
        )

        return {
            "document_count": count,
            "collection": self.collection.name,
        }

# ==========================================
# 3. INSTANCE GLOBAL
# ==========================================
# Objek ini yang di-import oleh router Anda
rag_store = RAGStore()
vector_db = VectorDBStore()