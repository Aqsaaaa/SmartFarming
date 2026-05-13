import json
import asyncio
import time
import traceback
from collections import deque
from typing import Deque, Dict, List, Optional

# --- IMPORT CHROMA DB UNTUK VECTOR STORE ---
import chromadb
from chromadb.utils import embedding_functions
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

class VectorDBStore:
    def __init__(self):
        import os
        _db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_data")
        os.makedirs(_db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=_db_path)
        
        # Menggunakan model embedding default (all-MiniLM-L6-v2) yang cepat dan ringan
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name="smart_farming_sops",
            embedding_function=self.embedding_fn # type: ignore
        )

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        print("=" * 60)
        print(f"[CHUNKER] Total teks: {len(text)} karakter")
        print(f"[CHUNKER] chunk_size={chunk_size}, overlap={overlap}")
        print(f"[CHUNKER] Menghasilkan {len(chunks)} chunk:")
        print("-" * 60)
        for i, chunk in enumerate(chunks):
            print(f"  Chunk #{i}: {len(chunk)} karakter")
            print(f"    Preview: {chunk[:120]}...")
            if i > 0:
                prev_end = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else chunks[i - 1]
                print(f"    Overlap with prev: {prev_end[:80]}...")
            print()
        print("=" * 60)

        return chunks

    async def upsert_document(self, doc_id: str, text: str, metadata: Optional[dict] = None):
        chunks = self._chunk_text(text)
        if not chunks:
            return

        await self.delete_document(doc_id)

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [metadata or {"source": doc_id} for _ in chunks]

        try:
            await asyncio.to_thread(
                self.collection.add,
                documents=chunks,
                metadatas=metadatas, # pyright: ignore[reportArgumentType]
                ids=ids
            )
        except Exception as e:
            print(f"[RAG ERROR] Gagal menyimpan ke ChromaDB: {e}")
            traceback.print_exc()
            raise

    async def delete_document(self, doc_id: str):
        """Menghapus dokumen dari Vector DB berdasarkan source / nama file."""
        try:
            await asyncio.to_thread(
                self.collection.delete,
                where={"source": doc_id}
            )
        except Exception:
            pass

    async def search_similar(self, query: str, top_k: int = 3) -> List[RetrievedDoc]:
        """Mencari potongan SOP yang paling relevan dengan pertanyaan user."""
        results = await asyncio.to_thread(
            self.collection.query,
            query_texts=[query],
            n_results=top_k
        )

        docs = []
        # Parsing hasil dari ChromaDB ke format RetrievedDoc
        if results and results.get('documents') and results['documents'][0]: # type: ignore
            for text_chunk, meta in zip(results['documents'][0], results['metadatas'][0]): # type: ignore
                docs.append(RetrievedDoc(text=text_chunk, metadata=meta)) # type: ignore

        return docs

    async def get_stats(self) -> dict:
        """Return statistics about the vector database."""
        try:
            count = await asyncio.to_thread(self.collection.count)
            return {
                "document_count": count,
                "collection_name": self.collection.name,
                "status": "active" if count > 0 else "empty"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }

# ==========================================
# 3. INSTANCE GLOBAL
# ==========================================
# Objek ini yang di-import oleh router Anda
rag_store = RAGStore()
vector_db = VectorDBStore()