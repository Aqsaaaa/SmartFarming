import json
import asyncio
import time
from collections import deque
from typing import Deque, Dict, List, Optional

# --- IMPORT CHROMA DB UNTUK VECTOR STORE ---
import chromadb
from chromadb.utils import embedding_functions

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
        # PersistentClient menyimpan database dalam folder lokal secara otomatis
        self.client = chromadb.PersistentClient(path="./chroma_data")
        
        # Menggunakan model embedding default (all-MiniLM-L6-v2) yang cepat dan ringan
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name="smart_farming_sops",
            embedding_function=self.embedding_fn # type: ignore
        )

    def _chunk_text(self, text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
        """Memecah teks panjang menjadi paragraf pendek agar hasil pencarian RAG akurat."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    async def upsert_document(self, doc_id: str, text: str, metadata: Optional[dict] = None):
        """Fungsi ini digunakan di Endpoint SOP Anda untuk menyimpan teks ke Vector DB."""
        chunks = self._chunk_text(text)
        if not chunks:
            return

        # Hapus dokumen lama jika ada (agar saat file di-update, tidak duplikat)
        await self.delete_document(doc_id)

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [metadata or {"source": doc_id} for _ in chunks]

        # Jalankan di background thread agar tidak memblokir FastAPI
        await asyncio.to_thread(
            self.collection.add,
            documents=chunks,
            metadatas=metadatas, # type: ignore
            ids=ids
        )

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

# ==========================================
# 3. INSTANCE GLOBAL
# ==========================================
# Objek ini yang di-import oleh router Anda
rag_store = RAGStore()
vector_db = VectorDBStore()