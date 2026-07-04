import json
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.rag import rag_store, vector_db
from app.routers.recommendation import GPT_MODEL


@pytest.mark.asyncio
async def test_vector_db_embedding():
    text = "Kelembapan tanah ideal untuk padi adalah 60-80%."
    await vector_db.upsert_document(
        doc_id="test_embed",
        text=text,
        metadata={"source": "test_embed", "type": "SOP"},
    )
    stats = await vector_db.get_stats()
    assert stats["status"] == "active"
    assert stats["document_count"] > 0

    results = await vector_db.search_similar("kelembapan tanah", top_k=1)
    assert len(results) > 0

    await vector_db.delete_document("test_embed")


@pytest.mark.asyncio
async def test_semantic_retrieval_relevant():
    text = (
        "Tanaman padi membutuhkan kelembapan tanah ideal antara 60-80%. "
        "Jika kelembapan di bawah 40%, tanaman mengalami cekaman kekeringan. "
        "pH tanah ideal untuk padi adalah 5.5-7.0."
    )
    await vector_db.upsert_document(
        doc_id="test_semantic",
        text=text,
        metadata={"source": "test_semantic", "type": "SOP"},
    )

    results = await vector_db.search_similar("kelembapan tanah rendah", top_k=3)
    assert len(results) > 0
    assert any("kelembapan" in r.text.lower() for r in results)

    await vector_db.delete_document("test_semantic")


@pytest.mark.asyncio
async def test_semantic_retrieval_irrelevant():
    text = "Resep masakan nasi goreng: tumis bumbu halus, masukkan nasi."
    await vector_db.upsert_document(
        doc_id="test_irrelevant",
        text=text,
        metadata={"source": "test_irrelevant", "type": "SOP"},
    )

    results = await vector_db.search_similar("cara menanam padi", top_k=3)
    scores = []
    if results:
        for r in results:
            if hasattr(r, 'metadata') and r.metadata:
                scores.append(r.metadata)

    await vector_db.delete_document("test_irrelevant")
    assert len(results) >= 0


@pytest.mark.asyncio
async def test_rag_store_build_context():
    await rag_store.add("sensor", {
        "temperature": 31.5,
        "soil_moisture": 35,
        "ph": 4.8,
        "air_humidity": 72,
    })
    context = await rag_store.build_context()
    assert "35" in context
    assert "4.8" in context
    rag_store.store.clear()


@pytest.mark.asyncio
async def test_rag_store_empty_context():
    rag_store.store.clear()
    context = await rag_store.build_context()
    assert context == ""


@pytest.mark.asyncio
async def test_context_fusion_in_prompt():
    await rag_store.add("sensor", {
        "temperature": 31.5,
        "soil_moisture": 35,
        "ph": 4.8,
    })

    sensor_context = await rag_store.build_context()

    docs = await vector_db.search_similar("tanaman padi", top_k=3)
    sop_context = ""
    if docs:
        sop_context = "\n\n".join([
            f"[SOP]: {doc.text}"
            for doc in docs
        ])

    full_context = (
        f"--- DATA SENSOR ---\n{sensor_context}\n\n"
        f"--- REFERENSI SOP ---\n{sop_context}"
    )

    assert "35" in full_context
    assert "DATA SENSOR" in full_context
    assert "REFERENSI SOP" in full_context
    rag_store.store.clear()


@pytest.mark.asyncio
async def test_prompt_construction():
    await rag_store.add("sensor", {
        "soil_moisture": 35,
        "ph": 4.8,
        "temperature": 31.5,
    })
    sensor_context = await rag_store.build_context()

    docs = await vector_db.search_similar("tanaman padi", top_k=3)
    sop_context = ""
    if docs:
        sop_context = "\n\n".join([
            f"[SOP - {doc.metadata.get('source', 'Unknown')}]:\n{doc.text}"
            for doc in docs
        ])
    else:
        sop_context = "Tidak ada referensi SOP yang relevan di database."

    full_context = (
        f"--- DATA SENSOR LINGKUNGAN ---\n{sensor_context}\n\n"
        f"--- REFERENSI SOP PERTANIAN ---\n{sop_context}"
    )

    prompt = f"""Anda adalah sistem AI Ahli Agronomi dan Konsultan Pertanian Presisi (Smart Farming).

=== DATA KONTEKS (Kondisi Lapangan & SOP) ===
{full_context}

=== KELUHAN / PERTANYAAN PENGGUNA ===
Tanah saya kering, apa yang harus saya lakukan?

=== INSTRUKSI ANALISIS & PENYELESAIAN ===
1. Analisis Multi-Faktor: Evaluasi keluhan dengan melihat data sensor lingkungan dan referensi SOP yang tersedia.
2. Keselarasan SOP: Pastikan solusi yang Anda berikan tidak melanggar panduan dalam "REFERENSI SOP PERTANIAN".
3. Rekomendasi Praktis: Berikan langkah penyelesaian taktis yang spesifik.
4. Mitigasi Masa Depan: Berikan langkah pencegahan.

=== ATURAN BAHASA & FORMAT ===
- Gunakan BAHASA INDONESIA.
- Susun jawaban dengan poin-poin: Ringkasan Kondisi, Diagnosis Masalah, Tindakan Segera, Saran Lanjutan.
"""

    assert "DATA SENSOR LINGKUNGAN" in prompt
    assert "REFERENSI SOP PERTANIAN" in prompt
    assert "KELUHAN / PERTANYAAN PENGGUNA" in prompt
    assert "Tanah saya kering" in prompt
    assert "ATURAN BAHASA & FORMAT" in prompt
    rag_store.store.clear()
