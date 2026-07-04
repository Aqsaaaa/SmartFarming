import os

from fastapi import APIRouter, Depends, Form
from pydantic import BaseModel

from ..middleware.service_auth import verify_service_token
from ..ollama_client import generate as ollama_generate
from ..rag import rag_store, vector_db

router = APIRouter()

GPT_MODEL = os.getenv("TEXT_MODEL", "gpt-oss:120b-cloud")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))


class RecommendResponse(BaseModel):
    recommendation: str


@router.post("", response_model=RecommendResponse)
async def get_text_recommendation(
    prompt: str = Form(...),
    _=Depends(verify_service_token),
):
    sensor_context = await rag_store.build_context()
    if not sensor_context.strip():
        sensor_context = "Tidak ada data sensor atau kondisi lapangan terbaru saat ini."

    retrieved_docs = await vector_db.search_similar(query=prompt, top_k=3)
    if retrieved_docs:
        sop_context = "\n\n".join([
            f"[SOP - {doc.metadata.get('source', 'Unknown')}]:\n{doc.text}"
            for doc in retrieved_docs
        ])
    else:
        sop_context = "Tidak ada referensi SOP yang relevan di database."

    full_context = (
        f"--- DATA SENSOR LINGKUNGAN ---\n{sensor_context}\n\n"
        f"--- REFERENSI SOP PERTANIAN ---\n{sop_context}"
    )

    final_prompt = f"""Anda adalah sistem AI Ahli Agronomi dan Konsultan Pertanian Presisi (Smart Farming).

=== DATA KONTEKS (Kondisi Lapangan & SOP) ===
{full_context}

=== KELUHAN / PERTANYAAN PENGGUNA ===
{prompt}

=== INSTRUKSI ANALISIS & PENYELESAIAN (Lakukan HANYA Jika Sesuai Topik) ===
1. Analisis Multi-Faktor: Evaluasi keluhan dengan melihat data sensor lingkungan dan referensi SOP yang tersedia.
2. Keselarasan SOP: Pastikan solusi yang Anda berikan tidak melanggar panduan dalam "REFERENSI SOP PERTANIAN".
3. Rekomendasi Praktis: Berikan langkah penyelesaian taktis yang spesifik. Hindari saran teoretis yang mengambang.
4. Mitigasi Masa Depan: Berikan langkah pencegahan.

=== ATURAN BAHASA & FORMAT ===
- JIKA pertanyaan menggunakan BAHASA SUNDA, WAJIB jawab pakai BAHASA SUNDA.
- JIKA pertanyaan menggunakan BAHASA INDONESIA, jawab BAHASA INDONESIA.
- Susun jawaban Anda menggunakan poin-poin agar mudah dibaca di layar HP, dengan struktur:
  - Ringkasan Kondisi
  - Diagnosis Masalah
  - Tindakan Segera
  - Saran Lanjutan
- Berikan jawaban langsung tanpa perlu mengulang pertanyaan atau menambahkan basa-basi AI.
"""

    try:
        answer = await ollama_generate(prompt=final_prompt, model=GPT_MODEL, temperature=TEMPERATURE)
        final_answer = answer.strip()

        if "saya hanya di design untuk menjawab pertanyaan seputar Smart Farming" in final_answer.lower() and len(final_answer) < 100:
            final_answer = "Saya hanya di design untuk menjawab pertanyaan seputar Smart Farming."

        return RecommendResponse(recommendation=final_answer)

    except Exception as e:
        print(f"Error generating response: {e}")
        return RecommendResponse(
            recommendation="Mohon maaf, sistem sedang mengalami gangguan teknis. Silakan coba beberapa saat lagi."
        )
