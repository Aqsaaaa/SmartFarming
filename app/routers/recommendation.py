from fastapi import APIRouter, Form
from pydantic import BaseModel
from ..ollama_client import generate as ollama_generate

# IMPORT KEDUA STORE: Untuk Hybrid RAG (Sensor Real-time + Dokumen SOP)
from ..rag import rag_store, vector_db 

router = APIRouter()

# Anda dapat mengubah string model ini sesuai dengan model terbaru yang aktif di server Anda 
# (misal: "llama3-70b-instruct", "mixtral:8x7b", dll)
GPT_MODEL = "qwen3.5:397b-cloud"

class RecommendResponse(BaseModel):
    recommendation: str

@router.post("", response_model=RecommendResponse)
async def get_text_recommendation(
    prompt: str = Form(...),
):
    # 1. AMBIL DATA SENSOR/CUACA (Dari memori/rag_store)
    sensor_context = await rag_store.build_context()
    if not sensor_context.strip():
        sensor_context = "Tidak ada data sensor atau kondisi lapangan terbaru saat ini."

    # 2. AMBIL REFERENSI SOP (Dari Vector Database berdasarkan semantic search)
    retrieved_docs = await vector_db.search_similar(query=prompt, top_k=3)
    if retrieved_docs:
        sop_context = "\n\n".join([f"[SOP - {doc.metadata.get('source', 'Unknown')}]:\n{doc.text}" for doc in retrieved_docs])
    else:
        sop_context = "Tidak ada referensi SOP yang relevan di database."

    # 3. GABUNGKAN MENJADI FULL KONTEKS
    full_context = f"--- DATA SENSOR LINGKUNGAN ---\n{sensor_context}\n\n--- REFERENSI SOP PERTANIAN ---\n{sop_context}"

    # 4. AUGMENTATION: Menyusun full prompt dengan GUARDRAIL KETAT
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
  - 📊 Ringkasan Kondisi
  - 🔍 Diagnosis Masalah
  - 🛠️ Tindakan Segera
  - 🛡️ Saran Lanjutan
- Berikan jawaban langsung tanpa perlu mengulang pertanyaan atau menambahkan basa-basi AI.
"""

    # 5. GENERATION
    try:
        answer = await ollama_generate(prompt=final_prompt, model=GPT_MODEL)
        # Membersihkan spasi kosong ekstra di awal/akhir respons
        final_answer = answer.strip()
        
        # Fallback tambahan (Hardcode Guard) jika LLM mencoba menipu aturan di atas
        if "saya hanya di design untuk menjawab pertanyaan seputar Smart Farming" in final_answer.lower() and len(final_answer) < 100:
            final_answer = "Saya hanya di design untuk menjawab pertanyaan seputar Smart Farming."
            
        return RecommendResponse(recommendation=final_answer)
    
    except Exception as e:
        # Menangkap error dari model/Ollama agar server tidak crash
        print(f"Error generating response: {e}")
        return RecommendResponse(recommendation="Mohon maaf, sistem sedang mengalami gangguan teknis. Silakan coba beberapa saat lagi.")
    

#     === BATASAN TOPIK (SANGAT PENTING) ===
# Evaluasi pertanyaan pengguna di bawah ini. Jika pertanyaan TIDAK MEMILIKI HUBUNGAN dengan Smart Farming, pertanian, agronomi, tanaman, hama, pupuk, cuaca pertanian, atau sensor lingkungan, Anda DILARANG KERAS menjawabnya. 
# Jika di luar topik, Anda WAJIB membatalkan analisis dan HANYA menjawab dengan kalimat ini (tanpa tambahan apapun):
# "saya hanya di design untuk menjawab pertanyaan seputar Smart Farming"