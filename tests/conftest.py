import asyncio
import os
import sys
from typing import AsyncGenerator, Generator

import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["RAG_SERVICE_TOKEN"] = "test-token"
os.environ["OLLAMA_URL"] = "http://localhost:11434"
os.environ["LARAVEL_API_URL"] = "http://localhost:8001"
os.environ["WORKER_ENABLED"] = "false"

from app.main import app
from app.rag import rag_store, vector_db


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def auth_headers() -> dict:
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
async def mock_sensor_data():
    await rag_store.add("sensor", {
        "temperature": 31.5,
        "soil_moisture": 35,
        "ph": 4.8,
        "air_humidity": 72,
    })
    await rag_store.add("sensor", {
        "temperature": 27.5,
        "soil_moisture": 70,
        "ph": 6.2,
        "air_humidity": 65,
    })
    yield
    rag_store.store.clear()


@pytest.fixture
async def mock_vector_db():
    text = (
        "Tanaman padi membutuhkan kelembapan tanah ideal antara 60-80%. "
        "Jika kelembapan di bawah 40%, tanaman mengalami cekaman kekeringan. "
        "pH tanah ideal untuk padi adalah 5.5-7.0. "
        "Pemupukan NPK dianjurkan pada fase vegetatif dengan dosis 200 kg/ha."
    )
    await vector_db.upsert_document(
        doc_id="test_doc",
        text=text,
        metadata={"source": "test_doc", "type": "SOP"},
    )
    yield
    try:
        await vector_db.delete_document("test_doc")
    except Exception:
        pass
