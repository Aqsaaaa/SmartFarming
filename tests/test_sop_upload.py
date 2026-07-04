import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_upload_sop_txt():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/sop",
            files={"file": ("test_sop.txt", b"Kelembapan tanah ideal: 60-80%", "text/plain")},
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "SOP berhasil diupload dan diproses"
    assert data["doc_id"] == "test_sop.txt"
    assert data["text_length"] > 0


@pytest.mark.asyncio
async def test_upload_sop_unsupported_extension():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/sop",
            files={"file": ("test.exe", b"some binary", "application/octet-stream")},
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_upload_sop_no_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/sop",
            files={"file": ("test.txt", b"content", "text/plain")},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_sop_stats():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/sop/stats",
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "vector_db" in data


@pytest.mark.asyncio
async def test_sop_stats_no_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/sop/stats")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_search_sop():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/sop/search",
            json={"query": "kelembapan tanah", "top_k": 3},
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data


@pytest.mark.asyncio
async def test_search_sop_no_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/sop/search",
            json={"query": "test"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_process_sop():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/sop/process",
            json={
                "doc_id": "test_process",
                "file_name": "test.pdf",
                "text": "Panduan budidaya padi sawah. Kelembapan ideal 60-80%.",
            },
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["doc_id"] == "test_process"
