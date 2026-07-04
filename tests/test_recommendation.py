import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_recommend_valid():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend",
            data={"prompt": "Tanah saya kering, apa yang harus saya lakukan?"},
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "recommendation" in data
    assert isinstance(data["recommendation"], str)
    assert len(data["recommendation"]) > 0


@pytest.mark.asyncio
async def test_recommend_no_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend",
            data={"prompt": "test"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_recommend_invalid_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend",
            data={"prompt": "test"},
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert response.status_code == 401
