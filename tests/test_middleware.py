import os
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_auth_missing_header():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/recommend", data={"prompt": "test"})
    assert response.status_code == 401
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_auth_malformed_header():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend",
            data={"prompt": "test"},
            headers={"Authorization": "InvalidFormat token"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_wrong_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend",
            data={"prompt": "test"},
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_empty_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend",
            data={"prompt": "test"},
            headers={"Authorization": "Bearer "},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_auth_valid_token():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/recommend",
            data={"prompt": "test"},
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code in (200, 503)
