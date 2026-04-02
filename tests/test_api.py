import pytest
from httpx import AsyncClient
from fastapi import status
from app.main import app

@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        resp = await client.get("/")
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["message"] == "Smart Farming API is running"

@pytest.mark.asyncio
async def test_sensor_endpoint():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        resp = await client.get("/sensor")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        # Verify expected keys exist
        for key in ["soil_moisture", "nitrogen", "phosphorus", "potassium", "temperature"]:
            assert key in data

# The weather endpoint hits an external API; we skip it in unit tests to avoid network calls.
# The image and recommendation endpoints require Ollama models; they are also skipped here.
