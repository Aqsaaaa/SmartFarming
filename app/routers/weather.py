import httpx
from fastapi import APIRouter, Query, HTTPException
from ..rag import rag_store

router = APIRouter()

OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

@router.get("", response_model=dict)
async def get_weather(
    lat: float = Query(..., description="Latitude in decimal degrees"),
    lon: float = Query(..., description="Longitude in decimal degrees"),
    hourly: str = Query("temperature_2m,precipitation", description="Comma‑separated hourly variables"),
):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly,
        "current_weather": True,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_BASE, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch weather data")
        data = resp.json()
    # Store a simplified version in RAG store
    await rag_store.add("weather", data)
    return data
