from fastapi import APIRouter
from ..rag import rag_store

router = APIRouter()

DUMMY_SENSOR_DATA = {
    "soil_moisture": 23.5,  # percent
    "nitrogen": 12.0,       # ppm
    "phosphorus": 8.2,      # ppm
    "potassium": 15.4,      # ppm
    "temperature": 18.7,    # Celsius
}

@router.get("/", response_model=dict)
async def get_sensor_data():
    # In a real implementation this would proxy to a sensor gateway.
    await rag_store.add("sensor", DUMMY_SENSOR_DATA)
    return DUMMY_SENSOR_DATA
