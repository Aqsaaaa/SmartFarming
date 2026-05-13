import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import image_analysis, weather, sensor, recommendation, sop_upload

app = FastAPI(title="Smart Farming API", version="0.1.0", redirect_slashes=False)

# Allow all origins for simplicity (can be restricted later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(weather.router, prefix="/weather", tags=["weather"])
app.include_router(sensor.router, prefix="/sensor", tags=["sensor"])
app.include_router(image_analysis.router, prefix="/analyze-image", tags=["image analysis"])
app.include_router(recommendation.router, prefix="/recommend", tags=["text recommendation"])
app.include_router(sop_upload.router, prefix="/api/sop", tags=["sop_upload"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Smart Farming API is running"}
