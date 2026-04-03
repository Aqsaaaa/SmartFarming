import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import weather, sensor, image, recommend, sop_dummy

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
app.include_router(image.router, prefix="/image", tags=["image"])
app.include_router(recommend.router, prefix="/recommend", tags=["recommend"])
app.include_router(sop_dummy.router, prefix="/sop", tags=["sop"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Smart Farming API is running"}
