import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .routers import image_analysis, weather, sensor, recommendation, sop_upload
from .worker import start_background_worker, stop_background_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_background_worker()
    yield
    await stop_background_worker()


app = FastAPI(
    title="Smart Farming API",
    version="0.1.0",
    redirect_slashes=False,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(weather.router, prefix="/weather", tags=["weather"])
app.include_router(sensor.router, prefix="/sensor", tags=["sensor"])
app.include_router(image_analysis.router, prefix="/analyze-image", tags=["image analysis"])
app.include_router(recommendation.router, prefix="/recommend", tags=["text recommendation"])
app.include_router(sop_upload.router, prefix="/api/sop", tags=["sop_upload"])


@app.get("/")
async def root():
    return {"message": "Smart Farming API is running"}
