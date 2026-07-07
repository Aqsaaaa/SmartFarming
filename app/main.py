import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(override=True)

from .routers import classify_disease, rag, recommendation
from .worker import start_background_worker, stop_background_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== LIFESPAN START ===")
    await start_background_worker()
    yield
    print("=== LIFESPAN STOP ===")
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

app.include_router(classify_disease.router, prefix="/classify-disease", tags=["disease classification"])
app.include_router(recommendation.router, prefix="/recommend", tags=["text recommendation"])
app.include_router(rag.router, prefix="/rag", tags=["rag"])


@app.get("/")
async def root():
    return {"message": "Smart Farming API is running"}
