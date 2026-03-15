"""
Gemini Vision Tutor - FastAPI Backend Entry Point
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Gemini Vision Tutor backend...")

    from services.gemini_service import GeminiService
    from services.firestore_service import FirestoreService
    from api.websocket import init_services

    gemini    = GeminiService()
    firestore = FirestoreService()

    init_services(gemini, firestore)

    app.state.gemini    = gemini
    app.state.firestore = firestore

    logger.info("All services ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Gemini Vision Tutor API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from api.websocket   import router as ws_router
from api.live_agent  import router as live_router
from api.tutor_routes import router as tutor_router

# Standard tutor WebSocket (snapshot + REST Gemini)
app.include_router(ws_router)

# NEW: Gemini Live API router (real-time audio + vision)
app.include_router(live_router)

# REST routes
app.include_router(tutor_router, prefix="/api", tags=["REST"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "gemini-vision-tutor",
        "modes": ["standard", "live"]
    }