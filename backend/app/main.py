"""
Eye Hear U - Backend API
FastAPI application for ASL sign recognition inference.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load ML model into memory on startup, clean up on shutdown."""
    from app.services.model_service import load_model
    try:
        model, index_to_gloss = load_model(settings)
        application.state.model = model
        application.state.index_to_gloss = index_to_gloss
        print(f"[startup] Model loaded ({len(index_to_gloss)} classes)")
    except FileNotFoundError as e:
        application.state.model = None
        application.state.index_to_gloss = None
        print(f"[startup] Model not loaded: {e}")
    except Exception as e:
        application.state.model = None
        application.state.index_to_gloss = None
        print(f"[startup] Model load failed: {e}")
    print(f"[startup] {settings.app_name} is ready.")
    yield


app = FastAPI(
    title=settings.app_name,
    description="API for ASL-to-English sign translation. Accepts video clips and returns predicted sign labels.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import health, predict  # noqa: E402

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
