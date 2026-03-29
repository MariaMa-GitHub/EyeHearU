"""
Eye Hear U - Backend API
FastAPI application for ASL sign recognition inference.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load ML model into memory on startup, clean up on shutdown."""
    settings = get_settings()
    from app.services.model_service import load_model
    from app.services.gloss_lm import load_gloss_lm

    backend_root = Path(__file__).resolve().parent.parent
    lm_path = backend_root / settings.gloss_lm_path

    try:
        model, index_to_gloss = load_model(settings)
        application.state.model = model
        application.state.index_to_gloss = index_to_gloss
        application.state.gloss_lm = load_gloss_lm(lm_path if lm_path.is_file() else None, index_to_gloss)
        print(f"[startup] Model loaded ({len(index_to_gloss)} classes)")
    except FileNotFoundError as e:
        application.state.model = None
        application.state.index_to_gloss = None
        application.state.gloss_lm = None
        print(f"[startup] Model not loaded: {e}")
    except Exception as e:
        application.state.model = None
        application.state.index_to_gloss = None
        application.state.gloss_lm = None
        print(f"[startup] Model load failed: {e}")

    if settings.gloss_english_mode == "t5":
        try:
            from app.services.gloss_to_english_t5 import _load_t5
            _load_t5()
            print("[startup] T5-small loaded for gloss→English")
        except Exception as e:
            print(f"[startup] T5 load failed (falling back to rule): {e}")
    print(f"[startup] {settings.app_name} is ready.")
    yield


app = FastAPI(
    title=get_settings().app_name,
    description="API for ASL-to-English sign translation. Accepts video clips and returns predicted sign labels.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import health, predict  # noqa: E402

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
