"""
Health and readiness endpoints.

These are used both by developers (manual checks) and by any future
deployment platform (Kubernetes, Render, Railway, etc.) to know whether
the service is alive and whether the ML model has been loaded.
"""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check():
    """Simple liveness probe - if this returns 200, the process is running."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check(request: Request):
    """
    Readiness probe.

    Returns:
        - status: "ready" if the ML model has been attached to app.state,
                  otherwise "initializing".
        - model_loaded: bool flag.
    """
    app = request.app
    model_loaded = getattr(app.state, "model", None) is not None
    return {
        "status": "ready" if model_loaded else "initializing",
        "model_loaded": model_loaded,
    }
