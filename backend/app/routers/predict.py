"""
Prediction endpoints.
Accepts video (or image) upload and returns predicted ASL sign label(s).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from app.schemas.prediction import PredictionResponse

router = APIRouter()

# Content types we accept for video inference
VIDEO_TYPES = ("video/mp4", "video/quicktime", "application/octet-stream")


@router.post("/predict", response_model=PredictionResponse)
async def predict_sign(request: Request, file: UploadFile = File(...)):
    """
    Accept a video file (mp4) and return the predicted ASL sign.
    Model must be loaded at startup (best_model.pt + label_map.json in same dir).
    """
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Prefer video; treat unknown/octet-stream as potential mp4
    is_video = (
        file.content_type in VIDEO_TYPES
        or (file.filename or "").lower().endswith(".mp4")
    )
    if not is_video:
        return PredictionResponse(
            sign="",
            confidence=0.0,
            top_k=[],
            message="Upload a video (mp4) for sign recognition.",
        )

    model = getattr(request.app.state, "model", None)
    index_to_gloss = getattr(request.app.state, "index_to_gloss", None)
    if model is None or index_to_gloss is None:
        return PredictionResponse(
            sign="",
            confidence=0.0,
            top_k=[],
            message="Model not loaded. Place best_model.pt and label_map.json in model_path dir.",
        )

    try:
        from app.services.preprocessing import preprocess_video
        from app.services.model_service import predict as model_predict
        from app.config import get_settings
        settings = get_settings()
        video_tensor = preprocess_video(contents)
        results = model_predict(
            model, index_to_gloss, video_tensor,
            top_k=5, device=settings.model_device,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    top_1 = results[0] if results else {"sign": "", "confidence": 0.0}
    return PredictionResponse(
        sign=top_1["sign"],
        confidence=top_1["confidence"],
        top_k=results,
        message=None,
    )
