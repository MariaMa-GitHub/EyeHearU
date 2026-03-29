"""
Prediction endpoint.
Accepts video (mp4) upload and returns predicted ASL sign label(s).
Multi-clip sentence decoding uses batched I3D inference + beam search over gloss LM
and a lightweight gloss-line formatter (join + light polish) for ``english``.
"""

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Query
from app.schemas.prediction import (
    PredictionResponse,
    SentenceBeamRow,
    SentenceClipResult,
    SentencePredictionResponse,
    TopKPrediction,
)

router = APIRouter()
logger = logging.getLogger(__name__)

VIDEO_TYPES = ("video/mp4", "video/quicktime", "application/octet-stream")
MAX_SENTENCE_CLIPS = 12


@router.post("/predict", response_model=PredictionResponse)
async def predict_sign(request: Request, file: UploadFile = File(...)):
    """
    Accept a video file (mp4) and return the predicted ASL sign.
    """
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    is_video = (
        file.content_type in VIDEO_TYPES
        or (file.filename or "").lower().endswith((".mp4", ".mov"))
    )
    if not is_video:
        raise HTTPException(
            status_code=400,
            detail=f"Upload a video (mp4/mov). Got content_type={file.content_type}",
        )

    model = getattr(request.app.state, "model", None)
    index_to_gloss = getattr(request.app.state, "index_to_gloss", None)
    if model is None or index_to_gloss is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )

    try:
        from app.services.preprocessing import preprocess_video
        from app.services.model_service import predict as model_predict
        from app.config import get_settings

        video_tensor = preprocess_video(contents)
        results = model_predict(
            model, index_to_gloss, video_tensor,
            top_k=5, device=get_settings().model_device,
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
    )


def _is_video_upload(file: UploadFile, contents: bytes) -> bool:
    return (
        file.content_type in VIDEO_TYPES
        or (file.filename or "").lower().endswith((".mp4", ".mov"))
    ) and len(contents) > 0


@router.post("/predict/sentence", response_model=SentencePredictionResponse)
async def predict_sentence(
    request: Request,
    files: list[UploadFile] = File(..., description="Ordered clips (one sign per file)"),
    beam_size: int = Query(8, ge=1, le=32),
    lm_weight: float = Query(1.0, ge=0.0, le=10.0),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Upload **multiple** video clips in order. Each clip is classified (top-k);
    then beam search + gloss n-gram LM picks a high-scoring gloss sequence.
    The ``english`` field joins ``best_glosses`` with light surface polish only (see
    ``gloss_sequence_to_english``); it is not automatic full-sentence generation.

    Query params: ``beam_size``, ``lm_weight`` (LM vs model balance), ``top_k`` per clip.
    """
    if len(files) > MAX_SENTENCE_CLIPS:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_SENTENCE_CLIPS} clips per request (got {len(files)}).",
        )

    model = getattr(request.app.state, "model", None)
    index_to_gloss = getattr(request.app.state, "index_to_gloss", None)
    gloss_lm = getattr(request.app.state, "gloss_lm", None)
    if model is None or index_to_gloss is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )

    from app.services.gloss_lm import load_gloss_lm

    if gloss_lm is None:
        gloss_lm = load_gloss_lm(None, index_to_gloss)

    raw_blobs: list[bytes] = []
    for f in files:
        contents = await f.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        if not _is_video_upload(f, contents):
            raise HTTPException(
                status_code=400,
                detail=f"Upload videos only (mp4/mov). Got content_type={f.content_type}",
            )
        raw_blobs.append(contents)

    from app.config import get_settings
    from app.services.beam_search import beam_search
    from app.services.model_service import predict_batch
    from app.services.preprocessing import preprocess_video

    settings = get_settings()
    try:
        tensors = [preprocess_video(blob) for blob in raw_blobs]
        clip_hyps = predict_batch(
            model,
            index_to_gloss,
            tensors,
            top_k=top_k,
            device=settings.model_device,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    try:
        beams = beam_search(
            clip_hyps,
            gloss_lm,
            beam_size=beam_size,
            lm_weight=lm_weight,
            top_sequences=5,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    mode = (settings.gloss_english_mode or "rule").strip().lower()

    if mode == "t5":
        from app.services.gloss_to_english import gloss_sequence_to_english as _rule_to_eng
        from app.services.gloss_to_english_t5 import gloss_sequence_to_english_t5 as _t5_to_eng

        def _to_eng(glosses: list[str]) -> str:
            try:
                return _t5_to_eng(glosses)
            except Exception as e:
                logger.warning(
                    "T5 gloss rewrite failed, using rule fallback: %s",
                    e,
                    exc_info=True,
                )
                return _rule_to_eng(glosses)

        # Same as Bedrock: only the best hypothesis pays for a heavy rewrite; beam rows stay rule.
        beam_to_eng = _rule_to_eng
    elif mode == "bedrock":
        from app.services.gloss_to_english import gloss_sequence_to_english as _rule_to_eng
        from app.services.gloss_to_english_bedrock import (
            gloss_sequence_to_english_bedrock as _bedrock_to_eng,
        )

        def _to_eng(glosses: list[str]) -> str:
            try:
                return _bedrock_to_eng(
                    glosses,
                    region=settings.bedrock_region,
                    model_id=settings.bedrock_model_id,
                    timeout_s=settings.bedrock_timeout_s,
                )
            except Exception as e:
                logger.warning(
                    "Bedrock gloss rewrite failed, using rule fallback: %s",
                    e,
                    exc_info=True,
                )
                return _rule_to_eng(glosses)

        beam_to_eng = _rule_to_eng
    else:
        from app.services.gloss_to_english import gloss_sequence_to_english as _to_eng
        beam_to_eng = _to_eng

    clips_out = [
        SentenceClipResult(
            top_k=[TopKPrediction(sign=x["sign"], confidence=x["confidence"]) for x in h]
        )
        for h in clip_hyps
    ]
    beam_rows = [
        SentenceBeamRow(
            glosses=list(b.glosses),
            score=round(b.score, 4),
            english=beam_to_eng(list(b.glosses)),
        )
        for b in beams
    ]
    best = beams[0] if beams else None
    best_glosses = list(best.glosses) if best else []
    english = _to_eng(best_glosses)

    return SentencePredictionResponse(
        clips=clips_out,
        beam=beam_rows,
        best_glosses=best_glosses,
        english=english,
    )
