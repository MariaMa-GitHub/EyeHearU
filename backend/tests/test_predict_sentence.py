"""Tests for POST /api/v1/predict/sentence."""

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock, patch

import torch

from app.main import app
from app.services.gloss_lm import GlossBeamLM


@pytest.fixture(autouse=True)
def sentence_state():
    app.state.model = MagicMock()
    app.state.index_to_gloss = {0: "hello", 1: "thanks", 2: "water"}
    app.state.gloss_lm = GlossBeamLM.uniform_over_vocab({"hello", "thanks", "water"})
    yield
    app.state.model = None
    app.state.index_to_gloss = None
    app.state.gloss_lm = None


@pytest.mark.asyncio
async def test_predict_sentence_two_clips():
    fake = torch.randn(1, 3, 64, 224, 224)
    clip_hyps = [
        [
            {"sign": "hello", "confidence": 0.9},
            {"sign": "thanks", "confidence": 0.03},
        ],
        [
            {"sign": "water", "confidence": 0.85},
            {"sign": "hello", "confidence": 0.1},
        ],
    ]
    with (
        patch("app.services.preprocessing.preprocess_video", return_value=fake),
        patch(
            "app.services.model_service.predict_batch",
            return_value=clip_hyps,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict/sentence",
                files=[
                    ("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4")),
                    ("files", ("b.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4")),
                ],
            )
    assert r.status_code == 200
    data = r.json()
    assert len(data["clips"]) == 2
    assert data["best_glosses"]
    assert data["english"]
    assert data["english"][-1] in ".?!"
    for row in data["beam"]:
        assert row["english"]
    assert len(data["beam"]) >= 1


@pytest.mark.asyncio
async def test_predict_sentence_too_many_clips():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = [
            ("files", (f"{i}.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))
            for i in range(15)
        ]
        r = await client.post("/api/v1/predict/sentence", files=files)
    assert r.status_code == 400
    assert "12" in r.json()["detail"]


@pytest.mark.asyncio
async def test_predict_sentence_503_without_model():
    app.state.model = None
    app.state.index_to_gloss = None
    app.state.gloss_lm = None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/predict/sentence",
            files=[("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))],
        )
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_predict_sentence_value_error_preprocess():
    with patch(
        "app.services.preprocessing.preprocess_video",
        side_effect=ValueError("bad clip"),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict/sentence",
                files=[("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))],
            )
    assert r.status_code == 400
    assert "bad clip" in r.json()["detail"]


@pytest.mark.asyncio
async def test_predict_sentence_inference_error():
    fake = torch.randn(1, 3, 64, 224, 224)
    with (
        patch("app.services.preprocessing.preprocess_video", return_value=fake),
        patch(
            "app.services.model_service.predict_batch",
            side_effect=RuntimeError("boom"),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict/sentence",
                files=[("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))],
            )
    assert r.status_code == 500


@pytest.mark.asyncio
async def test_predict_sentence_rejects_non_video():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/predict/sentence",
            files=[("files", ("x.jpg", b"\xff\xd8\xff", "image/jpeg"))],
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_predict_sentence_rejects_empty_blob():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/predict/sentence",
            files=[("files", ("a.mp4", b"", "video/mp4"))],
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_predict_sentence_uses_gloss_lm_when_state_none():
    app.state.gloss_lm = None
    fake = torch.randn(1, 3, 64, 224, 224)
    clip_hyps = [[{"sign": "hello", "confidence": 0.99}]]
    with (
        patch("app.services.preprocessing.preprocess_video", return_value=fake),
        patch(
            "app.services.model_service.predict_batch",
            return_value=clip_hyps,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict/sentence",
                files=[("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))],
            )
    assert r.status_code == 200
