"""POST /predict/sentence coverage for gloss_english_mode branches."""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.gloss_lm import GlossBeamLM


def _mock_settings(**kwargs) -> MagicMock:
    s = MagicMock()
    s.gloss_english_mode = kwargs.get("gloss_english_mode", "rule")
    s.model_device = "cpu"
    s.bedrock_region = kwargs.get("bedrock_region", "ca-central-1")
    s.bedrock_model_id = kwargs.get("bedrock_model_id", "anthropic.fake")
    s.bedrock_timeout_s = 20.0
    return s


@pytest.fixture(autouse=True)
def sentence_state():
    app.state.model = MagicMock()
    app.state.index_to_gloss = {0: "hello", 1: "thanks"}
    app.state.gloss_lm = GlossBeamLM.uniform_over_vocab({"hello", "thanks"})
    yield
    app.state.model = None
    app.state.index_to_gloss = None
    app.state.gloss_lm = None


@pytest.mark.asyncio
async def test_sentence_mode_t5_uses_t5_for_english():
    fake = torch.randn(1, 3, 64, 224, 224)
    clip_hyps = [[{"sign": "hello", "confidence": 0.9}]]
    with (
        patch("app.services.preprocessing.preprocess_video", return_value=fake),
        patch("app.services.model_service.predict_batch", return_value=clip_hyps),
        patch(
            "app.config.get_settings",
            return_value=_mock_settings(gloss_english_mode="t5"),
        ),
        patch(
            "app.services.gloss_to_english_t5.gloss_sequence_to_english_t5",
            return_value="From t5.",
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict/sentence",
                files=[("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))],
            )
    assert r.status_code == 200
    assert r.json()["english"] == "From t5."


@pytest.mark.asyncio
async def test_sentence_mode_bedrock_success():
    fake = torch.randn(1, 3, 64, 224, 224)
    clip_hyps = [[{"sign": "hello", "confidence": 0.9}]]
    with (
        patch("app.services.preprocessing.preprocess_video", return_value=fake),
        patch("app.services.model_service.predict_batch", return_value=clip_hyps),
        patch(
            "app.config.get_settings",
            return_value=_mock_settings(gloss_english_mode="bedrock"),
        ),
        patch(
            "app.services.gloss_to_english_bedrock.gloss_sequence_to_english_bedrock",
            return_value="From Bedrock.",
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict/sentence",
                files=[("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))],
            )
    assert r.status_code == 200
    assert r.json()["english"] == "From Bedrock."


@pytest.mark.asyncio
async def test_sentence_mode_bedrock_error_falls_back_to_rule(caplog):
    fake = torch.randn(1, 3, 64, 224, 224)
    clip_hyps = [[{"sign": "hello", "confidence": 0.9}]]
    caplog.set_level(logging.WARNING, logger="app.routers.predict")
    with (
        patch("app.services.preprocessing.preprocess_video", return_value=fake),
        patch("app.services.model_service.predict_batch", return_value=clip_hyps),
        patch(
            "app.config.get_settings",
            return_value=_mock_settings(gloss_english_mode="bedrock"),
        ),
        patch(
            "app.services.gloss_to_english_bedrock.gloss_sequence_to_english_bedrock",
            side_effect=RuntimeError("throttle"),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict/sentence",
                files=[("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))],
            )
    assert r.status_code == 200
    assert r.json()["english"].lower().startswith("hello")
    assert any(
        "Bedrock gloss rewrite failed" in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_sentence_mode_t5_error_logs_and_falls_back_to_rule(caplog):
    fake = torch.randn(1, 3, 64, 224, 224)
    clip_hyps = [[{"sign": "hello", "confidence": 0.9}]]
    caplog.set_level(logging.WARNING, logger="app.routers.predict")
    with (
        patch("app.services.preprocessing.preprocess_video", return_value=fake),
        patch("app.services.model_service.predict_batch", return_value=clip_hyps),
        patch(
            "app.config.get_settings",
            return_value=_mock_settings(gloss_english_mode="t5"),
        ),
        patch(
            "app.services.gloss_to_english_t5.gloss_sequence_to_english_t5",
            side_effect=RuntimeError("t5 oom"),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict/sentence",
                files=[("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))],
            )
    assert r.status_code == 200
    assert r.json()["english"].lower().startswith("hello")
    assert any("T5 gloss rewrite failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_sentence_unknown_mode_uses_rule():
    fake = torch.randn(1, 3, 64, 224, 224)
    clip_hyps = [[{"sign": "hello", "confidence": 0.9}]]
    with (
        patch("app.services.preprocessing.preprocess_video", return_value=fake),
        patch("app.services.model_service.predict_batch", return_value=clip_hyps),
        patch(
            "app.config.get_settings",
            return_value=_mock_settings(gloss_english_mode="not-a-real-mode"),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict/sentence",
                files=[("files", ("a.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4"))],
            )
    assert r.status_code == 200
    assert r.json()["english"].lower().startswith("hello")
