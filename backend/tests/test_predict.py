"""Tests for the /api/v1/predict endpoint with a mocked model."""

import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture(autouse=True)
def mock_model():
    """Inject a fake model and label map into app state."""
    app.state.model = MagicMock()
    app.state.index_to_gloss = {0: "hello", 1: "thanks", 2: "water"}
    yield
    app.state.model = None
    app.state.index_to_gloss = None


@pytest.mark.asyncio
async def test_predict_rejects_empty_file():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            files={"file": ("clip.mp4", b"", "video/mp4")},
        )
    assert response.status_code == 400
    assert "Empty" in response.json()["detail"]


@pytest.mark.asyncio
async def test_predict_rejects_non_video():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            files={"file": ("photo.jpg", b"\xff\xd8\xff", "image/jpeg")},
        )
    assert response.status_code == 400
    assert "video" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_predict_returns_result():
    import torch

    fake_tensor = torch.randn(1, 3, 64, 224, 224)
    fake_results = [
        {"sign": "hello", "confidence": 0.95},
        {"sign": "thanks", "confidence": 0.03},
    ]

    with (
        patch("app.services.preprocessing.preprocess_video", return_value=fake_tensor),
        patch("app.services.model_service.predict", return_value=fake_results),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/predict",
                files={"file": ("clip.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4")},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["sign"] == "hello"
    assert data["confidence"] == 0.95
    assert len(data["top_k"]) == 2


@pytest.mark.asyncio
async def test_predict_no_model_returns_503():
    app.state.model = None
    app.state.index_to_gloss = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/predict",
            files={"file": ("clip.mp4", b"\x00\x00\x00\x1cftyp", "video/mp4")},
        )
    assert response.status_code == 503
