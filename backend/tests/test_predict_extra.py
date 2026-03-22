"""Extra predict router coverage: errors and empty results."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch

from app.main import app


@pytest.fixture(autouse=True)
def restore_model_state():
    yield
    app.state.model = None
    app.state.index_to_gloss = None


@pytest.mark.asyncio
async def test_predict_value_error_from_preprocess():
    app.state.model = MagicMock()
    app.state.index_to_gloss = {0: "a"}

    with patch(
        "app.services.preprocessing.preprocess_video",
        side_effect=ValueError("bad video"),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict",
                files={"file": ("clip.mp4", b"x", "video/mp4")},
            )
    assert r.status_code == 400
    assert "bad video" in r.json()["detail"]


@pytest.mark.asyncio
async def test_predict_inference_error():
    app.state.model = MagicMock()
    app.state.index_to_gloss = {0: "a"}
    import torch

    with (
        patch(
            "app.services.preprocessing.preprocess_video",
            return_value=torch.zeros(1, 3, 64, 224, 224),
        ),
        patch(
            "app.services.model_service.predict",
            side_effect=RuntimeError("gpu melted"),
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict",
                files={"file": ("clip.mp4", b"x", "video/mp4")},
            )
    assert r.status_code == 500
    assert "Inference error" in r.json()["detail"]


@pytest.mark.asyncio
async def test_predict_empty_results_list():
    app.state.model = MagicMock()
    app.state.index_to_gloss = {0: "a"}
    import torch

    with (
        patch(
            "app.services.preprocessing.preprocess_video",
            return_value=torch.zeros(1, 3, 64, 224, 224),
        ),
        patch("app.services.model_service.predict", return_value=[]),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/v1/predict",
                files={"file": ("clip.mp4", b"x", "video/mp4")},
            )
    assert r.status_code == 200
    data = r.json()
    assert data["sign"] == ""
    assert data["confidence"] == 0.0
    assert data["top_k"] == []
