"""Tests for FastAPI lifespan (model load on startup)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_lifespan_loads_model():
    with patch("app.services.model_service.load_model") as lm:
        lm.return_value = (MagicMock(), {0: "hello", 1: "thanks"})
        with TestClient(app) as client:
            assert getattr(client.app.state, "model", None) is not None
            assert len(client.app.state.index_to_gloss) == 2
            assert getattr(client.app.state, "gloss_lm", None) is not None
            r = client.get("/health")
            assert r.status_code == 200


def test_lifespan_file_not_found():
    with patch("app.services.model_service.load_model", side_effect=FileNotFoundError("missing")):
        with TestClient(app) as client:
            assert client.app.state.model is None
            assert client.app.state.index_to_gloss is None
            assert client.app.state.gloss_lm is None


def test_lifespan_generic_exception():
    with patch("app.services.model_service.load_model", side_effect=RuntimeError("boom")):
        with TestClient(app) as client:
            assert client.app.state.model is None
            assert client.app.state.index_to_gloss is None
            assert client.app.state.gloss_lm is None


def test_lifespan_t5_mode_calls_load_t5():
    with (
        patch("app.services.model_service.load_model") as lm,
        patch("app.main.get_settings") as gs,
        patch("app.services.gloss_to_english_t5._load_t5") as t5load,
    ):
        settings = MagicMock()
        settings.gloss_english_mode = "t5"
        settings.app_name = "Eye Hear U API"
        settings.cors_origins = ["*"]
        settings.gloss_lm_path = "data/gloss_lm.json"
        settings.model_path = "model_cache/best_model.pt"
        settings.label_map_path = "../ml/i3d_label_map_mvp-sft-full-v1.json"
        gs.return_value = settings
        lm.return_value = (MagicMock(), {0: "hello"})
        with TestClient(app):
            pass
        t5load.assert_called_once()


def test_lifespan_t5_load_failure_still_ready():
    with (
        patch("app.services.model_service.load_model") as lm,
        patch("app.main.get_settings") as gs,
        patch(
            "app.services.gloss_to_english_t5._load_t5",
            side_effect=RuntimeError("t5 oom"),
        ),
    ):
        settings = MagicMock()
        settings.gloss_english_mode = "t5"
        settings.app_name = "Eye Hear U API"
        settings.cors_origins = ["*"]
        settings.gloss_lm_path = "data/gloss_lm.json"
        settings.model_path = "model_cache/best_model.pt"
        settings.label_map_path = "../ml/i3d_label_map_mvp-sft-full-v1.json"
        gs.return_value = settings
        lm.return_value = (MagicMock(), {0: "hello"})
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
