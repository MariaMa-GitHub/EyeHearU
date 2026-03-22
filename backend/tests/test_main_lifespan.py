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
            r = client.get("/health")
            assert r.status_code == 200


def test_lifespan_file_not_found():
    with patch("app.services.model_service.load_model", side_effect=FileNotFoundError("missing")):
        with TestClient(app) as client:
            assert client.app.state.model is None
            assert client.app.state.index_to_gloss is None


def test_lifespan_generic_exception():
    with patch("app.services.model_service.load_model", side_effect=RuntimeError("boom")):
        with TestClient(app) as client:
            assert client.app.state.model is None
            assert client.app.state.index_to_gloss is None
