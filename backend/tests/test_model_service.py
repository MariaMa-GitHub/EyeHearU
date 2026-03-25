"""Tests for model_service — label map loading and predict output format."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.model_service import _load_label_map, predict


def test_load_label_map_teammate_format(tmp_path):
    """Handles the nested format with gloss_to_index key."""
    data = {
        "source_csv": "/some/path/train.csv",
        "num_classes": 3,
        "gloss_to_index": {"hello": 0, "thanks": 1, "water": 2},
        "index_to_gloss": ["hello", "thanks", "water"],
    }
    p = tmp_path / "label_map.json"
    p.write_text(json.dumps(data))

    result = _load_label_map(p)
    assert result == {0: "hello", 1: "thanks", 2: "water"}


def test_load_label_map_simple_format(tmp_path):
    """Handles the simple {gloss: index} format."""
    data = {"hello": 0, "thanks": 1, "water": 2}
    p = tmp_path / "label_map.json"
    p.write_text(json.dumps(data))

    result = _load_label_map(p)
    assert result == {0: "hello", 1: "thanks", 2: "water"}


def test_load_label_map_index_list_format(tmp_path):
    """Handles format with only index_to_gloss as a list."""
    data = {"index_to_gloss": ["hello", "thanks", "water"]}
    p = tmp_path / "label_map.json"
    p.write_text(json.dumps(data))

    result = _load_label_map(p)
    assert result == {0: "hello", 1: "thanks", 2: "water"}


def test_predict_output_format():
    """predict() returns list of dicts with sign and confidence."""
    import torch

    index_to_gloss = {0: "hello", 1: "thanks", 2: "water"}

    mock_model = MagicMock()
    # Simulate I3D output: (1, 3, T') — 3 classes, temporal dim 4
    fake_logits = torch.randn(1, 3, 4)
    mock_model.return_value = fake_logits

    tensor = torch.randn(1, 3, 64, 224, 224)
    results = predict(mock_model, index_to_gloss, tensor, top_k=3, device="cpu")

    assert len(results) == 3
    for r in results:
        assert "sign" in r
        assert "confidence" in r
        assert isinstance(r["confidence"], float)
        assert 0 <= r["confidence"] <= 1.0
        assert r["sign"] in index_to_gloss.values()


def test_predict_handles_2d_logits():
    """predict() handles models that return (B, C) without temporal dim."""
    import torch

    index_to_gloss = {0: "hello", 1: "thanks"}

    mock_model = MagicMock()
    fake_logits = torch.randn(1, 2)
    mock_model.return_value = fake_logits

    tensor = torch.randn(1, 3, 64, 224, 224)
    results = predict(mock_model, index_to_gloss, tensor, top_k=2, device="cpu")

    assert len(results) == 2
    total_conf = sum(r["confidence"] for r in results)
    assert abs(total_conf - 1.0) < 0.01
