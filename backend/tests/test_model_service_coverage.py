"""Full coverage for app.services.model_service (S3 download, load_model branches)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.services import model_service as ms


def test_download_model_from_s3(tmp_path):
    settings = SimpleNamespace(
        aws_s3_bucket="b",
        aws_s3_region="r",
        aws_s3_model_key="k",
    )
    dest = tmp_path / "out.pt"
    mock_s3 = MagicMock()

    def _write_file(bucket, key, filename):
        Path(filename).write_bytes(b"x" * 10)

    mock_s3.download_file.side_effect = _write_file
    with patch("boto3.client", return_value=mock_s3):
        ms._download_model_from_s3(settings, dest)
    mock_s3.download_file.assert_called_once_with("b", "k", str(dest))
    assert dest.exists() and dest.stat().st_size == 10


def test_load_model_with_skipped_keys(tmp_path, monkeypatch):
    label = tmp_path / "lm.json"
    label.write_text(json.dumps({"gloss_to_index": {"a": 0, "b": 1}}))

    from ml.i3d_msft.pytorch_i3d import InceptionI3d

    real = InceptionI3d(400, in_channels=3)
    real.replace_logits(2)
    good_state = real.state_dict()
    bad_state = {**good_state, "extra_bad_key": torch.zeros(1)}
    ckpt = tmp_path / "m.pt"
    torch.save(bad_state, ckpt)

    settings = SimpleNamespace(
        model_path=str(ckpt),
        label_map_path=str(label),
        model_device="cpu",
        aws_s3_bucket="b",
        aws_s3_region="r",
        aws_s3_model_key="k",
    )
    model, idx = ms.load_model(settings)
    assert len(idx) == 2
    assert model is not None


def test_load_model_triggers_s3_download(tmp_path):
    label = tmp_path / "lm.json"
    label.write_text(json.dumps({"gloss_to_index": {"only": 0}}))
    ckpt = tmp_path / "m.pt"

    from ml.i3d_msft.pytorch_i3d import InceptionI3d

    m = InceptionI3d(400, in_channels=3)
    m.replace_logits(1)
    torch.save(m.state_dict(), ckpt)

    missing = tmp_path / "missing.pt"

    def fake_download(settings, dest: Path):
        dest.write_bytes(ckpt.read_bytes())

    settings = SimpleNamespace(
        model_path=str(missing),
        label_map_path=str(label),
        model_device="cpu",
        aws_s3_bucket="b",
        aws_s3_region="r",
        aws_s3_model_key="k",
    )
    with patch.object(ms, "_download_model_from_s3", side_effect=fake_download):
        model, idx = ms.load_model(settings)
    assert missing.exists()
    assert idx == {0: "only"}


def test_predict_unknown_class_index():
    import torch

    logits = torch.full((1, 20, 2), -10.0)
    logits[0, 15, :] = 10.0
    mock_m = MagicMock(return_value=logits)
    idx_map = {0: "a", 1: "b"}
    tensor = torch.randn(1, 3, 2, 2, 2)
    out = ms.predict(mock_m, idx_map, tensor, top_k=3, device="cpu")
    assert any(o["sign"] == "class_15" for o in out)


def test_predict_2d_logits_branch():
    import torch

    mock_m = MagicMock(return_value=torch.randn(1, 2))
    idx_map = {0: "a", 1: "b"}
    tensor = torch.randn(1, 3, 1, 1, 1)
    out = ms.predict(mock_m, idx_map, tensor, top_k=2, device="cpu")
    assert len(out) == 2


def test_load_model_missing_label_map(tmp_path):
    missing = tmp_path / "nope.json"
    settings = SimpleNamespace(
        model_path=str(tmp_path / "m.pt"),
        label_map_path=str(missing),
        model_device="cpu",
        aws_s3_bucket="b",
        aws_s3_region="r",
        aws_s3_model_key="k",
    )
    with pytest.raises(FileNotFoundError, match="Label map not found"):
        ms.load_model(settings)


def test_predict_unsqueeze_dim4():
    import torch

    mock_m = MagicMock(return_value=torch.randn(1, 2, 3))
    idx_map = {0: "a", 1: "b"}
    four_d = torch.randn(3, 64, 224, 224)
    out = ms.predict(mock_m, idx_map, four_d, top_k=2, device="cpu")
    assert len(out) == 2


def test_repo_root_path_inserted(tmp_path, monkeypatch):
    """Cover sys.path insert when repo root was not on path."""
    import importlib
    import sys

    root = str(Path(ms.__file__).resolve().parent.parent.parent.parent)
    saved = list(sys.path)
    try:
        while root in sys.path:
            sys.path.remove(root)
        importlib.reload(ms)
        assert root in sys.path
    finally:
        sys.path[:] = saved
        importlib.reload(ms)
