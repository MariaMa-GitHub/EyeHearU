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


def test_load_model_non_dict_checkpoint_with_state_dict(tmp_path):
    """Cover lines 82-84: checkpoint is not a dict but has .state_dict()."""
    label = tmp_path / "lm.json"
    label.write_text(json.dumps({"gloss_to_index": {"a": 0, "b": 1}}))

    from ml.i3d_msft.pytorch_i3d import InceptionI3d

    real = InceptionI3d(400, in_channels=3)
    real.replace_logits(2)
    good_state = real.state_dict()

    wrapper = MagicMock()
    wrapper.state_dict.return_value = good_state

    ckpt = tmp_path / "m.pt"
    torch.save(good_state, ckpt)

    settings = SimpleNamespace(
        model_path=str(ckpt),
        label_map_path=str(label),
        model_device="cpu",
        aws_s3_bucket="b",
        aws_s3_region="r",
        aws_s3_model_key="k",
    )
    with patch("torch.load", return_value=wrapper):
        model, idx = ms.load_model(settings)
    assert len(idx) == 2
    wrapper.state_dict.assert_called_once()


def test_load_model_non_dict_checkpoint_no_state_dict(tmp_path):
    """Cover lines 85-88: checkpoint is not a dict and has no .state_dict()."""
    label = tmp_path / "lm.json"
    label.write_text(json.dumps({"gloss_to_index": {"a": 0}}))

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    settings = SimpleNamespace(
        model_path=str(ckpt),
        label_map_path=str(label),
        model_device="cpu",
        aws_s3_bucket="b",
        aws_s3_region="r",
        aws_s3_model_key="k",
    )
    with patch("torch.load", return_value="not-a-dict"):
        with pytest.raises(ValueError, match="Unexpected checkpoint type"):
            ms.load_model(settings)


def test_load_model_state_dict_returns_non_dict(tmp_path):
    """Cover lines 89-92: .state_dict() returns something that is not a dict."""
    label = tmp_path / "lm.json"
    label.write_text(json.dumps({"gloss_to_index": {"a": 0}}))

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    wrapper = MagicMock()
    wrapper.state_dict.return_value = [1, 2, 3]

    settings = SimpleNamespace(
        model_path=str(ckpt),
        label_map_path=str(label),
        model_device="cpu",
        aws_s3_bucket="b",
        aws_s3_region="r",
        aws_s3_model_key="k",
    )
    with patch("torch.load", return_value=wrapper):
        with pytest.raises(ValueError, match="not a dict"):
            ms.load_model(settings)


def test_load_model_reports_missing_keys(tmp_path):
    """Cover line 111: load_state_dict reports missing_keys."""
    label = tmp_path / "lm.json"
    label.write_text(json.dumps({"gloss_to_index": {"a": 0, "b": 1}}))

    from ml.i3d_msft.pytorch_i3d import InceptionI3d

    real = InceptionI3d(400, in_channels=3)
    real.replace_logits(2)
    full_state = real.state_dict()
    # Remove a key so load_state_dict will report it missing
    partial_state = {k: v for i, (k, v) in enumerate(full_state.items()) if i > 0}

    ckpt = tmp_path / "m.pt"
    torch.save(partial_state, ckpt)

    settings = SimpleNamespace(
        model_path=str(ckpt),
        label_map_path=str(label),
        model_device="cpu",
        aws_s3_bucket="b",
        aws_s3_region="r",
        aws_s3_model_key="k",
    )
    model, idx = ms.load_model(settings)
    assert model is not None


def test_load_model_reports_unexpected_keys(tmp_path):
    """Cover line 116: load_state_dict reports unexpected_keys."""
    label = tmp_path / "lm.json"
    label.write_text(json.dumps({"gloss_to_index": {"a": 0, "b": 1}}))

    from ml.i3d_msft.pytorch_i3d import InceptionI3d

    real = InceptionI3d(400, in_channels=3)
    real.replace_logits(2)
    good_state = real.state_dict()

    ckpt = tmp_path / "m.pt"
    torch.save(good_state, ckpt)

    settings = SimpleNamespace(
        model_path=str(ckpt),
        label_map_path=str(label),
        model_device="cpu",
        aws_s3_bucket="b",
        aws_s3_region="r",
        aws_s3_model_key="k",
    )

    fake_result = SimpleNamespace(missing_keys=[], unexpected_keys=["bogus.weight"])
    with patch.object(InceptionI3d, "load_state_dict", return_value=fake_result):
        model, idx = ms.load_model(settings)
    assert model is not None


def test_predict_batch_two_clips():
    from ml.i3d_msft.pytorch_i3d import InceptionI3d

    m = InceptionI3d(400, 3)
    m.replace_logits(3)
    m.eval()
    t1 = torch.randn(1, 3, 64, 224, 224)
    t2 = torch.randn(1, 3, 64, 224, 224)
    idx = {0: "a", 1: "b", 2: "c"}
    out = ms.predict_batch(m, idx, [t1, t2], top_k=2, device="cpu")
    assert len(out) == 2
    assert all(len(row) == 2 for row in out)


def test_predict_batch_four_dim_tensor():
    from ml.i3d_msft.pytorch_i3d import InceptionI3d

    m = InceptionI3d(400, 3)
    m.replace_logits(2)
    m.eval()
    t = torch.randn(3, 64, 224, 224)
    idx = {0: "x", 1: "y"}
    out = ms.predict_batch(m, idx, [t], top_k=2, device="cpu")
    assert len(out) == 1


def test_predict_batch_empty():
    assert ms.predict_batch(MagicMock(), {}, [], top_k=5) == []


def test_predict_batch_non3d_logits():
    """Cover branch when model returns 2D logits (B, C) without temporal dim."""

    class FlatLogits(torch.nn.Module):
        def forward(self, x, pretrained=False):
            b = x.size(0)
            return torch.randn(b, 4)

    m = FlatLogits()
    m.eval()
    idx = {0: "a", 1: "b", 2: "c", 3: "d"}
    out = ms.predict_batch(
        m,
        idx,
        [torch.randn(1, 3, 64, 224, 224), torch.randn(1, 3, 64, 224, 224)],
        top_k=2,
        device="cpu",
    )
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
