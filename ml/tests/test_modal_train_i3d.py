"""Tests for modal_train_i3d.py — extracted helper functions."""

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modal_train_i3d import (
    _build_eval_cmd,
    _build_train_cmd,
    _parse_run_name,
    _resolve_active_plan,
    _upload_checkpoints,
    _upload_run_metadata,
)


# ── _parse_run_name ──────────────────────────────────────────────────

def test_parse_run_name_present():
    with patch.object(sys, "argv", ["prog", "--run-name", "my-run"]):
        assert _parse_run_name() == "my-run"


def test_parse_run_name_absent():
    with patch.object(sys, "argv", ["prog", "--epochs", "10"]):
        assert _parse_run_name() == ""


def test_parse_run_name_at_end():
    """--run-name is the last arg with no value following it."""
    with patch.object(sys, "argv", ["prog", "--run-name"]):
        assert _parse_run_name() == ""


# ── _build_train_cmd ─────────────────────────────────────────────────

def test_build_train_cmd_minimal():
    cmd = _build_train_cmd(
        bucket="b", region="us-east-1", epochs=5, batch_size=4,
        num_workers=1, head_only_epochs=2, head_lr=0.001, backbone_lr=0.0001,
        plan_id=None, clip_limit=None, init_checkpoint_s3_key=None,
        init_strict=False, s3_ckpt_prefix="prefix/run1",
    )
    assert "--bucket" in cmd
    assert "b" in cmd
    assert "--plan-id" not in cmd
    assert "--clip-limit" not in cmd
    assert "--init-checkpoint-s3-key" not in cmd
    assert "--init-strict" not in cmd
    assert cmd[-2:] == ["--s3-checkpoint-prefix", "prefix/run1"]


def test_build_train_cmd_all_options():
    cmd = _build_train_cmd(
        bucket="b", region="r", epochs=10, batch_size=8,
        num_workers=4, head_only_epochs=3, head_lr=0.01, backbone_lr=0.001,
        plan_id="plan-v1", clip_limit=100,
        init_checkpoint_s3_key="models/ckpt.pt", init_strict=True,
        s3_ckpt_prefix="pfx/run2",
    )
    assert "--plan-id" in cmd
    assert "plan-v1" in cmd
    assert "--clip-limit" in cmd
    assert "100" in cmd
    assert "--init-checkpoint-s3-key" in cmd
    assert "models/ckpt.pt" in cmd
    assert "--init-strict" in cmd


def test_build_train_cmd_device_always_cuda():
    cmd = _build_train_cmd(
        bucket="b", region="r", epochs=1, batch_size=1,
        num_workers=0, head_only_epochs=0, head_lr=0.001, backbone_lr=0.0001,
        plan_id=None, clip_limit=None, init_checkpoint_s3_key=None,
        init_strict=False, s3_ckpt_prefix="p",
    )
    idx = cmd.index("--device")
    assert cmd[idx + 1] == "cuda"


# ── _build_eval_cmd ──────────────────────────────────────────────────

def test_build_eval_cmd_local_checkpoint():
    cmd = _build_eval_cmd(
        bucket="b", region="r", plan_id="p1", split="val",
        checkpoint_arg="/path/best.pt", checkpoint_is_local=True,
    )
    assert "--checkpoint-local" in cmd
    assert "/path/best.pt" in cmd
    assert "--checkpoint-s3-key" not in cmd
    assert "--split" in cmd
    idx = cmd.index("--split")
    assert cmd[idx + 1] == "val"


def test_build_eval_cmd_s3_checkpoint():
    cmd = _build_eval_cmd(
        bucket="b", region="r", plan_id="p1", split="test",
        checkpoint_arg="models/ckpt.pt", checkpoint_is_local=False,
    )
    assert "--checkpoint-s3-key" in cmd
    assert "models/ckpt.pt" in cmd
    assert "--checkpoint-local" not in cmd


def test_build_eval_cmd_with_optional_args():
    cmd = _build_eval_cmd(
        bucket="b", region="r", plan_id="p1", split="val",
        checkpoint_arg="ckpt.pt", checkpoint_is_local=True,
        batch_size=12, num_workers=4, clip_limit=50,
        output_json="/out.json", gloss_dict_csv="/gloss.csv",
    )
    assert "--batch-size" in cmd
    assert "12" in cmd
    assert "--num-workers" in cmd
    assert "4" in cmd
    assert "--clip-limit" in cmd
    assert "50" in cmd
    assert "--output-json" in cmd
    assert "/out.json" in cmd
    assert "--gloss-dict-csv" in cmd
    assert "/gloss.csv" in cmd


def test_build_eval_cmd_no_optional_args():
    cmd = _build_eval_cmd(
        bucket="b", region="r", plan_id="p1", split="val",
        checkpoint_arg="ckpt.pt", checkpoint_is_local=False,
    )
    assert "--clip-limit" not in cmd
    assert "--output-json" not in cmd
    assert "--gloss-dict-csv" not in cmd


# ── _resolve_active_plan ─────────────────────────────────────────────

def test_resolve_active_plan():
    body = json.dumps({"active_plan_id": "plan-v4"}).encode()
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": BytesIO(body)}
    result = _resolve_active_plan(mock_s3, "my-bucket")
    assert result == "plan-v4"
    mock_s3.get_object.assert_called_once_with(
        Bucket="my-bucket",
        Key="processed/mvp/i3d/split_plans/ACTIVE_PLAN.json",
    )


# ── _upload_checkpoints ──────────────────────────────────────────────

def test_upload_checkpoints(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "best_model.pt").write_text("data")
    sub = ckpt_dir / "sub"
    sub.mkdir()
    (sub / "epoch_5.pt").write_text("data")

    mock_s3 = MagicMock()
    result = _upload_checkpoints(
        mock_s3, "bucket", ckpt_dir, "models/i3d", "plan-v4", "run1"
    )
    assert len(result) == 2
    assert all(s.startswith("s3://bucket/models/i3d/plan-v4/run1/") for s in result)
    assert mock_s3.upload_file.call_count == 2


def test_upload_checkpoints_empty_dir(tmp_path):
    ckpt_dir = tmp_path / "empty"
    ckpt_dir.mkdir()
    mock_s3 = MagicMock()
    result = _upload_checkpoints(mock_s3, "bucket", ckpt_dir, "pfx", "p", "r")
    assert result == []
    mock_s3.upload_file.assert_not_called()


# ── _upload_run_metadata ─────────────────────────────────────────────

def test_upload_run_metadata():
    mock_s3 = MagicMock()
    meta = {"run_id": "r1", "epochs": 10}
    _upload_run_metadata(mock_s3, "bucket", meta, "models/r1/metadata.json")
    mock_s3.put_object.assert_called_once()
    call_kwargs = mock_s3.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "bucket"
    assert call_kwargs["Key"] == "models/r1/metadata.json"
    assert call_kwargs["ContentType"] == "application/json"
    body = json.loads(call_kwargs["Body"].decode())
    assert body["run_id"] == "r1"
    assert body["epochs"] == 10


# ── Module-level constants ───────────────────────────────────────────

def test_app_name_default():
    from modal_train_i3d import APP_NAME
    # When no --run-name in sys.argv, should be default
    assert "eyehearu-i3d" in APP_NAME
