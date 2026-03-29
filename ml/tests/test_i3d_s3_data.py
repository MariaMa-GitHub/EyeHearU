"""Tests for i3d_msft/s3_data.py — S3 sync helpers for I3D training."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from i3d_msft.s3_data import (
    get_s3_client,
    get_active_plan_id,
    download_splits,
    _read_split_rows,
    collect_required_filenames,
    download_clip_subset,
)


# ── get_s3_client ──────────────────────────────────────────────────────

def test_get_s3_client():
    with patch("i3d_msft.s3_data.boto3") as mock_boto3:
        mock_boto3.client.return_value = MagicMock()
        client = get_s3_client("us-east-1")
        mock_boto3.client.assert_called_once_with("s3", region_name="us-east-1")
        assert client is not None


# ── get_active_plan_id ─────────────────────────────────────────────────

def test_get_active_plan_id_success():
    s3 = MagicMock()
    body = json.dumps({"active_plan_id": "plan-v1"}).encode()
    s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=body))}
    result = get_active_plan_id(s3, "my-bucket", mvp=True)
    assert result == "plan-v1"
    s3.get_object.assert_called_once_with(
        Bucket="my-bucket",
        Key="processed/mvp/i3d/split_plans/ACTIVE_PLAN.json",
    )


def test_get_active_plan_id_no_mvp():
    s3 = MagicMock()
    body = json.dumps({"active_plan_id": "plan-v2"}).encode()
    s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=body))}
    result = get_active_plan_id(s3, "bucket", mvp=False)
    assert result == "plan-v2"
    s3.get_object.assert_called_once_with(
        Bucket="bucket",
        Key="processed/i3d/split_plans/ACTIVE_PLAN.json",
    )


def test_get_active_plan_id_missing_key():
    s3 = MagicMock()
    body = json.dumps({"other_key": "value"}).encode()
    s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=body))}
    with pytest.raises(RuntimeError, match="missing active_plan_id"):
        get_active_plan_id(s3, "bucket")


# ── download_splits ────────────────────────────────────────────────────

def test_download_splits(tmp_path):
    s3 = MagicMock()
    local_dir = tmp_path / "splits"
    result = download_splits(s3, "bucket", "plan-1", local_dir, mvp=True)
    assert s3.download_file.call_count == 3
    assert set(result.keys()) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert result[split] == local_dir / f"{split}.csv"


def test_download_splits_no_mvp(tmp_path):
    s3 = MagicMock()
    local_dir = tmp_path / "splits"
    download_splits(s3, "bucket", "p1", local_dir, mvp=False)
    expected_base = "processed/i3d/split_plans/p1/splits"
    calls = s3.download_file.call_args_list
    assert calls[0] == call("bucket", f"{expected_base}/train.csv", str(local_dir / "train.csv"))


# ── _read_split_rows ──────────────────────────────────────────────────

def test_read_split_rows(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("user,filename,gloss\nalice,clip1.mp4,hello\nbob,clip2.mp4,world\n")
    rows = _read_split_rows(csv_path)
    assert len(rows) == 2
    assert rows[0]["gloss"] == "hello"
    assert rows[1]["user"] == "bob"


# ── collect_required_filenames ─────────────────────────────────────────

def test_collect_required_filenames(tmp_path):
    csv1 = tmp_path / "train.csv"
    csv1.write_text("user,filename,gloss\na,clip1.mp4,hi\nb,clip2.mp4,bye\n")
    csv2 = tmp_path / "val.csv"
    csv2.write_text("user,filename,gloss\nc,clip3.mp4,hi\n")
    result = collect_required_filenames([csv1, csv2])
    assert result == ["clip1.mp4", "clip2.mp4", "clip3.mp4"]


def test_collect_required_filenames_with_limit(tmp_path):
    csv1 = tmp_path / "train.csv"
    csv1.write_text("user,filename,gloss\na,clip1.mp4,hi\nb,clip2.mp4,bye\nc,clip3.mp4,yo\n")
    result = collect_required_filenames([csv1], limit=2)
    assert len(result) == 2


def test_collect_required_filenames_deduplicates(tmp_path):
    csv1 = tmp_path / "a.csv"
    csv1.write_text("user,filename,gloss\na,clip1.mp4,hi\nb,clip1.mp4,hi\n")
    result = collect_required_filenames([csv1])
    assert result == ["clip1.mp4"]


def test_collect_required_filenames_skips_empty(tmp_path):
    csv1 = tmp_path / "a.csv"
    csv1.write_text("user,filename,gloss\na,,hi\nb,clip1.mp4,bye\n")
    result = collect_required_filenames([csv1])
    assert result == ["clip1.mp4"]


# ── download_clip_subset ───────────────────────────────────────────────

def test_download_clip_subset_success(tmp_path):
    s3 = MagicMock()
    clips_root = tmp_path / "clips"
    downloaded, skipped = download_clip_subset(
        s3, "bucket", ["train/hi/a.mp4", "train/bye/b.mp4"], clips_root, mvp=True
    )
    assert downloaded == 2
    assert skipped == 0
    assert s3.download_file.call_count == 2


def test_download_clip_subset_skip_existing(tmp_path):
    s3 = MagicMock()
    clips_root = tmp_path / "clips"
    existing = clips_root / "a.mp4"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("data")
    downloaded, skipped = download_clip_subset(
        s3, "bucket", ["a.mp4", "b.mp4"], clips_root, mvp=True
    )
    assert downloaded == 1
    assert skipped == 1


def test_download_clip_subset_no_mvp(tmp_path):
    s3 = MagicMock()
    clips_root = tmp_path / "clips"
    download_clip_subset(s3, "bucket", ["c.mp4"], clips_root, mvp=False)
    key_arg = s3.download_file.call_args[0][1]
    assert key_arg == "processed/clips/c.mp4"


def test_download_clip_subset_missing_s3_key(tmp_path):
    from botocore.exceptions import ClientError

    s3 = MagicMock()
    error_response = {"Error": {"Code": "NoSuchKey"}}
    s3.download_file.side_effect = ClientError(error_response, "GetObject")
    clips_root = tmp_path / "clips"
    downloaded, skipped = download_clip_subset(
        s3, "bucket", ["missing.mp4"], clips_root, mvp=True
    )
    assert downloaded == 0
    assert skipped == 0


def test_download_clip_subset_404_code(tmp_path):
    from botocore.exceptions import ClientError

    s3 = MagicMock()
    error_response = {"Error": {"Code": "404"}}
    s3.download_file.side_effect = ClientError(error_response, "GetObject")
    clips_root = tmp_path / "clips"
    downloaded, skipped = download_clip_subset(
        s3, "bucket", ["missing.mp4"], clips_root, mvp=True
    )
    assert downloaded == 0


def test_download_clip_subset_other_error_raises(tmp_path):
    from botocore.exceptions import ClientError

    s3 = MagicMock()
    error_response = {"Error": {"Code": "AccessDenied"}}
    s3.download_file.side_effect = ClientError(error_response, "GetObject")
    clips_root = tmp_path / "clips"
    with pytest.raises(ClientError):
        download_clip_subset(s3, "bucket", ["clip.mp4"], clips_root, mvp=True)
