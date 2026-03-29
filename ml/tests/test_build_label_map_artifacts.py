"""Tests for i3d_msft/build_label_map_artifacts.py — label map rebuilder."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from i3d_msft.build_label_map_artifacts import main, _write_json


# ── _write_json ────────────────────────────────────────────────────────

def test_write_json_basic(tmp_path):
    out = tmp_path / "test.json"
    data = {"a": 1, "b": [2, 3]}
    _write_json(out, data)
    assert json.loads(out.read_text()) == data


def test_write_json_creates_parent_dirs(tmp_path):
    out = tmp_path / "sub" / "deep" / "test.json"
    _write_json(out, {"key": "val"})
    assert out.exists()


# ── main ───────────────────────────────────────────────────────────────

def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user", "filename", "gloss"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_test_env(tmp_path):
    """Create split CSVs and dummy video clips for testing."""
    import cv2

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    clips_root = tmp_path / "clips"
    clips_root.mkdir()

    train_rows = [
        {"user": "u1", "filename": "a.mp4", "gloss": "hello"},
        {"user": "u2", "filename": "b.mp4", "gloss": "bye"},
    ]
    val_rows = [
        {"user": "u3", "filename": "c.mp4", "gloss": "hello"},
    ]

    _write_csv(splits_dir / "train.csv", train_rows)
    _write_csv(splits_dir / "val.csv", val_rows)
    _write_csv(splits_dir / "test.csv", val_rows)

    for name in ["a.mp4", "b.mp4", "c.mp4"]:
        p = clips_root / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(p), fourcc, 30.0, (256, 256))
        for _ in range(5):
            writer.write(np.zeros((256, 256, 3), dtype=np.uint8))
        writer.release()

    return splits_dir, clips_root


def _make_mocks(splits_dir, clips_root):
    mock_s3 = MagicMock()

    def fake_download_splits(s3, bucket, plan_id, local_dir, mvp=True):
        import shutil
        local_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for s in ("train", "val", "test"):
            src = splits_dir / f"{s}.csv"
            dst = local_dir / f"{s}.csv"
            shutil.copy(src, dst)
            out[s] = dst
        return out

    def fake_download_clips(s3, bucket, filenames, root, mvp=True):
        import shutil
        root.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            src = clips_root / name
            dst = root / name
            if src.exists():
                shutil.copy(src, dst)
        return len(filenames), 0

    return mock_s3, fake_download_splits, fake_download_clips


def test_main_basic(tmp_path):
    """main() produces filtered_train.csv and label_map.json."""
    splits_dir, clips_root = _make_test_env(tmp_path)
    workdir = tmp_path / "workdir"
    mock_s3, fake_dl_splits, fake_dl_clips = _make_mocks(splits_dir, clips_root)

    with patch("i3d_msft.build_label_map_artifacts.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.build_label_map_artifacts.download_splits", side_effect=fake_dl_splits), \
         patch("i3d_msft.build_label_map_artifacts.download_clip_subset", side_effect=fake_dl_clips), \
         patch("sys.argv", ["build", "--bucket", "b", "--plan-id", "plan1",
                            "--no-verify-readable", "--local-root", str(workdir)]):
        main()

    lm_path = workdir / "label_map.json"
    assert lm_path.exists()
    lm = json.loads(lm_path.read_text())
    assert "num_classes" in lm
    assert "gloss_to_index" in lm
    assert lm["num_classes"] == len(lm["gloss_to_index"])


def test_main_with_clip_limit(tmp_path):
    splits_dir, clips_root = _make_test_env(tmp_path)
    workdir = tmp_path / "workdir"
    mock_s3, fake_dl_splits, fake_dl_clips = _make_mocks(splits_dir, clips_root)

    with patch("i3d_msft.build_label_map_artifacts.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.build_label_map_artifacts.download_splits", side_effect=fake_dl_splits), \
         patch("i3d_msft.build_label_map_artifacts.download_clip_subset", side_effect=fake_dl_clips), \
         patch("sys.argv", ["build", "--bucket", "b", "--plan-id", "plan1",
                            "--clip-limit", "10", "--no-verify-readable",
                            "--local-root", str(workdir)]):
        main()

    assert (workdir / "label_map.json").exists()


def test_main_with_s3_upload(tmp_path):
    splits_dir, clips_root = _make_test_env(tmp_path)
    workdir = tmp_path / "workdir"
    mock_s3, fake_dl_splits, fake_dl_clips = _make_mocks(splits_dir, clips_root)

    with patch("i3d_msft.build_label_map_artifacts.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.build_label_map_artifacts.download_splits", side_effect=fake_dl_splits), \
         patch("i3d_msft.build_label_map_artifacts.download_clip_subset", side_effect=fake_dl_clips), \
         patch("sys.argv", ["build", "--bucket", "b", "--plan-id", "plan1",
                            "--no-verify-readable",
                            "--upload-to-s3-base", "artifacts/plan1",
                            "--local-root", str(workdir)]):
        main()

    assert mock_s3.upload_file.call_count >= 2


def test_main_cleans_existing_workdir(tmp_path):
    """If local_root already exists, it's wiped for reproducibility."""
    splits_dir, clips_root = _make_test_env(tmp_path)
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "stale_file.txt").write_text("old data")
    mock_s3, fake_dl_splits, fake_dl_clips = _make_mocks(splits_dir, clips_root)

    with patch("i3d_msft.build_label_map_artifacts.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.build_label_map_artifacts.download_splits", side_effect=fake_dl_splits), \
         patch("i3d_msft.build_label_map_artifacts.download_clip_subset", side_effect=fake_dl_clips), \
         patch("sys.argv", ["build", "--bucket", "b", "--plan-id", "plan1",
                            "--no-verify-readable", "--local-root", str(workdir)]):
        main()

    assert not (workdir / "stale_file.txt").exists()
    assert (workdir / "label_map.json").exists()


def test_main_empty_filtered_raises(tmp_path):
    """If all clips are filtered out, main raises RuntimeError."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    _write_csv(splits_dir / "train.csv", [
        {"user": "u1", "filename": "missing.mp4", "gloss": "hello"},
    ])
    _write_csv(splits_dir / "val.csv", [
        {"user": "u2", "filename": "missing2.mp4", "gloss": "hello"},
    ])
    _write_csv(splits_dir / "test.csv", [])

    mock_s3 = MagicMock()

    def fake_download_splits(s3, bucket, plan_id, local_dir, mvp=True):
        import shutil
        local_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for s in ("train", "val", "test"):
            src = splits_dir / f"{s}.csv"
            dst = local_dir / f"{s}.csv"
            shutil.copy(src, dst)
            out[s] = dst
        return out

    def fake_download_clips(s3, bucket, filenames, root, mvp=True):
        root.mkdir(parents=True, exist_ok=True)
        return 0, 0  # nothing downloaded

    workdir = tmp_path / "workdir"
    with patch("i3d_msft.build_label_map_artifacts.get_s3_client", return_value=mock_s3), \
         patch("i3d_msft.build_label_map_artifacts.download_splits", side_effect=fake_download_splits), \
         patch("i3d_msft.build_label_map_artifacts.download_clip_subset", side_effect=fake_download_clips), \
         patch("sys.argv", ["build", "--bucket", "b", "--plan-id", "plan1",
                            "--no-verify-readable", "--local-root", str(workdir)]):
        with pytest.raises(RuntimeError, match="empty"):
            main()
