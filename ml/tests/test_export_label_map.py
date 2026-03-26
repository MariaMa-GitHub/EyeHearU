"""Tests for i3d_msft/export_label_map.py — CSV-to-label-map utility."""

import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from i3d_msft.export_label_map import build_gloss_dict_from_csv, main


class TestBuildGlossDictFromCSV:
    """Tests for build_gloss_dict_from_csv — pure CSV parsing logic."""

    def test_basic_csv(self, tmp_path):
        """Normal CSV with distinct glosses produces sorted index mapping."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,gloss\na,f1,hello\nb,f2,thanks\nc,f3,apple\n")
        result = build_gloss_dict_from_csv(csv_path)
        assert result == {"apple": 0, "hello": 1, "thanks": 2}

    def test_duplicate_glosses_deduplicated(self, tmp_path):
        """Duplicate glosses in CSV are collapsed to a single entry."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,gloss\na,f1,hello\nb,f2,hello\nc,f3,bye\n")
        result = build_gloss_dict_from_csv(csv_path)
        assert result == {"bye": 0, "hello": 1}

    def test_case_normalisation(self, tmp_path):
        """Glosses are lowercased so 'Hello' and 'HELLO' merge."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,gloss\na,f1,Hello\nb,f2,HELLO\n")
        result = build_gloss_dict_from_csv(csv_path)
        assert result == {"hello": 0}

    def test_whitespace_stripped(self, tmp_path):
        """Leading/trailing whitespace on gloss values is stripped."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,gloss\na,f1, hello \nb,f2,bye\n")
        result = build_gloss_dict_from_csv(csv_path)
        assert "hello" in result
        assert " hello " not in result

    def test_empty_gloss_skipped(self, tmp_path):
        """Rows with empty or whitespace-only gloss are excluded."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,gloss\na,f1,\nb,f2, \nc,f3,ok\n")
        result = build_gloss_dict_from_csv(csv_path)
        assert result == {"ok": 0}

    def test_missing_gloss_column(self, tmp_path):
        """CSV without a 'gloss' column returns empty dict (no crash)."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,label\na,f1,hello\n")
        result = build_gloss_dict_from_csv(csv_path)
        assert result == {}

    def test_empty_csv(self, tmp_path):
        """Header-only CSV returns empty dict."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,gloss\n")
        result = build_gloss_dict_from_csv(csv_path)
        assert result == {}

    def test_indices_are_sequential(self, tmp_path):
        """All indices are contiguous starting from 0."""
        csv_path = tmp_path / "train.csv"
        rows = "user,filename,gloss\n" + "\n".join(f"u,f,g{i}" for i in range(10))
        csv_path.write_text(rows)
        result = build_gloss_dict_from_csv(csv_path)
        assert sorted(result.values()) == list(range(10))


class TestExportLabelMapMain:
    """Tests for the main() CLI entry point."""

    def test_basic_export(self, tmp_path):
        """main() writes JSON with expected keys."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,gloss\na,f1,hello\nb,f2,bye\n")
        out_path = tmp_path / "label_map.json"

        with patch("sys.argv", ["prog", "--csv", str(csv_path), "--output", str(out_path)]):
            main()

        data = json.loads(out_path.read_text())
        assert data["num_classes"] == 2
        assert "gloss_to_index" in data
        assert "source_csv" in data
        assert "index_to_gloss" not in data  # not requested

    def test_inverse_flag(self, tmp_path):
        """--inverse adds index_to_gloss list."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,gloss\na,f1,hello\nb,f2,bye\n")
        out_path = tmp_path / "label_map.json"

        with patch("sys.argv", ["prog", "--csv", str(csv_path), "--output", str(out_path), "--inverse"]):
            main()

        data = json.loads(out_path.read_text())
        assert "index_to_gloss" in data
        assert data["index_to_gloss"] == ["bye", "hello"]

    def test_creates_parent_directory(self, tmp_path):
        """Output directory is created if it doesn't exist."""
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("user,filename,gloss\na,f1,ok\n")
        out_path = tmp_path / "nested" / "dir" / "label_map.json"

        with patch("sys.argv", ["prog", "--csv", str(csv_path), "--output", str(out_path)]):
            main()

        assert out_path.exists()
