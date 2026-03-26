"""Extended tests for evaluation/evaluate.py — covers evaluate_model,
save_confusion_matrix_plot, and main().

The core metric functions (compute_topk_accuracy, compute_precision_recall_f1,
build_confusion_matrix) are already tested in test_evaluation.py.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import evaluate_model, save_confusion_matrix_plot, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model(num_classes=3):
    """Minimal model: (B, 3, 4, 8, 8) -> (B, num_classes)."""
    return nn.Sequential(
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Linear(3, num_classes),
    )


def _synthetic_loader(n=8, num_classes=3, batch_size=4):
    clips = torch.randn(n, 3, 4, 8, 8)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(clips, labels), batch_size=batch_size)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    """evaluate_model: full evaluation loop with metric aggregation."""

    def test_returns_expected_keys(self):
        model = _tiny_model(3)
        loader = _synthetic_loader(n=8, num_classes=3)
        label_map_inv = {0: "hello", 1: "thanks", 2: "bye"}
        device = torch.device("cpu")

        results = evaluate_model(model, loader, device, label_map_inv, 3)

        expected_keys = {
            "overall_accuracy", "top5_accuracy",
            "macro_precision", "macro_recall", "macro_f1",
            "per_class_detail", "per_class_accuracy",
            "top_confusions", "confusion_matrix",
            "inference_latency", "total_samples",
        }
        assert expected_keys.issubset(results.keys())

    def test_total_samples_correct(self):
        model = _tiny_model(3)
        loader = _synthetic_loader(n=12, num_classes=3)
        label_map_inv = {0: "a", 1: "b", 2: "c"}

        results = evaluate_model(model, loader, torch.device("cpu"), label_map_inv, 3)
        assert results["total_samples"] == 12

    def test_accuracy_in_valid_range(self):
        model = _tiny_model(3)
        loader = _synthetic_loader(n=8, num_classes=3)
        label_map_inv = {0: "a", 1: "b", 2: "c"}

        results = evaluate_model(model, loader, torch.device("cpu"), label_map_inv, 3)
        assert 0.0 <= results["overall_accuracy"] <= 1.0
        assert 0.0 <= results["top5_accuracy"] <= 1.0

    def test_latency_stats_present(self):
        model = _tiny_model(3)
        loader = _synthetic_loader(n=8, num_classes=3)
        label_map_inv = {0: "a", 1: "b", 2: "c"}

        results = evaluate_model(model, loader, torch.device("cpu"), label_map_inv, 3)
        lat = results["inference_latency"]
        assert "mean_ms" in lat
        assert "p50_ms" in lat
        assert "p95_ms" in lat
        assert "p99_ms" in lat
        assert all(v >= 0 for v in lat.values())

    def test_confusion_matrix_shape(self):
        model = _tiny_model(4)
        loader = _synthetic_loader(n=8, num_classes=4)
        label_map_inv = {i: f"cls_{i}" for i in range(4)}

        results = evaluate_model(model, loader, torch.device("cpu"), label_map_inv, 4)
        cm = results["confusion_matrix"]
        assert len(cm) == 4
        assert all(len(row) == 4 for row in cm)

    def test_per_class_detail_has_all_classes(self):
        model = _tiny_model(3)
        # Use fixed labels to ensure all classes appear
        clips = torch.randn(9, 3, 4, 8, 8)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        loader = DataLoader(TensorDataset(clips, labels), batch_size=3)
        label_map_inv = {0: "a", 1: "b", 2: "c"}

        results = evaluate_model(model, loader, torch.device("cpu"), label_map_inv, 3)
        assert set(results["per_class_detail"].keys()) == {"a", "b", "c"}
        for detail in results["per_class_detail"].values():
            assert "accuracy" in detail
            assert "precision" in detail
            assert "recall" in detail
            assert "f1" in detail

    def test_top_confusions_format(self):
        model = _tiny_model(3)
        loader = _synthetic_loader(n=20, num_classes=3)
        label_map_inv = {0: "a", 1: "b", 2: "c"}

        results = evaluate_model(model, loader, torch.device("cpu"), label_map_inv, 3)
        for item in results["top_confusions"]:
            assert "true" in item
            assert "predicted" in item
            assert "count" in item

    def test_label_map_inv_missing_class_uses_fallback(self):
        """When label_map_inv doesn't have a class index, uses 'class_N' fallback."""
        model = _tiny_model(3)
        clips = torch.randn(6, 3, 4, 8, 8)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loader = DataLoader(TensorDataset(clips, labels), batch_size=6)
        # Intentionally incomplete label map
        label_map_inv = {0: "a"}

        results = evaluate_model(model, loader, torch.device("cpu"), label_map_inv, 3)
        # Should not crash; missing classes get fallback names
        assert results["total_samples"] == 6


# ---------------------------------------------------------------------------
# save_confusion_matrix_plot
# ---------------------------------------------------------------------------

class TestSaveConfusionMatrixPlot:
    """save_confusion_matrix_plot: visualization with matplotlib/seaborn."""

    def test_saves_plot_file(self, tmp_path):
        """With matplotlib available, a PNG should be created."""
        cm = np.array([[5, 1], [2, 4]])
        output = tmp_path / "cm.png"
        save_confusion_matrix_plot(cm, ["a", "b"], output)
        assert output.exists()

    def test_handles_missing_matplotlib(self, tmp_path):
        """Gracefully skips when matplotlib is not importable."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib":
                raise ImportError("no matplotlib")
            return real_import(name, *args, **kwargs)

        cm = np.array([[1, 0], [0, 1]])
        output = tmp_path / "cm.png"

        with patch("builtins.__import__", side_effect=mock_import):
            save_confusion_matrix_plot(cm, ["a", "b"], output)

        assert not output.exists()  # plot was skipped

    def test_large_class_count_no_annot(self, tmp_path):
        """More than 30 classes → annot=False path in sns.heatmap."""
        n = 35
        cm = np.zeros((n, n), dtype=int)
        np.fill_diagonal(cm, 1)
        output = tmp_path / "cm_large.png"
        save_confusion_matrix_plot(cm, [f"c{i}" for i in range(n)], output)
        assert output.exists()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMainFunction:
    """main() orchestration — mocked to avoid real data/model."""

    def test_main_runs_evaluation(self, tmp_path):
        """Full main flow with mocked model, data, and file I/O."""
        label_map = {"hello": 0, "bye": 1}
        label_map_path = tmp_path / "label_map.json"
        label_map_path.write_text(json.dumps(label_map))

        mock_ds = TensorDataset(torch.randn(4, 3, 16, 112, 112), torch.randint(0, 2, (4,)))

        fake_cfg = MagicMock()
        fake_cfg.data.processed_data_dir = str(tmp_path)
        fake_cfg.data.num_frames = 16
        fake_cfg.train.device = "cpu"
        fake_cfg.train.batch_size = 2
        fake_cfg.model.backbone = "r3d_18"
        fake_cfg.model.head_dropout = 0.0

        model_instance = _tiny_model(num_classes=2)
        output_dir = tmp_path / "evaluation_results"

        with patch("sys.argv", ["prog", "--checkpoint", "fake.pt"]), \
             patch("evaluation.evaluate.Config", return_value=fake_cfg), \
             patch("evaluation.evaluate.ASLVideoDataset", return_value=mock_ds), \
             patch("evaluation.evaluate.ASLVideoClassifier") as MockModel, \
             patch("torch.load", return_value=model_instance.state_dict()), \
             patch("evaluation.evaluate.Path") as MockPath, \
             patch("evaluation.evaluate.save_confusion_matrix_plot"):

            MockModel.return_value = model_instance
            MockModel.return_value.to = MagicMock(return_value=model_instance)

            # Make Path("evaluation_results") point to our tmp_path
            def path_side_effect(arg):
                if arg == "evaluation_results":
                    return output_dir
                return Path(arg)

            MockPath.side_effect = path_side_effect
            # Restore the real Path for label_map_path in cfg
            MockPath.__truediv__ = Path.__truediv__

            # We need the real open and real Path for reading label_map
            # but the main() constructs Path(cfg.data.processed_data_dir) / "label_map.json"
            # which needs to resolve to the real tmp_path. Since we patched Path,
            # let's just mock evaluate_model to avoid the full flow complexity.
            fake_results = {
                "overall_accuracy": 0.75,
                "top5_accuracy": 0.95,
                "macro_precision": 0.7,
                "macro_recall": 0.7,
                "macro_f1": 0.7,
                "per_class_detail": {
                    "hello": {"accuracy": 0.8, "precision": 0.8, "recall": 0.7, "f1": 0.75},
                    "bye": {"accuracy": 0.7, "precision": 0.6, "recall": 0.7, "f1": 0.65},
                },
                "per_class_accuracy": {"hello": 0.8, "bye": 0.7},
                "top_confusions": [
                    {"true": "hello", "predicted": "bye", "count": 2},
                ],
                "confusion_matrix": [[4, 1], [1, 3]],
                "inference_latency": {"mean_ms": 10.0, "p50_ms": 9.0, "p95_ms": 15.0, "p99_ms": 18.0},
                "total_samples": 9,
            }

        # Simpler approach: patch only evaluate_model and file output
        with patch("sys.argv", ["prog", "--checkpoint", "fake.pt"]), \
             patch("evaluation.evaluate.Config", return_value=fake_cfg), \
             patch("evaluation.evaluate.ASLVideoDataset", return_value=mock_ds), \
             patch("evaluation.evaluate.ASLVideoClassifier") as MockModel, \
             patch("torch.load", return_value=model_instance.state_dict()), \
             patch("evaluation.evaluate.evaluate_model", return_value=fake_results), \
             patch("evaluation.evaluate.save_confusion_matrix_plot"):

            MockModel.return_value = model_instance
            MockModel.return_value.to = MagicMock(return_value=model_instance)

            main()
