"""Unit tests for evaluation metrics."""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import (
    compute_topk_accuracy,
    compute_precision_recall_f1,
    build_confusion_matrix,
    set_seed,
)


class TestTopKAccuracy:
    def test_perfect_predictions(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        labels = torch.tensor([0, 1, 2])
        assert compute_topk_accuracy(logits, labels, k=1) == pytest.approx(1.0)

    def test_all_wrong_top1(self):
        logits = torch.tensor([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])
        labels = torch.tensor([0, 1, 2])
        assert compute_topk_accuracy(logits, labels, k=1) == pytest.approx(0.0)

    def test_top5_includes_correct(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        labels = torch.tensor([1])
        assert compute_topk_accuracy(logits, labels, k=5) == pytest.approx(1.0)

    def test_top1_misses_but_top5_hits(self):
        logits = torch.tensor([[1.0, 2.0, 5.0, 4.0, 3.0]])
        labels = torch.tensor([1])
        assert compute_topk_accuracy(logits, labels, k=1) == pytest.approx(0.0)
        assert compute_topk_accuracy(logits, labels, k=5) == pytest.approx(1.0)

    def test_k_larger_than_num_classes(self):
        """k > num_classes should not raise; clamps k to num_classes."""
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        labels = torch.tensor([0, 1])
        # k=5 but only 3 classes — must not raise and correct label is always in top-3
        acc = compute_topk_accuracy(logits, labels, k=5)
        assert acc == pytest.approx(1.0)


class TestPrecisionRecallF1:
    def test_perfect_classification(self):
        preds = torch.tensor([0, 1, 2, 0, 1, 2])
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        per_class, macro_p, macro_r, macro_f1 = compute_precision_recall_f1(preds, labels, 3)
        assert macro_p == pytest.approx(1.0)
        assert macro_r == pytest.approx(1.0)
        assert macro_f1 == pytest.approx(1.0)

    def test_all_wrong(self):
        preds = torch.tensor([1, 2, 0])
        labels = torch.tensor([0, 1, 2])
        per_class, macro_p, macro_r, macro_f1 = compute_precision_recall_f1(preds, labels, 3)
        assert macro_p == pytest.approx(0.0)
        assert macro_f1 == pytest.approx(0.0)

    def test_partial_correctness(self):
        preds = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, 1, 1, 0])
        per_class, macro_p, macro_r, macro_f1 = compute_precision_recall_f1(preds, labels, 2)
        assert 0.0 < macro_f1 < 1.0

    def test_empty_class_handled(self):
        preds = torch.tensor([0, 0, 0])
        labels = torch.tensor([0, 0, 0])
        per_class, macro_p, macro_r, macro_f1 = compute_precision_recall_f1(preds, labels, 3)
        assert per_class[1]["precision"] == 0.0
        assert per_class[1]["recall"] == 0.0


class TestConfusionMatrix:
    def test_shape(self):
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])
        cm = build_confusion_matrix(preds, labels, 3)
        assert cm.shape == (3, 3)

    def test_perfect_diagonal(self):
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])
        cm = build_confusion_matrix(preds, labels, 3)
        assert np.trace(cm) == 3
        assert cm.sum() == 3

    def test_all_misclassified(self):
        preds = torch.tensor([1, 0])
        labels = torch.tensor([0, 1])
        cm = build_confusion_matrix(preds, labels, 2)
        assert cm[0, 0] == 0
        assert cm[0, 1] == 1
        assert cm[1, 0] == 1
        assert cm[1, 1] == 0


class TestSeedReproducibility:
    def test_seed_determinism(self):
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.allclose(a, b)
