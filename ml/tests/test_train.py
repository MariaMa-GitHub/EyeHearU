"""Tests for training/train.py — training loop functions.

Covers:
  - train_one_epoch: forward + backward pass with a tiny model
  - evaluate: inference-only evaluation loop
  - _worker_init_fn: DataLoader worker seed initialiser
  - main: full orchestration (heavily mocked)
"""

import json
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train import train_one_epoch, evaluate, _worker_init_fn, set_seed, main, SEED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model(num_classes=3):
    """A minimal model that accepts (B, 3, 4, 8, 8) video-like tensors."""
    return nn.Sequential(
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Linear(3, num_classes),
    )


def _synthetic_loader(n=8, num_classes=3, batch_size=4):
    """DataLoader of random (B, 3, 4, 8, 8) tensors with random labels."""
    clips = torch.randn(n, 3, 4, 8, 8)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(clips, labels), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:
    """train_one_epoch runs forward+backward on a tiny model."""

    def test_returns_loss_and_accuracy(self):
        model = _tiny_model()
        loader = _synthetic_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loss, acc = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0

    def test_model_parameters_update(self):
        """Verify weights actually change after a training step."""
        model = _tiny_model()
        loader = _synthetic_loader()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        params_before = [p.clone() for p in model.parameters()]
        train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        params_after = list(model.parameters())

        changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        assert changed, "Parameters should change after training"

    def test_empty_loader_returns_zero(self):
        """Empty DataLoader should return loss=0, acc=0 without crashing."""
        model = _tiny_model()
        empty_loader = DataLoader(TensorDataset(torch.empty(0, 3, 4, 8, 8), torch.empty(0, dtype=torch.long)))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loss, acc = train_one_epoch(model, empty_loader, criterion, optimizer, torch.device("cpu"))
        assert loss == 0.0
        assert acc == 0.0


class TestEvaluate:
    """evaluate() runs inference without gradients."""

    def test_returns_loss_and_accuracy(self):
        model = _tiny_model()
        loader = _synthetic_loader()
        criterion = nn.CrossEntropyLoss()

        loss, acc = evaluate(model, loader, criterion, torch.device("cpu"))
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0

    def test_no_gradient_accumulation(self):
        """Model should be in eval mode and no gradients tracked."""
        model = _tiny_model()
        loader = _synthetic_loader()
        criterion = nn.CrossEntropyLoss()

        evaluate(model, loader, criterion, torch.device("cpu"))
        for p in model.parameters():
            assert p.grad is None

    def test_empty_loader_returns_zero(self):
        model = _tiny_model()
        empty_loader = DataLoader(TensorDataset(torch.empty(0, 3, 4, 8, 8), torch.empty(0, dtype=torch.long)))
        criterion = nn.CrossEntropyLoss()

        loss, acc = evaluate(model, empty_loader, criterion, torch.device("cpu"))
        assert loss == 0.0
        assert acc == 0.0


class TestWorkerInitFn:
    """_worker_init_fn seeds numpy and random for DataLoader workers."""

    def test_sets_numpy_seed(self):
        _worker_init_fn(0)
        a = np.random.rand()
        _worker_init_fn(0)
        b = np.random.rand()
        assert a == b

    def test_sets_random_seed(self):
        _worker_init_fn(5)
        a = random.random()
        _worker_init_fn(5)
        b = random.random()
        assert a == b

    def test_different_workers_get_different_seeds(self):
        _worker_init_fn(0)
        a = np.random.rand()
        _worker_init_fn(1)
        b = np.random.rand()
        assert a != b


class TestSetSeed:
    """set_seed in train.py also sets cudnn flags."""

    def test_deterministic_flags(self):
        set_seed(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


class TestMainFunction:
    """main() orchestration — heavily mocked to avoid real data/GPU."""

    def test_main_with_label_map(self, tmp_path):
        """main() runs the full training loop when label_map.json exists."""
        # Create label_map
        label_map = {"hello": 0, "bye": 1}
        (tmp_path / "label_map.json").write_text(json.dumps(label_map))

        # Create mock dataset that returns synthetic data
        mock_ds = TensorDataset(torch.randn(4, 3, 16, 112, 112), torch.randint(0, 2, (4,)))

        fake_cfg = MagicMock()
        fake_cfg.data.processed_data_dir = str(tmp_path)
        fake_cfg.data.num_frames = 16
        fake_cfg.data.num_classes = 2
        fake_cfg.data.num_workers = 0
        fake_cfg.train.device = "cpu"
        fake_cfg.train.batch_size = 2
        fake_cfg.train.learning_rate = 0.001
        fake_cfg.train.weight_decay = 0.01
        fake_cfg.train.epochs = 2
        fake_cfg.train.checkpoint_dir = str(tmp_path / "checkpoints")
        fake_cfg.train.save_every_n_epochs = 1
        fake_cfg.train.early_stopping_patience = 10
        fake_cfg.model.backbone = "r3d_18"
        fake_cfg.model.pretrained = False
        fake_cfg.model.head_dropout = 0.0
        fake_cfg.model.backbone_freeze_epochs = 0

        with patch("training.train.Config", return_value=fake_cfg), \
             patch("training.train.ASLVideoDataset", return_value=mock_ds), \
             patch("training.train.ASLVideoClassifier") as MockModel:

            # Make the model mock behave like a real model
            model_instance = _tiny_model(num_classes=2)
            MockModel.return_value = model_instance
            # .to(device) returns self
            MockModel.return_value.to = MagicMock(return_value=model_instance)
            model_instance.freeze_backbone = MagicMock()
            model_instance.unfreeze_backbone = MagicMock()

            main()

        # Verify checkpoints were saved
        assert (tmp_path / "checkpoints").exists()

    def test_main_without_label_map(self, tmp_path):
        """main() falls back to default num_classes when label_map.json missing."""
        mock_ds = TensorDataset(torch.randn(4, 3, 16, 112, 112), torch.randint(0, 2, (4,)))

        fake_cfg = MagicMock()
        fake_cfg.data.processed_data_dir = str(tmp_path)  # no label_map.json here
        fake_cfg.data.num_frames = 16
        fake_cfg.data.num_classes = 5
        fake_cfg.data.num_workers = 0
        fake_cfg.train.device = "cpu"
        fake_cfg.train.batch_size = 2
        fake_cfg.train.learning_rate = 0.001
        fake_cfg.train.weight_decay = 0.01
        fake_cfg.train.epochs = 1
        fake_cfg.train.checkpoint_dir = str(tmp_path / "checkpoints")
        fake_cfg.train.save_every_n_epochs = 1
        fake_cfg.train.early_stopping_patience = 10
        fake_cfg.model.backbone = "r3d_18"
        fake_cfg.model.pretrained = False
        fake_cfg.model.head_dropout = 0.0
        fake_cfg.model.backbone_freeze_epochs = 0

        with patch("training.train.Config", return_value=fake_cfg), \
             patch("training.train.ASLVideoDataset", return_value=mock_ds), \
             patch("training.train.ASLVideoClassifier") as MockModel:

            model_instance = _tiny_model(num_classes=5)
            MockModel.return_value = model_instance
            MockModel.return_value.to = MagicMock(return_value=model_instance)
            model_instance.freeze_backbone = MagicMock()
            model_instance.unfreeze_backbone = MagicMock()

            main()

        # Should have used num_classes=5 from config
        MockModel.assert_called_once_with(
            num_classes=5,
            backbone="r3d_18",
            pretrained=False,
            head_dropout=0.0,
        )

    def test_main_early_stopping(self, tmp_path):
        """main() stops early when validation accuracy doesn't improve."""
        label_map = {"a": 0, "b": 1}
        (tmp_path / "label_map.json").write_text(json.dumps(label_map))

        mock_ds = TensorDataset(torch.randn(4, 3, 16, 112, 112), torch.randint(0, 2, (4,)))

        fake_cfg = MagicMock()
        fake_cfg.data.processed_data_dir = str(tmp_path)
        fake_cfg.data.num_frames = 16
        fake_cfg.data.num_classes = 2
        fake_cfg.data.num_workers = 0
        fake_cfg.train.device = "cpu"
        fake_cfg.train.batch_size = 4
        fake_cfg.train.learning_rate = 0.001
        fake_cfg.train.weight_decay = 0.01
        fake_cfg.train.epochs = 100  # many epochs, but early stopping should trigger
        fake_cfg.train.checkpoint_dir = str(tmp_path / "checkpoints")
        fake_cfg.train.save_every_n_epochs = 50
        fake_cfg.train.early_stopping_patience = 2  # very low patience
        fake_cfg.model.backbone = "r3d_18"
        fake_cfg.model.pretrained = False
        fake_cfg.model.head_dropout = 0.0
        fake_cfg.model.backbone_freeze_epochs = 1

        call_count = [0]

        def fake_evaluate(model, loader, criterion, device):
            call_count[0] += 1
            # Return decreasing accuracy to trigger early stopping
            return 1.0, max(0.0, 0.5 - call_count[0] * 0.1)

        with patch("training.train.Config", return_value=fake_cfg), \
             patch("training.train.ASLVideoDataset", return_value=mock_ds), \
             patch("training.train.ASLVideoClassifier") as MockModel, \
             patch("training.train.evaluate", side_effect=fake_evaluate):

            model_instance = _tiny_model(num_classes=2)
            MockModel.return_value = model_instance
            MockModel.return_value.to = MagicMock(return_value=model_instance)
            model_instance.freeze_backbone = MagicMock()
            model_instance.unfreeze_backbone = MagicMock()

            main()

        # Should have stopped well before 100 epochs
        assert call_count[0] < 10

    def test_main_empty_dataset_exits(self, tmp_path):
        """main() exits with code 1 when training dataset is empty."""
        label_map = {"a": 0}
        (tmp_path / "label_map.json").write_text(json.dumps(label_map))

        empty_ds = TensorDataset(torch.empty(0, 3, 16, 112, 112), torch.empty(0, dtype=torch.long))

        fake_cfg = MagicMock()
        fake_cfg.data.processed_data_dir = str(tmp_path)
        fake_cfg.data.num_frames = 16
        fake_cfg.data.num_workers = 0
        fake_cfg.train.device = "cpu"
        fake_cfg.train.batch_size = 2

        with patch("training.train.Config", return_value=fake_cfg), \
             patch("training.train.ASLVideoDataset", return_value=empty_ds), \
             pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
