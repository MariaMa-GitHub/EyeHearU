"""Tests for i3d_msft/pytorch_i3d.py — Inception I3D model (vendored from Microsoft).

This is third-party model architecture code, but we test its public API to
ensure our vendored copy works correctly with our pipeline. Tests run on CPU
with random weights (no pretrained checkpoint needed).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from i3d_msft.pytorch_i3d import (
    Identity,
    MaxPool3dSamePadding,
    Unit3D,
    InceptionModule,
    InceptionI3d,
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_passthrough(self):
        layer = Identity()
        x = torch.randn(2, 3)
        assert torch.equal(layer(x), x)


# ---------------------------------------------------------------------------
# MaxPool3dSamePadding
# ---------------------------------------------------------------------------

class TestMaxPool3dSamePadding:
    def test_compute_pad_divisible(self):
        pool = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        # s=8, stride=2, kernel=3 → 8%2==0 → max(3-2,0)=1
        assert pool.compute_pad(0, 8) == 1

    def test_compute_pad_non_divisible(self):
        pool = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        # s=7, stride=2, kernel=3 → 7%2==1 → max(3-1,0)=2
        assert pool.compute_pad(0, 7) == 2

    def test_forward_shape(self):
        pool = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        x = torch.randn(1, 3, 8, 16, 16)
        out = pool(x)
        # "same" padding with stride 2 → ceil(dim/stride)
        assert out.shape[2] == 4  # ceil(8/2)
        assert out.shape[3] == 8  # ceil(16/2)
        assert out.shape[4] == 8


# ---------------------------------------------------------------------------
# Unit3D
# ---------------------------------------------------------------------------

class TestUnit3D:
    def test_forward_with_batch_norm(self):
        unit = Unit3D(in_channels=3, output_channels=16, kernel_shape=[3, 3, 3])
        x = torch.randn(1, 3, 8, 16, 16)
        out = unit(x)
        assert out.shape[1] == 16

    def test_forward_without_batch_norm(self):
        unit = Unit3D(in_channels=3, output_channels=8, kernel_shape=[1, 1, 1],
                      use_batch_norm=False, use_bias=True)
        x = torch.randn(1, 3, 4, 8, 8)
        out = unit(x)
        assert out.shape[1] == 8

    def test_forward_no_activation(self):
        unit = Unit3D(in_channels=3, output_channels=8, kernel_shape=[1, 1, 1],
                      activation_fn=None, use_batch_norm=False, use_bias=True)
        x = torch.randn(1, 3, 4, 8, 8)
        out = unit(x)
        # Without ReLU, output can be negative
        assert out.min() < 0 or out.max() > 0  # sanity check: not all zeros

    def test_compute_pad_divisible(self):
        unit = Unit3D(in_channels=3, output_channels=8, kernel_shape=[3, 3, 3], stride=(2, 2, 2))
        # s=8, stride=2, kernel=3 → 8%2==0 → max(3-2,0)=1
        assert unit.compute_pad(0, 8) == 1

    def test_compute_pad_non_divisible(self):
        unit = Unit3D(in_channels=3, output_channels=8, kernel_shape=[3, 3, 3], stride=(2, 2, 2))
        # s=7, stride=2, kernel=3 → 7%2==1 → max(3-1,0)=2
        assert unit.compute_pad(0, 7) == 2


# ---------------------------------------------------------------------------
# InceptionModule
# ---------------------------------------------------------------------------

class TestInceptionModule:
    def test_forward_shape(self):
        # out_channels: [64, 96, 128, 16, 32, 32] → output = 64+128+32+32 = 256
        mod = InceptionModule(192, [64, 96, 128, 16, 32, 32], "test")
        x = torch.randn(1, 192, 4, 14, 14)
        out = mod(x)
        assert out.shape[1] == 256  # sum of branch outputs


# ---------------------------------------------------------------------------
# InceptionI3d
# ---------------------------------------------------------------------------

class TestInceptionI3d:
    def test_invalid_endpoint_raises(self):
        with pytest.raises(ValueError, match="Unknown final endpoint"):
            InceptionI3d(final_endpoint="InvalidEndpoint")

    def test_default_forward_shape(self):
        """Full model with default Logits endpoint, 10 classes."""
        model = InceptionI3d(num_classes=10)
        x = torch.randn(1, 3, 64, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(x)
        # spatial_squeeze removes the two trailing spatial dims (H, W → squeezed)
        # output is (B, num_classes, T') where T' depends on temporal pooling
        assert out.shape[0] == 1
        assert out.shape[1] == 10

    def test_replace_logits(self):
        model = InceptionI3d(num_classes=400)
        model.replace_logits(856)
        assert model._num_classes == 856

    def test_remove_last(self):
        model = InceptionI3d(num_classes=10)
        model.remove_last()
        assert isinstance(model.logits, Identity)

    def test_early_endpoint_conv3d_1a(self):
        """Construct with early endpoint — only first conv layer built."""
        model = InceptionI3d(num_classes=10, final_endpoint="Conv3d_1a_7x7")
        assert "Conv3d_1a_7x7" in model.end_points
        assert "MaxPool3d_2a_3x3" not in model.end_points

    def test_early_endpoint_mixed_3b(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Mixed_3b")
        assert "Mixed_3b" in model.end_points
        assert "Mixed_3c" not in model.end_points

    def test_early_endpoint_maxpool_2a(self):
        model = InceptionI3d(num_classes=10, final_endpoint="MaxPool3d_2a_3x3")
        assert "MaxPool3d_2a_3x3" in model.end_points
        assert "Conv3d_2b_1x1" not in model.end_points

    def test_early_endpoint_conv3d_2b(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Conv3d_2b_1x1")
        assert "Conv3d_2b_1x1" in model.end_points
        assert "Conv3d_2c_3x3" not in model.end_points

    def test_early_endpoint_conv3d_2c(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Conv3d_2c_3x3")
        assert "Conv3d_2c_3x3" in model.end_points
        assert "MaxPool3d_3a_3x3" not in model.end_points

    def test_early_endpoint_maxpool_3a(self):
        model = InceptionI3d(num_classes=10, final_endpoint="MaxPool3d_3a_3x3")
        assert "MaxPool3d_3a_3x3" in model.end_points
        assert "Mixed_3b" not in model.end_points

    def test_early_endpoint_mixed_3c(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Mixed_3c")
        assert "Mixed_3c" in model.end_points
        assert "MaxPool3d_4a_3x3" not in model.end_points

    def test_early_endpoint_maxpool_4a(self):
        model = InceptionI3d(num_classes=10, final_endpoint="MaxPool3d_4a_3x3")
        assert "MaxPool3d_4a_3x3" in model.end_points
        assert "Mixed_4b" not in model.end_points

    def test_early_endpoint_mixed_4b(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Mixed_4b")
        assert "Mixed_4b" in model.end_points
        assert "Mixed_4c" not in model.end_points

    def test_early_endpoint_mixed_4c(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Mixed_4c")
        assert "Mixed_4c" in model.end_points
        assert "Mixed_4d" not in model.end_points

    def test_early_endpoint_mixed_4d(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Mixed_4d")
        assert "Mixed_4d" in model.end_points
        assert "Mixed_4e" not in model.end_points

    def test_early_endpoint_mixed_4e(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Mixed_4e")
        assert "Mixed_4e" in model.end_points
        assert "Mixed_4f" not in model.end_points

    def test_early_endpoint_mixed_4f(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Mixed_4f")
        assert "Mixed_4f" in model.end_points
        assert "MaxPool3d_5a_2x2" not in model.end_points

    def test_early_endpoint_maxpool_5a(self):
        model = InceptionI3d(num_classes=10, final_endpoint="MaxPool3d_5a_2x2")
        assert "MaxPool3d_5a_2x2" in model.end_points
        assert "Mixed_5b" not in model.end_points

    def test_early_endpoint_mixed_5b(self):
        model = InceptionI3d(num_classes=10, final_endpoint="Mixed_5b")
        assert "Mixed_5b" in model.end_points
        assert "Mixed_5c" not in model.end_points

    def test_forward_pretrained_mode(self):
        """Forward with pretrained=True freezes some layers."""
        model = InceptionI3d(num_classes=10)
        x = torch.randn(1, 3, 64, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(x, pretrained=True, n_tune_layers=3)
        assert out.shape[0] == 1
        assert out.shape[1] == 10

    def test_forward_no_spatial_squeeze(self):
        """spatial_squeeze=False keeps extra dimensions."""
        model = InceptionI3d(num_classes=10, spatial_squeeze=False)
        x = torch.randn(1, 3, 64, 224, 224)
        model.eval()
        with torch.no_grad():
            out = model(x)
        # Without squeeze, output keeps spatial dims
        assert len(out.shape) == 5

    def test_extract_features(self):
        model = InceptionI3d(num_classes=10)
        x = torch.randn(1, 3, 64, 224, 224)
        model.eval()
        with torch.no_grad():
            features = model.extract_features(x)
        # After avg_pool, features should be (B, 1024, 1, 1, 1)
        assert features.shape[1] == 1024
