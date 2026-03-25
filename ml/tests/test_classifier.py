"""Unit tests for the ASLVideoClassifier model."""

import pytest
import torch

from models.classifier import ASLVideoClassifier


@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 16, 224, 224)


class TestASLVideoClassifier:
    def test_forward_shape(self, dummy_input):
        model = ASLVideoClassifier(num_classes=10, backbone="r3d_18", pretrained=False)
        out = model(dummy_input)
        assert out.shape == (2, 10)

    def test_different_num_classes(self, dummy_input):
        for n in [5, 48, 2000]:
            model = ASLVideoClassifier(num_classes=n, backbone="r3d_18", pretrained=False)
            out = model(dummy_input)
            assert out.shape == (2, n)

    def test_freeze_unfreeze_backbone(self):
        model = ASLVideoClassifier(num_classes=10, backbone="r3d_18", pretrained=False)
        model.freeze_backbone()
        for name, p in model.backbone.named_parameters():
            if "fc" not in name:
                assert not p.requires_grad

        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad

    def test_mc3_backbone(self, dummy_input):
        model = ASLVideoClassifier(num_classes=10, backbone="mc3_18", pretrained=False)
        out = model(dummy_input)
        assert out.shape == (2, 10)

    def test_r2plus1d_backbone(self, dummy_input):
        model = ASLVideoClassifier(num_classes=10, backbone="r2plus1d_18", pretrained=False)
        out = model(dummy_input)
        assert out.shape == (2, 10)

    def test_unsupported_backbone(self):
        with pytest.raises(ValueError, match="Unsupported backbone"):
            ASLVideoClassifier(num_classes=10, backbone="nonexistent", pretrained=False)

    def test_head_dropout(self, dummy_input):
        model = ASLVideoClassifier(num_classes=10, backbone="r3d_18", pretrained=False, head_dropout=0.0)
        model.eval()
        out1 = model(dummy_input)
        out2 = model(dummy_input)
        assert torch.allclose(out1, out2)
