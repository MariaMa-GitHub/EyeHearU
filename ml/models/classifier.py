"""
ASL Video Classifier — 3D CNN backbone + classification head.

Approach B: Train a video classifier on short clips of isolated ASL signs.

The backbone (e.g., I3D ResNet-50 or R3D-18) is pretrained on Kinetics-400
and fine-tuned on our processed ASL clip dataset.  Input is a tensor of
shape (B, C, T, H, W) — batch of video clips with C=3 channels,
T=16 uniformly sampled frames, H=W=224.

TODO (ML team): Select the final backbone architecture after benchmarking
      I3D, SlowFast, and R3D on the processed dataset.
"""

import torch
import torch.nn as nn


class ASLVideoClassifier(nn.Module):
    """
    Video classifier for isolated ASL sign recognition.

    Wraps a pretrained 3D CNN backbone (from torchvision or PyTorchVideo)
    with a custom classification head.
    """

    def __init__(
        self,
        num_classes: int = 2000,
        backbone: str = "r3d_18",
        pretrained: bool = True,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone, feature_dim = self._build_backbone(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def _build_backbone(self, name: str, pretrained: bool):
        """
        Build and return the video backbone, stripping the original
        classification head so we can attach our own.

        Currently supports torchvision 3D models.  PyTorchVideo models
        (I3D, SlowFast) can be added with minimal changes.
        """
        import torchvision.models.video as video_models

        if name == "r3d_18":
            weights = "R3D_18_Weights.KINETICS400_V1" if pretrained else None
            model = video_models.r3d_18(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_dim

        if name == "mc3_18":
            weights = "MC3_18_Weights.KINETICS400_V1" if pretrained else None
            model = video_models.mc3_18(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_dim

        if name == "r2plus1d_18":
            weights = "R2Plus1D_18_Weights.KINETICS400_V1" if pretrained else None
            model = video_models.r2plus1d_18(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_dim

        raise ValueError(
            f"Unsupported backbone: {name}. "
            f"Choose from: r3d_18, mc3_18, r2plus1d_18"
        )

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) video clip tensor

        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.backbone(x)  # (B, feature_dim)
        return self.head(features)
