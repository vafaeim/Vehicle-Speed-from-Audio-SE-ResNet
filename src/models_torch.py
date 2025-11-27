"""
PyTorch Implementation of Squeeze-and-Excitation ResNet (SE-ResNet) for Acoustic Vehicle Speed Estimation.

Architecture adheres strictly to the manuscript specifications:
- Input shape: (B, 1, 128, 313) Mel-Spectrograms (128 mel bins, 313 time frames).
- Entry flow stem: Conv2d(7x7, stride=2) -> BatchNorm2d -> ReLU -> MaxPool2d(3x3, stride=2).
- Configurable residual stages:
    * Standard (default): 3 stages [2, 2, 2] with channels [96, 192, 384] (~2.26M params)
    * Shallow: 2 stages [2, 2] with channels [96, 192] (~0.58M params)
    * Deep: 4 stages [2, 2, 2, 2] with channels [96, 192, 384, 512] (~5.11M params)
- Configurable Squeeze-and-Excitation (SE) channel recalibration:
    * use_se: bool (toggle for baseline ResNet ablation, ~2.14M params)
    * reduction ratio r in {8, 16, 32} with bottleneck max(1, channels // r)
- Global Average Pooling (GAP), configurable Dropout (0.10 to 0.50), and Linear(1) regression head.
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise recalibration.

    Formulation (Manuscript Equations 2 & 3):
        z_c = F_sq(u_c) = (1 / (F * T)) * sum(u_c)  [Global Average Pooling]
        s = F_ex(z, W) = sigmoid(W_2 * relu(W_1 * z))  [Channel Excitation]
        output = s * u
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.bottleneck_channels = max(1, channels // reduction)

        self.fc1 = nn.Linear(channels, self.bottleneck_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.bottleneck_channels, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        b, c, _, _ = x.shape
        # Squeeze: Spatial Global Average Pooling -> (B, C)
        s = x.mean(dim=(2, 3))
        # Excitation: Two-layer bottleneck MLP with sigmoid activation
        e = self.relu(self.fc1(s))
        e = self.sigmoid(self.fc2(e))
        # Recalibrate channels: (B, C, 1, 1) * (B, C, H, W)
        return x * e.view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    """
    Residual block integrated with optional Squeeze-and-Excitation mechanism.

    Structure:
        x -> Conv2D(3x3, stride) -> BN -> ReLU -> Conv2D(3x3, 1) -> BN -> [SE] -> (+) -> ReLU
        |___________________________ (1x1 Conv + BN if downsample) ______________|
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
        use_se: bool = True,
        se_ratio: int = 16,
    ):
        super().__init__()
        actual_stride = 2 if (downsample or stride == 2) else 1

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=actual_stride,
            padding=1,
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels, reduction=se_ratio)
        else:
            self.se = nn.Identity()

        # Shortcut projection if spatial resolution or channel capacity changes
        if actual_stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=actual_stride,
                    bias=True,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)
        out = out + identity
        out = self.relu(out)
        return out


class ConvStem(nn.Module):
    """
    Entry Flow convolutional stem matching manuscript Figure 2 and TensorFlow baseline:
    Conv2D(7x7, stride=2, padding=3) -> BatchNorm2d -> ReLU -> MaxPool2d(3x3, stride=2, padding=1)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 96,
        use_maxpool: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class SEResNet(nn.Module):
    """
    Parametric Squeeze-and-Excitation ResNet for Audio Speed Estimation.

    Parameters:
    -----------
    in_channels : int, default=1
        Input channels (single Mel-spectrogram channel).
    base_filters : int, default=96
        Width of first convolutional stage.
    stage_blocks : Sequence[int], optional
        Number of residual blocks per stage. Default: [2, 2, 2].
    stage_channels : Sequence[int], optional
        Filter widths for each stage. Default: [96, 192, 384].
    use_se : bool, default=True
        Whether to include Squeeze-and-Excitation channel recalibration.
    se_ratio : int, default=16
        Bottleneck reduction ratio r in {8, 16, 32}.
    dropout : float, default=0.30
        Dropout probability in regression head (0.10 to 0.50).
    stages : int, optional
        Predefined stage depth helper:
          - 2: Shallow [2, 2] with channels [96, 192]
          - 3: Standard [2, 2, 2] with channels [96, 192, 384]
          - 4: Deep [2, 2, 2, 2] with channels [96, 192, 384, 512]
    use_maxpool : bool, default=True
        Whether to include MaxPool2d in entry stem.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 96,
        stage_blocks: Optional[Sequence[int]] = None,
        stage_channels: Optional[Sequence[int]] = None,
        use_se: bool = True,
        se_ratio: int = 16,
        dropout: float = 0.3,
        stages: Optional[int] = None,
        use_maxpool: bool = True,
    ):
        super().__init__()

        # Resolve predefined stage configurations
        if stages is not None:
            if stages == 2:
                stage_blocks = [2, 2]
                stage_channels = [base_filters, base_filters * 2]  # [96, 192]
            elif stages == 3:
                stage_blocks = [2, 2, 2]
                stage_channels = [base_filters, base_filters * 2, base_filters * 4]  # [96, 192, 384]
            elif stages == 4:
                stage_blocks = [2, 2, 2, 2]
                stage_channels = [base_filters, base_filters * 2, base_filters * 4, 512]  # [96, 192, 384, 512]
            else:
                raise ValueError(f"Unsupported stages count: {stages}. Choose from 2, 3, or 4.")

        if stage_blocks is None:
            stage_blocks = [2, 2, 2]

        if stage_channels is None:
            if len(stage_blocks) == 2:
                stage_channels = [base_filters, base_filters * 2]
            elif len(stage_blocks) == 3:
                stage_channels = [base_filters, base_filters * 2, base_filters * 4]
            elif len(stage_blocks) == 4:
                stage_channels = [base_filters, base_filters * 2, base_filters * 4, 512]
            else:
                stage_channels = [base_filters * (2 ** i) for i in range(len(stage_blocks))]

        if len(stage_blocks) != len(stage_channels):
            raise ValueError(
                f"Length mismatch: stage_blocks ({len(stage_blocks)}) vs stage_channels ({len(stage_channels)})"
            )

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.stage_blocks = list(stage_blocks)
        self.stage_channels = list(stage_channels)
        self.use_se = use_se
        self.se_ratio = se_ratio
        self.dropout_rate = dropout
        self.use_maxpool = use_maxpool

        # 1. Entry Flow Stem
        self.stem = ConvStem(
            in_channels=in_channels,
            out_channels=self.stage_channels[0],
            use_maxpool=use_maxpool,
        )

        # 2. Residual Stages
        current_channels = self.stage_channels[0]
        stage_modules: List[nn.Module] = []

        for stage_idx, (num_blocks, out_ch) in enumerate(zip(self.stage_blocks, self.stage_channels)):
            blocks: List[nn.Module] = []
            for block_idx in range(num_blocks):
                # Downsample on first block of all stages after Stage 1
                downsample = (stage_idx > 0 and block_idx == 0)
                blocks.append(
                    ResidualBlock(
                        in_channels=current_channels,
                        out_channels=out_ch,
                        downsample=downsample,
                        use_se=use_se,
                        se_ratio=se_ratio,
                    )
                )
                current_channels = out_ch
            stage_modules.append(nn.Sequential(*blocks))

        self.stages = nn.Sequential(*stage_modules)

        # 3. Global Average Pooling & Regression Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(current_channels, 1, bias=True)

        # Explicit alias for modular inspection
        self.regression_head = nn.Sequential(self.dropout, self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: Tensor of shape (B, 1, 128, 313)
        Output: Tensor of shape (B, 1) representing vehicle speed (km/h).
        """
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


def build_se_resnet(
    input_shape: Tuple[int, ...] = (1, 128, 313),
    in_channels: int = 1,
    base_filters: int = 96,
    stage_blocks: Optional[Sequence[int]] = None,
    stage_channels: Optional[Sequence[int]] = None,
    use_se: bool = True,
    se_ratio: int = 16,
    dropout: float = 0.3,
    stages: Optional[int] = None,
    use_maxpool: bool = True,
) -> SEResNet:
    """
    Factory function to instantiate SE-ResNet matching manuscript specifications.

    Handles input_shape tuples in either (C, H, W) or (H, W, C) layout:
    - (1, 128, 313) -> in_channels = 1
    - (128, 313, 1) -> in_channels = 1
    """
    if len(input_shape) == 3:
        if input_shape[0] == 1:
            in_channels = input_shape[0]
        elif input_shape[-1] == 1:
            in_channels = input_shape[-1]

    return SEResNet(
        in_channels=in_channels,
        base_filters=base_filters,
        stage_blocks=stage_blocks,
        stage_channels=stage_channels,
        use_se=use_se,
        se_ratio=se_ratio,
        dropout=dropout,
        stages=stages,
        use_maxpool=use_maxpool,
    )


def build_ablation_model(
    input_shape: Tuple[int, ...] = (1, 128, 313),
    use_se: bool = True,
    se_ratio: int = 16,
    stages: int = 3,
    base_filters: int = 96,
    dropout: float = 0.3,
) -> SEResNet:
    """
    Factory function for ablation experiments (PROJECT.md contract).
    """
    return build_se_resnet(
        input_shape=input_shape,
        base_filters=base_filters,
        use_se=use_se,
        se_ratio=se_ratio,
        stages=stages,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
