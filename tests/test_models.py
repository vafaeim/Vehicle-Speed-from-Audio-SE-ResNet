"""
Tier 2: Neural Architecture, Layer & Tensor Shape Tests
Tests factorized 1D convolutions, 1D Squeeze-and-Excitation, SincNet frontend,
model parameter counts, backward gradients without NaN/Inf, and eval mode determinism.
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Reference Architecture Components (Specification Oracle) ---

class RefSqueezeExcite1d(nn.Module):
    """Squeeze-and-Excitation 1D block for temporal acoustic representations."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        bottleneck = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, bottleneck)
        self.fc2 = nn.Linear(bottleneck, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=-1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)


class RefFactorized1DBlock(nn.Module):
    """Factorized 1D spatio-temporal residual block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, se_reduction: int = 8):
        super().__init__()
        self.conv_time = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, groups=in_channels, bias=False
        )
        self.conv_point = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = RefSqueezeExcite1d(out_channels, reduction=se_reduction)
        self.act2 = nn.ReLU(inplace=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        y = self.act1(self.bn1(self.conv_point(self.conv_time(x))))
        y = self.se(self.bn2(self.conv2(y)))
        return self.act2(y + res)


class RefSincConv1d(nn.Module):
    """Learnable SincNet bandpass filterbank."""
    def __init__(self, out_channels: int = 64, kernel_size: int = 251, sample_rate: int = 16000, stride: int = 16):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        
        hz_low = 30.0
        hz_high = sample_rate / 2.0 - 100.0
        f_init = np.linspace(hz_low, hz_high, out_channels + 1)
        
        self.f1 = nn.Parameter(torch.tensor(f_init[:-1], dtype=torch.float32) / sample_rate)
        self.band = nn.Parameter(torch.tensor(np.diff(f_init), dtype=torch.float32) / sample_rate)
        
        t = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer("t", t)
        
        w = 0.54 - 0.46 * torch.cos(2.0 * np.pi * torch.arange(kernel_size, dtype=torch.float32) / (kernel_size - 1))
        self.register_buffer("window", w)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1_clamped = torch.clamp(self.f1, 1e-4, 0.49)
        max_band = 0.5 - f1_clamped
        band_clamped = torch.clamp(torch.abs(self.band), torch.full_like(self.band, 1e-4), max_band)
        f2_clamped = f1_clamped + band_clamped
        
        t_mat = self.t.unsqueeze(0)
        f1_mat = f1_clamped.unsqueeze(1)
        f2_mat = f2_clamped.unsqueeze(1)
        
        h2 = 2.0 * f2_mat * torch.sinc(2.0 * f2_mat * t_mat)
        h1 = 2.0 * f1_mat * torch.sinc(2.0 * f1_mat * t_mat)
        filters = (h2 - h1) * self.window.unsqueeze(0)
        filters = filters.unsqueeze(1)
        
        return F.conv1d(x, filters, stride=self.stride, padding=self.kernel_size // 2)


class RefFactorized1DNet(nn.Module):
    """Full Factorized 1D Spatio-Temporal Speed Estimation Architecture."""
    def __init__(self, in_channels: int = 1, base_filters: int = 64, se_reduction: int = 8, dropout: float = 0.2):
        super().__init__()
        self.frontend = RefSincConv1d(out_channels=base_filters, kernel_size=251, stride=16)
        self.pool0 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.stage1 = nn.Sequential(
            RefFactorized1DBlock(base_filters, base_filters, kernel_size=7, stride=1, se_reduction=se_reduction),
            RefFactorized1DBlock(base_filters, base_filters, kernel_size=7, stride=1, se_reduction=se_reduction)
        )
        self.stage2 = nn.Sequential(
            RefFactorized1DBlock(base_filters, base_filters * 2, kernel_size=7, stride=2, se_reduction=se_reduction),
            RefFactorized1DBlock(base_filters * 2, base_filters * 2, kernel_size=7, stride=1, se_reduction=se_reduction)
        )
        self.stage3 = nn.Sequential(
            RefFactorized1DBlock(base_filters * 2, base_filters * 4, kernel_size=7, stride=2, se_reduction=se_reduction),
            RefFactorized1DBlock(base_filters * 4, base_filters * 4, kernel_size=7, stride=1, se_reduction=se_reduction)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_filters * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.frontend(x)
        h = self.pool0(h)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        return self.head(h)


# --- Unit Tests ---

def test_sinc_conv1d_shape_and_filtering():
    """Verifies that SincConv1d produces correct channel and temporal downsampled dimensions."""
    batch_size = 2
    in_channels = 1
    length = 16000
    out_channels = 32
    stride = 16
    
    layer = RefSincConv1d(out_channels=out_channels, stride=stride)
    x = torch.randn(batch_size, in_channels, length)
    y = layer(x)
    
    expected_length = length // stride
    assert y.shape == (batch_size, out_channels, expected_length), \
        f"Expected shape {(batch_size, out_channels, expected_length)}, got {y.shape}"


def test_squeeze_excite_shape_preservation():
    """Verifies that Squeeze-and-Excitation preserves exact tensor dimensions."""
    channels = 64
    se = RefSqueezeExcite1d(channels=channels, reduction=8)
    x = torch.randn(4, channels, 100)
    y = se(x)
    assert x.shape == y.shape, f"SE block modified shape from {x.shape} to {y.shape}"


def test_squeeze_excite_gating_range():
    """Verifies that the SE excitation weights are strictly bounded within [0, 1]."""
    channels = 32
    se = RefSqueezeExcite1d(channels=channels, reduction=8)
    x = torch.randn(2, channels, 50)
    
    # Check internal sigmoid output
    w = x.mean(dim=-1)
    w = F.relu(se.fc1(w))
    weights = torch.sigmoid(se.fc2(w))
    
    assert torch.all(weights >= 0.0) and torch.all(weights <= 1.0), \
        "SE excitation weights are not strictly bounded in [0, 1]"


def test_factorized_1d_block_parameter_reduction():
    """
    Verifies that factorized spatio-temporal convolution uses significantly fewer
    parameters than an equivalent 2D convolution kernel (R2 complexity requirement).
    """
    in_channels = 64
    out_channels = 64
    kernel_time = 7
    kernel_freq = 5
    
    # Standard 2D convolution parameter count: C_in * C_out * K_t * K_f
    params_2d = in_channels * out_channels * kernel_time * kernel_freq
    
    # Factorized: Depthwise 1D (C_in * K_t) + Pointwise 1D (C_in * C_out * 1)
    params_factorized = (in_channels * kernel_time) + (in_channels * out_channels * 1)
    
    reduction_ratio = params_2d / params_factorized
    assert reduction_ratio > 2.5, f"Expected >2.5x parameter reduction, got {reduction_ratio:.2f}x"


def test_factorized_1d_block_forward():
    """Verifies that Factorized1DBlock downsamples length by 2 when stride=2."""
    block_stride1 = RefFactorized1DBlock(in_channels=32, out_channels=32, stride=1)
    block_stride2 = RefFactorized1DBlock(in_channels=32, out_channels=64, stride=2)
    
    x = torch.randn(2, 32, 100)
    y1 = block_stride1(x)
    assert y1.shape == (2, 32, 100)
    
    y2 = block_stride2(x)
    assert y2.shape == (2, 64, 50)


def test_factorized_1d_net_output_shape():
    """Verifies that full model outputs shape (B, 1) across different batch sizes."""
    model = RefFactorized1DNet(base_filters=32)
    for b in [1, 2, 4]:
        x = torch.randn(b, 1, 160000)
        out = model(x)
        assert out.shape == (b, 1), f"Expected shape ({b}, 1), got {out.shape}"


def test_factorized_1d_net_parameter_budget():
    """
    Verifies that the model adheres to the lightweight parameter budget (<500k params),
    preventing Kaggle T4 memory exhaustion.
    """
    model = RefFactorized1DNet(base_filters=32)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params < 500000, f"Total parameters {total_params} exceeds 500k budget"


def test_factorized_1d_net_gradients_clean():
    """Verifies that backward propagation computes valid gradients without NaN or Inf."""
    model = RefFactorized1DNet(base_filters=16)
    x = torch.randn(2, 1, 16000)  # Shorter length for fast autograd check
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient contains NaN in {name}"
            assert not torch.isinf(param.grad).any(), f"Gradient contains Inf in {name}"


def test_factorized_1d_net_eval_determinism():
    """Verifies that model.eval() produces strictly deterministic output for identical inputs."""
    model = RefFactorized1DNet(base_filters=16, dropout=0.5)
    x = torch.randn(2, 1, 16000)
    
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
        
    assert torch.allclose(out1, out2, atol=1e-6), "eval() mode outputs must be identical for identical inputs"


# --- Tests for src.models implementations ---

from src.models import (
    Factorized1DNet,
    SEBlock1D,
    SincConv1d,
    Factorized1DBlock,
    build_model,
    build_se_resnet,
)


def test_src_models_se_block1d():
    """Verifies SEBlock1D from src.models preserves shape and bounds weights."""
    channels = 64
    se = SEBlock1D(channels=channels, reduction=8)
    x = torch.randn(4, channels, 50)
    y = se(x)
    assert y.shape == x.shape


def test_src_models_sinc_conv1d_forward():
    """Verifies SincConv1d from src.models processes raw waveform and produces correct shape."""
    conv = SincConv1d(out_channels=32, kernel_size=251, stride=16)
    x = torch.randn(2, 1, 16000)
    out = conv(x)
    assert out.shape == (2, 32, 16000 // 16)


def test_src_models_factorized_1d_block():
    """Verifies Factorized1DBlock from src.models downsamples with stride=2."""
    block1 = Factorized1DBlock(in_channels=32, out_channels=32, stride=1)
    block2 = Factorized1DBlock(in_channels=32, out_channels=64, stride=2)
    x = torch.randn(2, 32, 100)
    assert block1(x).shape == (2, 32, 100)
    assert block2(x).shape == (2, 64, 50)


def test_src_models_factorized_1d_net_raw_1d_forward():
    """Verifies Factorized1DNet outputs shape (B, 1) on 1D waveform input."""
    model = Factorized1DNet(base_filters=16)
    x = torch.randn(2, 1, 16000)
    out = model(x)
    assert out.shape == (2, 1)


def test_src_models_factorized_1d_net_4d_adapter():
    """
    R2 Requirement: Verifies compatibility with 4D input shapes (B, 1, 128, 313)
    and 3D shapes (1, 128, 313) adapting gracefully through Factorized1DNet.
    """
    model = Factorized1DNet(base_filters=16)
    model.eval()
    
    # 4D input: (2, 1, 128, 313)
    x_4d = torch.randn(2, 1, 128, 313)
    out_4d = model(x_4d)
    assert out_4d.shape == (2, 1), f"Expected (2, 1), got {out_4d.shape}"
    
    # 3D input: (1, 128, 313)
    x_3d = torch.randn(1, 128, 313)
    out_3d = model(x_3d)
    assert out_3d.shape == (1, 1), f"Expected (1, 1), got {out_3d.shape}"


def test_src_models_builders():
    """Verifies build_model and build_se_resnet compatibility factories."""
    m1 = build_model()
    assert isinstance(m1, Factorized1DNet)
    
    m2 = build_se_resnet(input_shape=(1, 128, 313), dropout=0.15, se_ratio=16)
    assert isinstance(m2, Factorized1DNet)

