# factorized 1d spatio temporal neural network architectures

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config

# 1d squeeze and excitation channel attention block
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        bottleneck = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, bottleneck)
        self.fc2 = nn.Linear(bottleneck, channels)
        
    def forward(self, x):
        w = x.mean(dim=-1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)

# alias for squeeze and excite block
SqueezeExcite1D = SEBlock1D
RefSqueezeExcite1d = SEBlock1D

# learnable sincnet bandpass filterbank frontend
class SincConv1d(nn.Module):
    def __init__(self, out_channels=64, kernel_size=251, sample_rate=16000, stride=16):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        
        hz_low = 30.0
        hz_high = sample_rate / 2.0 - 100.0
        
        # Mel-scale filter distribution focusing resolution in vehicle harmonic range
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
            
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
            
        mel_low = hz_to_mel(hz_low)
        mel_high = hz_to_mel(hz_high)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        f_init = mel_to_hz(mel_points)
        
        # initialize filter parameters
        self.f1 = nn.Parameter(torch.tensor(f_init[:-1], dtype=torch.float32) / sample_rate)
        self.band = nn.Parameter(torch.tensor(np.diff(f_init), dtype=torch.float32) / sample_rate)
        
        # precompute time grid and hamming window
        t = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer('t', t)
        
        w = 0.54 - 0.46 * torch.cos(2.0 * np.pi * torch.arange(kernel_size, dtype=torch.float32) / (kernel_size - 1))
        self.register_buffer('window', w)
        
    def forward(self, x):
        # Smooth parameter clamping/bounding without zero gradients using straight-through estimator
        f1_bounded = torch.clamp(self.f1, min=1e-4, max=0.49)
        f1_clamped = self.f1 + (f1_bounded - self.f1).detach()
        
        max_band = 0.5 - f1_clamped
        band_abs = torch.abs(self.band)
        band_bounded = torch.clamp(band_abs, min=torch.full_like(self.band, 1e-4), max=max_band)
        band_clamped = band_abs + (band_bounded - band_abs).detach()
        f2_clamped = f1_clamped + band_clamped
        
        device_type = x.device.type if x.device.type in ('cuda', 'cpu') else 'cpu'
        # Evaluate filter synthesis in float32 under AMP to prevent precision cancellation
        with torch.amp.autocast(device_type=device_type, enabled=False):
            t_mat = self.t.unsqueeze(0).to(device=x.device, dtype=torch.float32)
            f1_mat = f1_clamped.unsqueeze(1).to(dtype=torch.float32)
            f2_mat = f2_clamped.unsqueeze(1).to(dtype=torch.float32)
            window = self.window.unsqueeze(0).to(device=x.device, dtype=torch.float32)
            
            h2 = 2.0 * f2_mat * torch.sinc(2.0 * f2_mat * t_mat)
            h1 = 2.0 * f1_mat * torch.sinc(2.0 * f1_mat * t_mat)
            filters = (h2 - h1) * window
            
            # L2 filter normalization
            filters = filters / (torch.norm(filters, dim=-1, keepdim=True) + 1e-6)
            filters = filters.unsqueeze(1)
            
        filters = filters.to(dtype=x.dtype)
        return F.conv1d(x, filters, stride=self.stride, padding=self.kernel_size // 2)

# alias for sinc conv layer
RefSincConv1d = SincConv1d

# factorized 1d residual convolution block
class Factorized1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, se_reduction=8):
        super().__init__()
        # temporal depthwise convolution
        self.conv_time = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, groups=in_channels, bias=False
        )
        # frequency and channel projection
        self.conv_point = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        
        # second temporal convolution: depthwise-separable (reduces parameter bloat)
        self.conv2_time = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=out_channels, bias=False
        )
        self.conv2_point = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels, reduction=se_reduction)
        self.act2 = nn.GELU()
        
        # residual shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        res = self.shortcut(x)
        y = self.act1(self.bn1(self.conv_point(self.conv_time(x))))
        y = self.se(self.bn2(self.conv2_point(self.conv2_time(y))))
        return self.act2(y + res)

# alias for factorized block
RefFactorized1DBlock = Factorized1DBlock

# factorized 1d spatio temporal neural network
class Factorized1DNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=64, se_reduction=8, dropout=0.2, kernel_size=7):
        super().__init__()
        self.base_filters = base_filters
        
        # frontend feature extraction
        self.frontend = SincConv1d(out_channels=base_filters, kernel_size=Config.SINC_KERNEL_SIZE, stride=Config.SINC_STRIDE)
        self.pool0 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # stage 1 residual blocks
        self.stage1 = nn.Sequential(
            Factorized1DBlock(base_filters, base_filters, kernel_size=kernel_size, stride=1, se_reduction=se_reduction),
            Factorized1DBlock(base_filters, base_filters, kernel_size=kernel_size, stride=1, se_reduction=se_reduction)
        )
        # stage 2 residual blocks
        self.stage2 = nn.Sequential(
            Factorized1DBlock(base_filters, base_filters * 2, kernel_size=kernel_size, stride=2, se_reduction=se_reduction),
            Factorized1DBlock(base_filters * 2, base_filters * 2, kernel_size=kernel_size, stride=1, se_reduction=se_reduction)
        )
        # stage 3 residual blocks
        self.stage3 = nn.Sequential(
            Factorized1DBlock(base_filters * 2, base_filters * 4, kernel_size=kernel_size, stride=2, se_reduction=se_reduction),
            Factorized1DBlock(base_filters * 4, base_filters * 4, kernel_size=kernel_size, stride=1, se_reduction=se_reduction)
        )
        
        # Sequence modeling head for temporal Doppler tracking
        rnn_hidden = 128
        self.rnn = nn.GRU(
            input_size=base_filters * 4,  # 256
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Temporal attention pooling mechanism (bidirectional output: rnn_hidden * 2 = 256)
        self.attn = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Multi-pooling projection head (h_att: 256 + h_avg: 256 + h_max: 256 = 768)
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden * 6, 128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )
        
        # initialize regression bias to global mean to avoid dead epochs
        nn.init.constant_(self.head[-1].bias, 60.0)
        
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # extract phase preserving temporal representations
        h = self.frontend(x)
        h = torch.abs(h)
        h = self.pool0(h)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        
        # h is (batch, channels, time) -> permute for GRU (batch, time, channels)
        h = h.permute(0, 2, 1)
        
        # pass through bidirectional GRU
        self.rnn.flatten_parameters()
        out, _ = self.rnn(h)  # (batch, time, 256)
        
        # Temporal Attention Pooling
        attn_scores = self.attn(out)  # (batch, time, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, time, 1)
        h_att = torch.sum(out * attn_weights, dim=1)  # (batch, 256)
        
        # Global Statistics Pooling
        h_avg = out.mean(dim=1)  # (batch, 256)
        h_max = out.max(dim=1)[0]  # (batch, 256)
        
        # Concatenate multi-pooled representations (batch, 768)
        h_pooled = torch.cat([h_att, h_avg, h_max], dim=-1)
        
        return self.head(h_pooled)

# aliases for network
Factorized1DSENet = Factorized1DNet
RefFactorized1DNet = Factorized1DNet

# builder helper for speed estimator
def build_model(config=Config):
    return Factorized1DNet(
        in_channels=config.IN_CHANNELS,
        base_filters=config.BASE_FILTERS,
        se_reduction=config.SE_REDUCTION,
        dropout=config.DROPOUT,
        kernel_size=config.KERNEL_SIZE_TIME
    )

# compatibility builder for baseline interface
def build_se_resnet(input_shape=None, config=Config):
    base_f = config.BASE_FILTERS if hasattr(config, 'BASE_FILTERS') else 64
    return Factorized1DNet(base_filters=base_f)
