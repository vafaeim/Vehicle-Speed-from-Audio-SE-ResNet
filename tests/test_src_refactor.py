# integration tests verifying refactored src modules

import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import (
    set_seed,
    parse_speed_from_filename,
    discover_dataset_files,
    get_all_audio_paths_and_labels,
    compute_rmse,
    compute_mae,
    profile_peak_memory,
)
from src.data_loader import (
    pad_or_crop_audio,
    apply_gain_augmentation,
    apply_additive_noise,
    compute_cartesian_stft,
    compute_instantaneous_frequency,
    VS13Dataset,
    create_10fold_splits,
    get_vs13_datasets,
)
from src.models import (
    SEBlock1D,
    SincConv1d,
    Factorized1DBlock,
    Factorized1DNet,
    build_model,
    build_se_resnet,
)
from src.losses import (
    CauchyLoss,
    SmoothCauchyLoss,
    HuberLoss,
    KinematicAccelerationLoss,
    DomainBoundaryLoss,
    PhysicsInformedLoss,
    CombinedPhysicsLoss,
)
from src.train_engine import (
    train_one_epoch,
    evaluate,
    train_fold,
    evaluate_ensemble,
)

def test_config_attributes():
    assert Config.SAMPLE_RATE == 16000
    assert Config.DURATION_SECONDS == 10
    assert Config.AUDIO_LENGTH_SAMPLES == 160000
    assert Config.N_FFT == 2048
    assert Config.HOP_LENGTH == 512
    assert Config.N_MELS == 128
    assert Config.N_BINS == 1025
    assert Config.IN_CHANNELS == 1
    assert Config.BASE_FILTERS == 64
    assert Config.KERNEL_SIZE_TIME == 7
    assert Config.KERNEL_SIZE_FREQ == 5
    assert Config.SE_REDUCTION == 8
    assert Config.BATCH_SIZE == 32
    assert Config.LOSS_TYPE == "cauchy"
    assert Config.CAUCHY_GAMMA == 5.0
    assert Config.PHYSICS_WEIGHT == 0.1
    assert Config.MAX_ACCELERATION == 30.0

def test_utils_speed_parsing():
    assert parse_speed_from_filename("Mazda3_50.wav") == 50.0
    assert parse_speed_from_filename("CitroenC4Picasso_101.wav") == 101.0
    assert parse_speed_from_filename("Vehicle_35_1.wav") == 35.0
    assert parse_speed_from_filename("random_name.wav") is None

def test_utils_seed_and_metrics():
    set_seed(42)
    val1 = torch.rand(5)
    set_seed(42)
    val2 = torch.rand(5)
    assert torch.equal(val1, val2)

    pred = np.array([50.0, 60.0, 70.0])
    target = np.array([52.0, 58.0, 74.0])
    rmse = compute_rmse(pred, target)
    mae = compute_mae(pred, target)
    assert rmse > 0.0
    assert mae > 0.0

    mem = profile_peak_memory("cpu")
    assert mem >= 0.0

def test_data_loader_waveform_transforms():
    # pad and crop
    short = np.ones(1000, dtype=np.float32)
    padded = pad_or_crop_audio(short, target_length=2000)
    assert len(padded) == 2000
    assert np.all(padded[1000:] == 0.0)

    long_audio = np.ones(3000, dtype=np.float32)
    cropped = pad_or_crop_audio(long_audio, target_length=2000)
    assert len(cropped) == 2000

    # augmentations
    gained = apply_gain_augmentation(short, gain_db_range=(-3.0, 3.0))
    assert len(gained) == len(short)
    assert np.all(np.isfinite(gained))

    noisy = apply_additive_noise(short, snr_db_range=(15.0, 20.0))
    assert len(noisy) == len(short)
    assert np.all(np.isfinite(noisy))

def test_data_loader_phase_preserving_stft():
    wave = torch.randn(1, 16000)
    stft_cart = compute_cartesian_stft(wave, n_fft=512, hop_length=128)
    assert stft_cart.ndim == 4
    assert stft_cart.shape[1] == 2  # real and imag

    inst_freq = compute_instantaneous_frequency(wave, n_fft=512, hop_length=128, sample_rate=16000)
    assert inst_freq.ndim == 4
    assert inst_freq.shape[1] == 1

def test_models_components():
    # SEBlock1D
    se = SEBlock1D(channels=32, reduction=8)
    x = torch.randn(2, 32, 100)
    out = se(x)
    assert out.shape == x.shape

    # SincConv1d
    sinc = SincConv1d(out_channels=16, kernel_size=251, stride=16)
    x_raw = torch.randn(2, 1, 16000)
    y_sinc = sinc(x_raw)
    assert y_sinc.shape == (2, 16, 1000)

    # Factorized1DBlock
    block1 = Factorized1DBlock(in_channels=16, out_channels=16, kernel_size=7, stride=1)
    out1 = block1(y_sinc)
    assert out1.shape == (2, 16, 1000)

    block2 = Factorized1DBlock(in_channels=16, out_channels=32, kernel_size=7, stride=2)
    out2 = block2(y_sinc)
    assert out2.shape == (2, 32, 500)

    # Full Factorized1DNet
    model = Factorized1DNet(base_filters=16)
    out_speed = model(x_raw)
    assert out_speed.shape == (2, 1)

def test_losses_all():
    pred = torch.tensor([[50.0], [70.0]], requires_grad=True)
    target = torch.tensor([[52.0], [68.0]])
    
    cauchy = CauchyLoss(gamma=5.0)
    l_c = cauchy(pred, target)
    assert l_c.item() > 0.0

    huber = HuberLoss(delta=5.0)
    l_h = huber(pred, target)
    assert l_h.item() > 0.0

    kinematic = KinematicAccelerationLoss(max_accel=30.0, dt=1.0)
    valid_seq = torch.tensor([[50.0, 60.0, 75.0]])
    invalid_seq = torch.tensor([[50.0, 100.0]])
    assert kinematic(valid_seq).item() == 0.0
    assert kinematic(invalid_seq).item() > 0.0

    bound = DomainBoundaryLoss(speed_min=10.0, speed_max=140.0)
    assert bound(torch.tensor([50.0])).item() == 0.0
    assert bound(torch.tensor([5.0])).item() > 0.0

    combined = PhysicsInformedLoss(gamma=5.0, physics_weight=0.1)
    l_comb = combined(pred, target, speed_seq=invalid_seq)
    assert l_comb.item() > 0.0
    l_comb.backward()
    assert pred.grad is not None

def test_train_engine_components():
    model = Factorized1DNet(base_filters=8)
    criterion = PhysicsInformedLoss(gamma=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler(enabled=False)

    # create dummy dataloader
    x = torch.randn(4, 1, 16000)
    y = torch.tensor([[50.0], [60.0], [70.0], [80.0]])
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)

    loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device="cpu",
        use_amp=False
    )
    assert loss > 0.0

    eval_results = evaluate(model, dataloader, criterion, "cpu")
    assert "rmse" in eval_results
    assert "mae" in eval_results
    assert eval_results["rmse"] >= 0.0
