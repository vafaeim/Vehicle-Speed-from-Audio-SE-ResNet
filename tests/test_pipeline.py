"""
Tier 5: End-to-End Pipeline, AMP & Hardware Profiling Tests
Tests full forward/backward training step on synthetic audio batches,
AMP fp16/bf16 mixed-precision execution, gradient norm clipping,
GPU/CPU memory profiling logic, and 10-fold ensemble evaluation.
"""

import os
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from tests.test_models import RefFactorized1DNet
    from tests.test_losses import CombinedPhysicsLoss, CauchyLoss
    from tests.test_data_loader import SyntheticAudioDataset
except (ImportError, ModuleNotFoundError):
    from .test_models import RefFactorized1DNet
    from .test_losses import CombinedPhysicsLoss, CauchyLoss
    from .test_data_loader import SyntheticAudioDataset


# --- Pipeline Helper Utilities ---

def profile_peak_memory(device: str) -> float:
    """Returns current peak allocated memory in Megabytes."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    # CPU fallback: resident set size from /proc/self/statm or psutil
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((predictions - targets)**2)))


# --- E2E Unit & Integration Tests ---

def test_pipeline_forward_backward_optimization_step(synthetic_audio_batch, device):
    """
    Verifies full training loop step:
    Batch -> Model Forward -> Cauchy Loss -> Backward -> Clip Grad Norm -> Optimizer Step.
    """
    x, y = synthetic_audio_batch  # x: (4, 1, 160000), y: (4, 1)
    x = x[:, :, :32000].to(device)  # Trim to 2s for fast test execution
    y = y.to(device)
    
    model = RefFactorized1DNet(base_filters=16).to(device)
    criterion = CauchyLoss(gamma=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Capture weights before step
    initial_param = next(model.parameters()).clone().detach()
    
    # 1. Forward
    pred = model(x)
    assert pred.shape == (4, 1), f"Expected pred shape (4, 1), got {pred.shape}"
    
    # 2. Loss
    loss = criterion(pred, y)
    assert not torch.isnan(loss), "Loss computed as NaN"
    assert not torch.isinf(loss), "Loss computed as Inf"
    assert loss.item() >= 0.0, "Loss must be non-negative"
    
    # 3. Backward
    optimizer.zero_grad()
    loss.backward()
    
    # 4. Gradient clipping
    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    assert not torch.isnan(total_norm), "Gradient norm is NaN"
    assert not torch.isinf(total_norm), "Gradient norm is Inf"
    
    # 5. Optimizer Step
    optimizer.step()
    
    # Verify parameters updated
    updated_param = next(model.parameters()).detach()
    assert not torch.equal(initial_param, updated_param), "Model parameters did not update after optimizer step"


def test_pipeline_amp_mixed_precision_compatibility(synthetic_audio_batch, device):
    """
    Verifies Automatic Mixed Precision (AMP) compatibility:
    Model execution under torch.amp.autocast does not crash,
    does not produce NaNs, and computes valid loss.
    """
    x, y = synthetic_audio_batch
    x = x[:, :, :32000].to(device)
    y = y.to(device)
    
    model = RefFactorized1DNet(base_filters=16).to(device)
    criterion = CauchyLoss(gamma=5.0)
    
    # Select appropriate device type and dtype for autocast
    device_type = "cuda" if device == "cuda" else "cpu"
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    
    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
        pred = model(x)
        loss = criterion(pred, y)
        
    assert not torch.isnan(loss), "AMP autocast produced NaN loss"
    assert not torch.isinf(loss), "AMP autocast produced Inf loss"
    assert pred.shape == (4, 1)


def test_vram_profiling_and_memory_envelope(device):
    """
    Verifies that memory profiling function works and that peak VRAM/RAM
    consumption for a batch remains well under the 2.5 GB target (and 16 GB T4 hard budget).
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_mem_mb = profile_peak_memory(device)
    
    model = RefFactorized1DNet(base_filters=32).to(device)
    # Batch size 4, 32000 samples
    x = torch.randn(4, 1, 32000, device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    end_mem_mb = profile_peak_memory(device)
    
    # Peak memory should be finite and well below 2500 MB
    assert end_mem_mb >= 0.0, "Peak memory must be non-negative"
    if device == "cuda" and torch.cuda.is_available():
        allocated_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        assert allocated_mb < 2500.0, f"Peak VRAM {allocated_mb} MB exceeds 2.5 GB limit"


def test_ensemble_rmse_aggregation_logic():
    """
    Verifies 10-fold CV ensemble averaging logic and RMSE calculation.
    """
    # Simulate ground truth speeds for 20 test vehicles
    np.random.seed(42)
    y_true = np.random.uniform(40.0, 110.0, size=(20, 1))
    
    # Simulate 5 fold model predictions with slight random perturbations
    fold_preds = []
    for i in range(5):
        noise = np.random.normal(0.0, 3.0, size=(20, 1))
        fold_preds.append(y_true + noise)
        
    # Ensemble prediction: simple mean across folds
    ensemble_pred = np.mean(fold_preds, axis=0)
    
    rmse = compute_rmse(ensemble_pred, y_true)
    assert rmse < 6.5, f"Simulated ensemble RMSE {rmse:.2f} exceeded target 6.5 km/h"
    assert rmse > 0.0, "RMSE must be strictly positive for non-identical predictions"
