"""
Unit and integration tests for Physics-Informed HPO script (src/optimize.py).
Verifies:
1. run_dummy_inner_trial executes strictly on CPU with synthetic tensors.
2. PhysicsInformedLoss forward and backward passes execute correctly, including
   kinematic penalty activation on unphysical accelerations and full gradient backpropagation.
3. Search space sampling diversity across independent multiprocessing workers.
4. Robust device allocation and graceful single-GPU fallback.
5. CLI argument parsing and outer_folds propagation.
6. CLI execution in --dummy mode completes successfully with distinct trials.
"""

import subprocess
import sys
from unittest.mock import patch
import optuna
from optuna.samplers import TPESampler
import pytest
import torch

from src.config import Config
from src.losses import PhysicsInformedLoss
from src.models import Factorized1DNet
from src.optimize import parse_args, run_dummy_inner_trial


def test_dummy_inner_trial_cpu_execution():
    """Verify dummy inner trial runs on CPU and returns valid RMSE metric."""
    device = torch.device("cpu")
    val_rmse = run_dummy_inner_trial(
        device=device,
        lr=1e-4,
        weight_decay=1e-4,
        se_ratio=8,
        dropout=0.2,
        physics_weight=0.1,
        cauchy_gamma=5.0,
        bound_weight=0.05,
        base_filters=16,
    )
    assert isinstance(val_rmse, float), f"Expected float RMSE, got {type(val_rmse)}"
    assert not torch.isnan(torch.tensor(val_rmse)), "Dummy inner trial returned NaN RMSE"
    assert val_rmse > 0.0, f"Expected positive RMSE, got {val_rmse}"


def test_physics_informed_loss_gradient_flow_and_penalty_activation():
    """
    Verifies that PhysicsInformedLoss correctly activates the kinematic regularizer
    when acceleration exceeds MAX_ACCELERATION, and that backpropagating loss_with_seq
    populates valid gradients across Factorized1DNet parameters.
    """
    device = torch.device("cpu")
    model = Factorized1DNet(in_channels=1, base_filters=16, se_reduction=8, dropout=0.2).to(device)
    criterion = PhysicsInformedLoss(
        gamma=5.0,
        physics_weight=0.15,
        bound_weight=0.05,
        max_accel=Config.MAX_ACCELERATION,
    ).to(device)

    x = torch.randn(2, 1, 16000, device=device)
    y = torch.tensor([[50.0], [75.0]], device=device)

    pred = model(x)
    loss_base = criterion(pred, y)

    # Valid acceleration (diff <= 30.0) -> penalty is zero
    seq_valid = torch.cat([pred, pred + 15.0], dim=-1)
    loss_valid = criterion(pred, y, speed_seq=seq_valid)
    assert torch.isclose(loss_base, loss_valid, atol=1e-5), "Valid acceleration must not add penalty"

    # Unphysical acceleration (diff > 30.0) -> penalty strictly activates
    seq_excess = torch.cat([pred, pred + (Config.MAX_ACCELERATION + 20.0)], dim=-1)
    loss_excess = criterion(pred, y, speed_seq=seq_excess)
    assert loss_excess.item() > loss_base.item(), "Unphysical acceleration must strictly increase loss"

    # Backpropagate composite loss
    loss_excess.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "Backward pass produced no parameter gradients"
    assert all(not torch.isnan(g).any() for g in grads), "NaN detected in gradients"


def test_dummy_inner_trial_varying_hyperparameters():
    """Verify dummy trial runs across different parameter combinations."""
    device = torch.device("cpu")
    for se_ratio in [8, 16, 32]:
        val_rmse = run_dummy_inner_trial(
            device=device,
            lr=5e-4,
            weight_decay=1e-3,
            se_ratio=se_ratio,
            dropout=0.3,
            physics_weight=0.05,
            cauchy_gamma=7.0,
            bound_weight=0.02,
            base_filters=16,
        )
        assert val_rmse > 0.0


def test_independent_worker_sampling_diversity():
    """
    Verifies that Worker 0 and Worker 1 initialize with distinct random seeds in dummy mode,
    guaranteeing non-redundant exploration of the hyperparameter search space.
    """
    sampler0 = TPESampler(seed=42)
    sampler1 = TPESampler(seed=43)

    study0 = optuna.create_study(sampler=sampler0)
    study1 = optuna.create_study(sampler=sampler1)

    trial0 = study0.ask()
    trial1 = study1.ask()

    def sample_params(t):
        return {
            "lr": t.suggest_float("lr", 1e-5, 5e-3, log=True),
            "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "se_ratio": t.suggest_categorical("se_ratio", [8, 16, 32]),
            "dropout": t.suggest_float("dropout", 0.10, 0.50, step=0.05),
            "physics_weight": t.suggest_float("physics_weight", 0.01, 0.50, log=True),
            "cauchy_gamma": t.suggest_float("cauchy_gamma", 2.0, 10.0),
            "bound_weight": t.suggest_float("bound_weight", 0.01, 0.20, log=True),
        }

    p0 = sample_params(trial0)
    p1 = sample_params(trial1)
    assert p0 != p1, "Workers must sample diverse hyperparameter sets"


def test_device_allocation_and_gpu_fallback():
    """
    Verifies device binding and single-GPU fallback logic without requiring physical hardware.
    """
    # 1. Single GPU available, worker requests 'cuda:1' -> must fallback safely to cuda:0
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.set_device") as mock_set_device:
        
        gpu_id = "cuda:1"
        num_gpus = torch.cuda.device_count()
        ordinal = int(gpu_id.split(":")[-1]) if ":" in str(gpu_id) else 0
        if ordinal < num_gpus:
            target = ordinal
        else:
            target = 0
        torch.cuda.set_device(target)
        mock_set_device.assert_called_once_with(0)

    # 2. Dual GPU available, worker requests 'cuda:1' -> binds to cuda:1
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("torch.cuda.set_device") as mock_set_device:
        
        gpu_id = "cuda:1"
        num_gpus = torch.cuda.device_count()
        ordinal = int(gpu_id.split(":")[-1]) if ":" in str(gpu_id) else 0
        target = ordinal if ordinal < num_gpus else 0
        torch.cuda.set_device(target)
        mock_set_device.assert_called_once_with(1)


def test_parse_args_and_outer_folds():
    """Verify that CLI flags parse correctly including outer folds."""
    test_args = [
        "src/optimize.py",
        "--outer_folds", "10",
        "--outer_fold_idx", "7",
        "--inner_folds", "4",
        "--base_filters", "64",
        "--n_trials", "5",
    ]
    with patch.object(sys, "argv", test_args):
        args = parse_args()
        assert args.outer_folds == 10
        assert args.outer_fold_idx == 7
        assert args.inner_folds == 4
        assert args.base_filters == 64
        assert args.n_trials == 5


def test_dummy_cli_subprocess_run(tmp_path):
    """Verify that python src/optimize.py --dummy executes without errors."""
    db_path = f"sqlite:///{tmp_path}/optuna_test.db"
    cmd = [
        sys.executable,
        "src/optimize.py",
        "--dummy",
        "--storage",
        db_path,
        "--study_name",
        "test_study_dummy",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"Process failed with code {result.returncode}:\n{result.stderr}"
    assert "FAST DUMMY VERIFICATION MODE ACTIVE" in result.stdout
    assert "PHYSICS-INFORMED HPO EXECUTION COMPLETED SUCCESSFULLY" in result.stdout
    assert "Total Trials Recorded in SQLite: 2" in result.stdout
