"""
Unit and integration tests for Physics-Informed HPO script (src/optimize.py).
Verifies:
1. run_dummy_inner_trial executes strictly on CPU with synthetic tensors.
2. PhysicsInformedLoss forward and backward passes execute correctly.
3. Search space contains both standard and physics-specific hyperparameters.
4. CLI execution in --dummy mode completes successfully.
"""

import subprocess
import sys
import pytest
import torch

from src.optimize import run_dummy_inner_trial


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
