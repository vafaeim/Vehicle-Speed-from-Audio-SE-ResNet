"""
Asynchronous Dual-GPU Hyperparameter Optimization Engine with Nested Cross-Validation (Nested CV)
for Physics-Informed Factorized 1D-SE Vehicle Speed Estimation.

Framework: Optuna with Tree-structured Parzen Estimator (TPE) sampler.
Storage Backend: SQLite database (default: sqlite:///optuna_study.db?timeout=60).
Multiprocessing Architecture:
    - Spawns 2 independent worker processes using the 'spawn' start method.
    - Worker 1 is mapped to 'cuda:0' and Worker 2 is mapped to 'cuda:1' (without DistributedDataParallel).
    - Asynchronous execution eliminates all-reduce synchronization bottlenecks across dual GPUs.
Architecture:
    - Updated Factorized 1D-SE network (Factorized1DNet / Factorized1DSENet).
    - Operates directly on raw 1D audio waveforms via learnable SincConv1d filterbank frontend.
Loss Formulation:
    - PhysicsInformedLoss (Lorentzian Cauchy M-estimator + Kinematic Acceleration regularizer + Domain Boundary).
Search Space:
    - Standard Hyperparameters:
        * Learning Rate: log-uniform [1e-5, 5e-3]
        * Weight Decay: log-uniform [1e-6, 1e-2]
        * SE Reduction Ratio (r): categorical {8, 16, 32}
        * Dropout: uniform [0.10, 0.50]
    - Physics-Informed Parameters:
        * Physics Loss Weight (kinematic penalty): log-uniform [0.01, 0.50]
        * Cauchy Gamma (Lorentzian scale): uniform [2.0, 10.0]
        * Bound Weight (speed domain penalty): log-uniform [0.01, 0.20]
Performance Optimization Note:
    - The factorized 1D-SE architecture operates directly on time-domain audio waveforms
      via SincConv1d. It does NOT compute Mel-spectrograms on the fly.
    - In-memory waveform caching is provided natively by VS13Dataset (use_cache=True).
    - Redundant Mel-spectrogram caching is therefore bypassed.
Fast Dummy Verification Interface:
    - '--dummy' CLI flag strictly forces CPU execution on synthetically generated random tensors
      (batch_size=2, raw audio shape (2, 1, 32000), labels (2, 1)), running 1 trial per worker.
    - Strictly verifies that PhysicsInformedLoss is computed correctly during the forward
      and backward optimization passes without requiring CUDA or disk I/O.
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repository root is in sys.path for direct CLI execution and spawned processes
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from torch.utils.data import DataLoader

from src.config import Config
try:
    from src.models import Factorized1DNet, Factorized1DSENet
except ImportError:
    from models import Factorized1DNet, Factorized1DSENet

try:
    from src.losses import PhysicsInformedLoss
except ImportError:
    from losses import PhysicsInformedLoss

from src.data_loader import VS13Dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("optimize_physics")


def normalize_storage_url(storage_url: str) -> str:
    """
    Ensures SQLite URLs specify an explicit busy timeout to prevent database locks
    under concurrent multi-process writes.
    """
    if storage_url.startswith("sqlite://") and "timeout=" not in storage_url:
        delimiter = "&" if "?" in storage_url else "?"
        return f"{storage_url}{delimiter}timeout=60"
    return storage_url


def run_dummy_inner_trial(
    device: torch.device,
    lr: float,
    weight_decay: float,
    se_ratio: int,
    dropout: float,
    physics_weight: float,
    cauchy_gamma: float,
    bound_weight: float,
    base_filters: int = 16,
    audio_length: int = 4000,
) -> float:
    """
    Executes a fast synthetic dummy training and evaluation step strictly on CPU.
    Validates end-to-end Factorized 1D-SE model forward pass, PhysicsInformedLoss computation
    (both regression, boundary, and kinematic components), backward gradient flow,
    optimizer step, and RMSE metric computation.
    """
    # 1. Instantiate the updated Factorized 1D-SE architecture
    model = Factorized1DNet(
        in_channels=1,
        base_filters=base_filters,
        se_reduction=se_ratio,
        dropout=dropout,
        kernel_size=Config.KERNEL_SIZE_TIME,
    ).to(device)

    # 2. Instantiate the PhysicsInformedLoss with sampled hyperparameters
    criterion = PhysicsInformedLoss(
        gamma=cauchy_gamma,
        physics_weight=physics_weight,
        bound_weight=bound_weight,
        max_accel=Config.MAX_ACCELERATION,
        loss_type="cauchy",
        huber_delta=getattr(Config, "HUBER_DELTA", 10.0),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Synthetic batch of size 2: raw 1D audio (2, 1, audio_length) and target speeds (2, 1)
    x_train = torch.randn(2, 1, audio_length, device=device)
    y_train = torch.tensor([[50.0], [75.0]], dtype=torch.float32, device=device)

    x_val = torch.randn(2, 1, audio_length, device=device)
    y_val = torch.tensor([[52.0], [72.0]], dtype=torch.float32, device=device)

    # 3. Training step (1 epoch)
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred = model(x_train)
    assert pred.shape == (2, 1), f"Expected pred shape (2, 1), got {pred.shape}"

    # Forward loss computation with PhysicsInformedLoss
    loss = criterion(pred, y_train)
    assert not torch.isnan(loss), "PhysicsInformedLoss returned NaN"
    assert not torch.isinf(loss), "PhysicsInformedLoss returned Inf"
    assert loss.item() >= 0.0, f"PhysicsInformedLoss must be non-negative, got {loss.item()}"

    # Verify kinematic trajectory regularizer component as well
    # Acceleration threshold is Config.MAX_ACCELERATION (30.0 km/h/s); exceed by +15.0 to activate penalty
    speed_seq = torch.cat([pred, pred + (Config.MAX_ACCELERATION + 15.0)], dim=-1)  # (2, 2) sequential frames
    loss_with_seq = criterion(pred, y_train, speed_seq=speed_seq)
    assert not torch.isnan(loss_with_seq) and not torch.isinf(loss_with_seq), (
        "Kinematic penalty produced NaN/Inf in PhysicsInformedLoss"
    )
    assert loss_with_seq.item() > loss.item(), (
        f"Kinematic acceleration penalty did not increase total loss (loss={loss.item()}, loss_with_seq={loss_with_seq.item()})"
    )

    # Backward pass on full composite physics-informed loss (including kinematic penalty)
    loss_with_seq.backward()

    # Verify that gradients were successfully computed for parameters
    has_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
    assert has_grad, "Backward pass failed to propagate gradients through Factorized 1D-SE network"

    # Optimizer step
    optimizer.step()

    # 4. Validation evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = criterion(val_pred, y_val)
        val_rmse = float(torch.sqrt(torch.mean((val_pred - y_val) ** 2)).item())

    del model, criterion, optimizer, pred, loss, loss_with_seq, x_train, y_train, x_val, y_val, val_pred, val_loss
    return val_rmse


def run_nested_inner_cv(
    device: torch.device,
    lr: float,
    weight_decay: float,
    se_ratio: int,
    dropout: float,
    physics_weight: float,
    cauchy_gamma: float,
    bound_weight: float,
    data_dir: Optional[str],
    outer_fold_idx: int = 0,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    epochs: int = 20,
    batch_size: int = 16,
    base_filters: int = 32,
) -> float:
    """
    Executes Nested CV inner loop on the VS13 dataset with Factorized 1D-SE and PhysicsInformedLoss:
    1. Extracts outer training partition (preventing leakage).
    2. Performs K_inner = 3 fold CV on outer training data.
    3. Returns mean inner validation RMSE across the folds.
    """
    from src.utils import get_all_audio_paths_and_labels

    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(f"VS13 dataset directory not found: {data_dir}")

    all_paths, all_speeds = get_all_audio_paths_and_labels(data_dir)
    if len(all_paths) == 0:
        raise ValueError(f"No audio files found in {data_dir}")

    # Outer split
    outer_kfold = KFold(n_splits=n_outer_folds, shuffle=True, random_state=Config.SEED)
    splits = list(outer_kfold.split(all_paths))
    outer_train_indices, _ = splits[outer_fold_idx]

    outer_train_paths = [all_paths[i] for i in outer_train_indices]
    outer_train_speeds = all_speeds[outer_train_indices]

    # Inner split
    inner_kfold = KFold(n_splits=n_inner_folds, shuffle=True, random_state=Config.SEED)
    inner_rmses: List[float] = []

    for inner_fold, (in_train_idx, in_val_idx) in enumerate(inner_kfold.split(outer_train_paths)):
        train_paths = [outer_train_paths[i] for i in in_train_idx]
        train_speeds = outer_train_speeds[in_train_idx]
        val_paths = [outer_train_paths[i] for i in in_val_idx]
        val_speeds = outer_train_speeds[in_val_idx]

        logger.info(
            f"  [Outer {outer_fold_idx+1}/{n_outer_folds}] Inner Fold {inner_fold+1}/{n_inner_folds}: "
            f"train={len(train_paths)}, val={len(val_paths)}"
        )

        # VS13Dataset caches raw waveforms in memory (use_cache=True)
        train_ds = VS13Dataset(train_paths, train_speeds, is_training=True, use_cache=True)
        val_ds = VS13Dataset(val_paths, val_speeds, is_training=False, use_cache=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = Factorized1DNet(
            in_channels=1,
            base_filters=base_filters,
            se_reduction=se_ratio,
            dropout=dropout,
            kernel_size=Config.KERNEL_SIZE_TIME,
        ).to(device)

        criterion = PhysicsInformedLoss(
            gamma=cauchy_gamma,
            physics_weight=physics_weight,
            bound_weight=bound_weight,
            max_accel=Config.MAX_ACCELERATION,
            loss_type="cauchy",
            huber_delta=getattr(Config, "HUBER_DELTA", 10.0),
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

        best_val_rmse = float("inf")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                pred = model(x_b)
                loss = criterion(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.CLIP_GRAD_NORM)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()

            # Validation evaluation
            model.eval()
            val_sq_errors = []
            with torch.no_grad():
                for x_v, y_v in val_loader:
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    v_pred = model(x_v)
                    val_sq_errors.extend((v_pred - y_v).pow(2).cpu().numpy().tolist())

            epoch_val_rmse = float(np.sqrt(np.mean(val_sq_errors))) if val_sq_errors else float("inf")
            if epoch_val_rmse < best_val_rmse:
                best_val_rmse = epoch_val_rmse

            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(
                    f"    Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | "
                    f"val_RMSE={epoch_val_rmse:.2f} | best={best_val_rmse:.2f}"
                )

        inner_rmses.append(best_val_rmse)
        logger.info(f"  Inner Fold {inner_fold+1}/{n_inner_folds} done -> best RMSE: {best_val_rmse:.2f} km/h")

    return float(np.mean(inner_rmses))


def run_worker(
    gpu_id: str,
    study_name: str = "physics_informed_hpo",
    storage_url: str = "sqlite:///optuna_study.db?timeout=60",
    n_trials: int = 1,
    is_dummy: bool = False,
    outer_fold: int = 0,
    outer_folds: int = 5,
    inner_folds: int = 3,
    epochs: int = 20,
    batch_size: int = 16,
    data_dir: Optional[str] = None,
    base_filters: int = 32,
):
    """
    Multiprocessing Worker Entry Point.

    Binds worker to target GPU ('cuda:0' or 'cuda:1', or strictly 'cpu' in dummy mode).
    Connects to the persistent shared SQLite study and runs Optuna TPE optimization.
    """
    storage_url = normalize_storage_url(storage_url)

    if is_dummy:
        # STRICTLY force 'cpu' device; never access CUDA
        device = torch.device("cpu")
        torch.set_num_threads(1)
        if hasattr(torch.backends, "nnpack"):
            torch.backends.nnpack.enabled = False
        print(f"[{gpu_id}] Dummy Mode active: strictly bound to CPU device.")
    else:
        # Bind explicitly to target GPU with graceful fallback for single-GPU systems
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            ordinal = int(gpu_id.split(":")[-1]) if ":" in str(gpu_id) else 0
            if ordinal < num_gpus:
                torch.cuda.set_device(ordinal)
                device = torch.device(f"cuda:{ordinal}")
                print(f"[{gpu_id}] Successfully bound to GPU device: {device}")
            else:
                fallback_ordinal = 0
                torch.cuda.set_device(fallback_ordinal)
                device = torch.device(f"cuda:{fallback_ordinal}")
                print(
                    f"[{gpu_id}] Requested {gpu_id} but only {num_gpus} GPU(s) available; "
                    f"safely sharing cuda:{fallback_ordinal}."
                )
        else:
            device = torch.device("cpu")
            torch.set_num_threads(1)
            if hasattr(torch.backends, "nnpack"):
                torch.backends.nnpack.enabled = False
            print(f"[{gpu_id}] CUDA unavailable; operating on CPU device.")

    # Attach to shared persistent Optuna study
    # Seed sampler independently per worker in dummy mode to ensure non-redundant exploration
    if is_dummy:
        worker_id = int(gpu_id.split(":")[-1]) if ":" in str(gpu_id) else 0
        sampler = TPESampler(seed=42 + worker_id)
    else:
        sampler = TPESampler(seed=None)

    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        # 1. Standard Hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        se_ratio = trial.suggest_categorical("se_ratio", [8, 16, 32])
        dropout = trial.suggest_float("dropout", 0.10, 0.50, step=0.05)

        # 2. Physics-Informed Parameters
        physics_weight = trial.suggest_float("physics_weight", 0.01, 0.50, log=True)
        cauchy_gamma = trial.suggest_float("cauchy_gamma", 2.0, 10.0)
        bound_weight = trial.suggest_float("bound_weight", 0.01, 0.20, log=True)

        if is_dummy:
            val_rmse = run_dummy_inner_trial(
                device=device,
                lr=lr,
                weight_decay=weight_decay,
                se_ratio=se_ratio,
                dropout=dropout,
                physics_weight=physics_weight,
                cauchy_gamma=cauchy_gamma,
                bound_weight=bound_weight,
                base_filters=base_filters,
            )
        else:
            val_rmse = run_nested_inner_cv(
                device=device,
                lr=lr,
                weight_decay=weight_decay,
                se_ratio=se_ratio,
                dropout=dropout,
                physics_weight=physics_weight,
                cauchy_gamma=cauchy_gamma,
                bound_weight=bound_weight,
                data_dir=data_dir,
                outer_fold_idx=outer_fold,
                n_outer_folds=outer_folds,
                n_inner_folds=inner_folds,
                epochs=epochs,
                batch_size=batch_size,
                base_filters=base_filters,
            )

        return val_rmse

    study.optimize(objective, n_trials=n_trials)
    print(f"[{gpu_id}] Completed {n_trials} trial(s) in study '{study_name}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Asynchronous Dual-GPU Optuna HPO for Physics-Informed Factorized 1D-SE Pipeline"
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Run fast synthetic CPU verification without touching CUDA or disk (Acceptance Criteria)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Total optimization trials to run per worker process",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_study.db?timeout=60",
        help="Optuna persistent storage URL (SQLite with timeout)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="physics_informed_hpo",
        help="Optuna study name",
    )
    parser.add_argument(
        "--outer_folds",
        type=int,
        default=5,
        help="Number of outer folds for nested CV generalization evaluation",
    )
    parser.add_argument(
        "--inner_folds",
        type=int,
        default=3,
        help="Number of inner folds for hyperparameter selection",
    )
    parser.add_argument(
        "--outer_fold_idx",
        type=int,
        default=0,
        help="Outer fold index to execute HPO on (0 to outer_folds - 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per fold (overridden to 1 in dummy mode)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (overridden to 2 in dummy mode)",
    )
    parser.add_argument(
        "--base_filters",
        type=int,
        default=32,
        help="Base filters for Factorized1DNet (default: 32)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to VS13 dataset root directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # In dummy mode: override settings for rapid (<5s) CPU execution
    if args.dummy:
        n_trials = args.n_trials if args.n_trials != 20 else 1
        epochs = 1
        batch_size = 2
        base_filters = 16
        print("=" * 70)
        print(" FAST DUMMY VERIFICATION MODE ACTIVE (Physics-Informed Pipeline)")
        print("  - Device: STRICTLY FORCED TO 'cpu'")
        print("  - Model: Factorized 1D-SE (Factorized1DNet)")
        print("  - Loss: PhysicsInformedLoss (Cauchy + Kinematic + Boundary)")
        print("  - Data: Synthetically generated random tensors (batch_size=2)")
        print(f"  - Trials: {n_trials} per worker process ({n_trials * 2} total)")
        print(f"  - Storage Backend: {args.storage}")
        print("=" * 70)
    else:
        n_trials = args.n_trials
        epochs = args.epochs
        batch_size = args.batch_size
        base_filters = args.base_filters
        print("=" * 70)
        print(" ASYNCHRONOUS DUAL-GPU PHYSICS-INFORMED HPO INITIALIZED")
        print(f"  - Study Name: {args.study_name}")
        print(f"  - Storage: {args.storage}")
        print(f"  - Target GPUs: Worker 1 -> 'cuda:0', Worker 2 -> 'cuda:1'")
        print(f"  - Architecture: Factorized1DNet (base_filters={base_filters})")
        print(f"  - Loss: PhysicsInformedLoss")
        print(f"  - Nested CV: K_outer={args.outer_folds}, K_inner={args.inner_folds}")
        print("=" * 70)

    args.storage = normalize_storage_url(args.storage)

    # 1. Initialize persistent SQLite Optuna study in main process
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
    )

    # 2. EXACT Python multiprocessing code using 'spawn' start method
    mp.set_start_method("spawn", force=True)

    # 3. Instantiate two independent worker processes binding to 'cuda:0' and 'cuda:1'
    p1 = mp.Process(
        target=run_worker,
        args=("cuda:0", args.study_name, args.storage, n_trials, args.dummy),
        kwargs={
            "outer_fold": args.outer_fold_idx,
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "data_dir": args.data_dir,
            "base_filters": base_filters,
        },
    )
    p2 = mp.Process(
        target=run_worker,
        args=("cuda:1", args.study_name, args.storage, n_trials, args.dummy),
        kwargs={
            "outer_fold": args.outer_fold_idx,
            "outer_folds": args.outer_folds,
            "inner_folds": args.inner_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "data_dir": args.data_dir,
            "base_filters": base_filters,
        },
    )

    print("[Main] Launching asynchronous Worker 1 ('cuda:0') and Worker 2 ('cuda:1')...")
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    if p1.exitcode != 0 or p2.exitcode != 0:
        print(f"[Main] ERROR: Worker failure detected (p1={p1.exitcode}, p2={p2.exitcode})")
        sys.exit(1)

    # 4. Load resulting study and verify SQLite persistence
    updated_study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    completed_trials = [
        t for t in updated_study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    print("\n" + "=" * 70)
    print(" PHYSICS-INFORMED HPO EXECUTION COMPLETED SUCCESSFULLY")
    print(f"  - Total Trials Recorded in SQLite: {len(updated_study.trials)}")
    print(f"  - Completed Trials: {len(completed_trials)}")
    if len(completed_trials) > 0:
        print(f"  - Best Trial Objective (RMSE): {updated_study.best_value:.4f} km/h")
        print(f"  - Best Hyperparameters Discovered:")
        for k, v in updated_study.best_params.items():
            print(f"      * {k}: {v}")
    elif len(updated_study.trials) > 0:
        print("  - Warning: No trials completed successfully (all failed or pruned).")
    print("=" * 70)


if __name__ == "__main__":
    main()
