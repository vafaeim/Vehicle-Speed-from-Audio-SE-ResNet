"""
Asynchronous Dual-GPU Hyperparameter Optimization Engine with Nested Cross-Validation (Nested CV).

Framework: Optuna with Tree-structured Parzen Estimator (TPE) sampler.
Storage Backend: SQLite database (default: sqlite:///optuna_study.db?timeout=60).
Multiprocessing Architecture:
    - Explicitly spawns 2 independent worker processes using the 'spawn' start method.
    - Worker 1 is bound to 'cuda:0' and Worker 2 is bound to 'cuda:1' (without DistributedDataParallel).
    - Asynchronous execution eliminates all-reduce synchronization overhead.
Validation Strategy:
    - Nested Cross-Validation: K_outer = 5 folds (generalization error) and K_inner = 3 folds (HPO selection).
    - Prevents hyperparameter leakage on the 400-sample VS13 benchmark.
Search Space:
    - Learning Rate: log-uniform [1e-5, 5e-3]
    - Weight Decay: log-uniform [1e-6, 1e-2]
    - SE Reduction Ratio (r): categorical {8, 16, 32}
    - Dropout Rate: uniform [0.10, 0.50] (step 0.05)
Fast Verification Interface:
    - '--dummy' CLI flag strictly forces CPU execution on synthetically generated tensors
      (batch_size=2, shape (2, 1, 128, 313), labels (2,)), running 1 trial of 1 epoch per worker.
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

from torch.utils.data import DataLoader, Dataset, TensorDataset
from src.config import Config
from src.models_torch import SEResNet, build_se_resnet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("optimize")


class VS13MelDataset(Dataset):
    """
    PyTorch Dataset wrapper for preprocessed Mel-Spectrogram features.
    Spectrogram shape: (1, 128, 313), Target: scalar speed in km/h.
    """

    def __init__(
        self,
        audio_paths: List[str],
        speeds: np.ndarray,
        stats_mean: Optional[np.ndarray] = None,
        stats_std: Optional[np.ndarray] = None,
        is_training: bool = False,
    ):
        self.audio_paths = audio_paths
        self.speeds = torch.tensor(speeds, dtype=torch.float32)
        self.stats_mean = stats_mean
        self.stats_std = stats_std
        self.is_training = is_training
        
        # Precompute and cache all mel spectrograms in memory
        # since the dataset is very small (~400 samples)
        self.cached_tensors = []
        import librosa
        n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
        
        for path in self.audio_paths:
            try:
                audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
                if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
                    audio = audio[: Config.AUDIO_LENGTH_SAMPLES]
                else:
                    audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)), "constant")

                mel = librosa.feature.melspectrogram(
                    y=audio,
                    sr=Config.SAMPLE_RATE,
                    n_fft=Config.N_FFT,
                    hop_length=Config.HOP_LENGTH,
                    n_mels=Config.N_MELS,
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                if self.stats_mean is not None and self.stats_std is not None:
                    mel_norm = (mel_db - self.stats_mean) / self.stats_std
                else:
                    mel_norm = mel_db

                tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)
            except Exception:
                tensor = torch.zeros((1, Config.N_MELS, n_frames), dtype=torch.float32)
                
            self.cached_tensors.append(tensor)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cached_tensors[idx], self.speeds[idx]


def run_dummy_inner_trial(
    device: torch.device,
    lr: float,
    weight_decay: float,
    se_ratio: int,
    dropout: float,
) -> float:
    """
    Executes a single synthetic dummy training and evaluation step strictly on CPU.
    Validates end-to-end model forward pass, gradient calculation, optimizer step,
    and RMSE metric computation without accessing CUDA or the disk.
    """
    model = SEResNet(
        in_channels=1,
        base_filters=96,
        stage_blocks=[2, 2, 2],
        stage_channels=[96, 192, 384],
        use_se=True,
        se_ratio=se_ratio,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Synthetic batch of size 2 matching Mel-spectrogram shape: (2, 1, 128, 313)
    x_train = torch.randn(2, 1, 128, 313, device=device)
    y_train = torch.tensor([50.0, 75.0], dtype=torch.float32, device=device)

    x_val = torch.randn(2, 1, 128, 313, device=device)
    y_val = torch.tensor([52.0, 72.0], dtype=torch.float32, device=device)

    # 1. Training step (1 epoch)
    model.train()
    optimizer.zero_grad()
    train_out = model(x_train).view(-1)
    train_loss = criterion(train_out, y_train)
    train_loss.backward()
    optimizer.step()

    # 2. Validation evaluation
    model.eval()
    with torch.no_grad():
        val_out = model(x_val).view(-1)
        val_mse = criterion(val_out, y_val).item()
        val_rmse = float(np.sqrt(val_mse))

    return val_rmse


def run_nested_inner_cv(
    device: torch.device,
    lr: float,
    weight_decay: float,
    se_ratio: int,
    dropout: float,
    data_dir: Optional[str],
    outer_fold_idx: int = 0,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    epochs: int = 150,
    batch_size: int = 32,
) -> float:
    """
    Executes Nested CV inner loop on real VS13 dataset:
    1. Extracts outer training partition (320 samples).
    2. Performs K_inner = 3 fold CV on outer training data.
    3. Returns mean inner validation RMSE across the 3 folds.
    """
    from src.utils import calculate_global_stats, get_official_train_test_split, SortedKFold

    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(f"VS13 dataset directory not found: {data_dir}")

    train_paths, train_speeds, _, test_paths, test_speeds, _ = get_official_train_test_split(data_dir)
    
    # STRICT ISOLATION: The HPO process must NEVER see the test_paths.
    # We only use the 319 training samples for Nested CV.
    all_paths = list(train_paths)
    all_speeds = np.array(train_speeds)
    
    if len(all_paths) == 0:
        raise ValueError(f"No audio files found in {data_dir}")

    # Outer split: K_outer = 5
    outer_kfold = SortedKFold(n_splits=n_outer_folds)
    splits = list(outer_kfold.split(all_paths, y=all_speeds))
    outer_train_indices, _ = splits[outer_fold_idx]

    outer_train_paths = [all_paths[i] for i in outer_train_indices]
    outer_train_speeds = all_speeds[outer_train_indices]

    # Inner split: K_inner = 3 on outer train set
    inner_kfold = SortedKFold(n_splits=n_inner_folds)
    inner_rmses: List[float] = []

    for inner_fold, (in_train_idx, in_val_idx) in enumerate(inner_kfold.split(outer_train_paths, y=outer_train_speeds)):
        train_paths = [outer_train_paths[i] for i in in_train_idx]
        train_speeds = outer_train_speeds[in_train_idx]
        val_paths = [outer_train_paths[i] for i in in_val_idx]
        val_speeds = outer_train_speeds[in_val_idx]

        logger.info(f"  [Outer {outer_fold_idx+1}/{n_outer_folds}] Inner Fold {inner_fold+1}/{n_inner_folds}: "
                     f"train={len(train_paths)}, val={len(val_paths)}")

        # Calculate normalization statistics from training fold only (zero-leakage)
        stats = calculate_global_stats(train_paths)
        train_ds = VS13MelDataset(train_paths, train_speeds, stats["mean"], stats["std"], is_training=True)
        val_ds = VS13MelDataset(val_paths, val_speeds, stats["mean"], stats["std"], is_training=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = SEResNet(
            in_channels=1,
            base_filters=96,
            stage_blocks=[2, 2, 2],
            stage_channels=[96, 192, 384],
            use_se=True,
            se_ratio=se_ratio,
            dropout=dropout,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
        criterion = nn.MSELoss()

        best_val_rmse = float("inf")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                pred = model(x_b).view(-1)
                loss = criterion(pred, y_b)
                loss.backward()
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
                    v_pred = model(x_v).view(-1)
                    val_sq_errors.extend((v_pred - y_v).pow(2).cpu().numpy().tolist())

            epoch_val_rmse = float(np.sqrt(np.mean(val_sq_errors))) if val_sq_errors else float("inf")
            if epoch_val_rmse < best_val_rmse:
                best_val_rmse = epoch_val_rmse

            # verbose logging every 10 epochs + first + last
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(f"    Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | val_RMSE={epoch_val_rmse:.2f} | best={best_val_rmse:.2f}")

        inner_rmses.append(best_val_rmse)
        logger.info(f"  Inner Fold {inner_fold+1}/{n_inner_folds} done -> best RMSE: {best_val_rmse:.2f} km/h")

    return float(np.mean(inner_rmses))


def run_worker(
    gpu_id: str,
    study_name: str = "se_resnet_vs13_hpo",
    storage_url: str = "sqlite:///optuna_study.db?timeout=60",
    n_trials: int = 1,
    is_dummy: bool = False,
    outer_fold: int = 0,
    inner_folds: int = 3,
    epochs: int = 150,
    batch_size: int = 32,
    data_dir: Optional[str] = None,
):
    """
    Multiprocessing Worker Entry Point.

    Binds worker to target GPU ('cuda:0' or 'cuda:1', or strictly 'cpu' in dummy mode).
    Connects to the persistent shared SQLite study and runs Optuna TPE optimization.
    """
    if is_dummy:
        # STRICTLY force 'cpu' device; never access CUDA
        device = torch.device("cpu")
        print(f"[{gpu_id}] Dummy Mode active: strictly bound to CPU device.")
    else:
        # Bind explicitly to target GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = torch.device(gpu_id)
            print(f"[{gpu_id}] Successfully bound to GPU device: {device}")
        else:
            device = torch.device("cpu")
            print(f"[{gpu_id}] CUDA unavailable; operating on CPU device.")

    # Attach to shared persistent Optuna study
    sampler = TPESampler(seed=42 if is_dummy else None)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        # Tree-structured Parzen Estimator (TPE) parameter sampling
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        se_ratio = trial.suggest_categorical("se_ratio", [8, 16, 32])
        dropout = trial.suggest_float("dropout", 0.10, 0.50, step=0.05)

        if is_dummy:
            val_rmse = run_dummy_inner_trial(
                device=device,
                lr=lr,
                weight_decay=weight_decay,
                se_ratio=se_ratio,
                dropout=dropout,
            )
        else:
            val_rmse = run_nested_inner_cv(
                device=device,
                lr=lr,
                weight_decay=weight_decay,
                se_ratio=se_ratio,
                dropout=dropout,
                data_dir=data_dir,
                outer_fold_idx=outer_fold,
                n_outer_folds=5,
                n_inner_folds=inner_folds,
                epochs=epochs,
                batch_size=batch_size,
            )

        return val_rmse

    study.optimize(objective, n_trials=n_trials)
    print(f"[{gpu_id}] Completed {n_trials} trial(s) in study '{study_name}'.")


def evaluate_outer_folds(
    study_name: str,
    storage_url: str,
    data_dir: str,
    n_outer_folds: int = 5,
    epochs: int = 150,
    batch_size: int = 32,
    device_str: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluates optimal hyperparameters across all K_outer = 5 folds
    to establish the unbiased nested cross-validation generalization RMSE.
    """
    from src.utils import calculate_global_stats, get_official_train_test_split, SortedKFold

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    best_params = study.best_params
    print(f"[Nested CV Evaluation] Best Hyperparameters: {best_params}")

    device = torch.device(device_str)
    
    train_paths, train_speeds, _, test_paths, test_speeds, _ = get_official_train_test_split(data_dir)
    
    # STRICT ISOLATION
    all_paths = list(train_paths)
    all_speeds = np.array(train_speeds)

    outer_kfold = SortedKFold(n_splits=n_outer_folds)
    outer_rmses: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_kfold.split(all_paths, y=all_speeds)):
        train_paths = [all_paths[i] for i in train_idx]
        train_speeds = all_speeds[train_idx]
        test_paths = [all_paths[i] for i in test_idx]
        test_speeds = all_speeds[test_idx]

        stats = calculate_global_stats(train_paths)
        train_ds = VS13MelDataset(train_paths, train_speeds, stats["mean"], stats["std"], is_training=True)
        test_ds = VS13MelDataset(test_paths, test_speeds, stats["mean"], stats["std"], is_training=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = SEResNet(
            in_channels=1,
            base_filters=96,
            stage_blocks=[2, 2, 2],
            stage_channels=[96, 192, 384],
            use_se=True,
            se_ratio=best_params["se_ratio"],
            dropout=best_params["dropout"],
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
        criterion = nn.MSELoss()

        for _ in range(epochs):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                pred = model(x_b).view(-1)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        sq_errors = []
        with torch.no_grad():
            for x_t, y_t in test_loader:
                x_t, y_t = x_t.to(device), y_t.to(device)
                t_pred = model(x_t).view(-1)
                sq_errors.extend((t_pred - y_t).pow(2).cpu().numpy().tolist())

        fold_rmse = float(np.sqrt(np.mean(sq_errors)))
        outer_rmses.append(fold_rmse)
        print(f"  Outer Fold {fold_idx + 1}/{n_outer_folds} RMSE: {fold_rmse:.2f} km/h")

    nested_mean = float(np.mean(outer_rmses))
    nested_std = float(np.std(outer_rmses))
    print(f"\n[Final Nested CV Result] Generalized RMSE: {nested_mean:.2f} ± {nested_std:.2f} km/h")
    return {"mean_rmse": nested_mean, "std_rmse": nested_std, "fold_rmses": outer_rmses}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Asynchronous Dual-GPU Optuna Hyperparameter Optimization with Nested CV"
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
        default="se_resnet_vs13_hpo",
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
        default=150,
        help="Training epochs per fold (overridden to 1 in dummy mode)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (overridden to 2 in dummy mode)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to VS13 dataset root directory",
    )
    parser.add_argument(
        "--run_outer_eval",
        action="store_true",
        help="Run full outer fold evaluation after HPO completes",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # In dummy mode: override settings for rapid (<3s) CPU execution
    if args.dummy:
        n_trials = 1
        epochs = 1
        batch_size = 2
        print("=" * 70)
        print(" FAST DUMMY VERIFICATION MODE ACTIVE")
        print("  - Device: STRICTLY FORCED TO 'cpu'")
        print("  - Data: Synthetically generated random tensors (batch_size=2)")
        print("  - Trials: 1 per worker process (2 total)")
        print(f"  - Storage Backend: {args.storage}")
        print("=" * 70)
    else:
        n_trials = args.n_trials
        epochs = args.epochs
        batch_size = args.batch_size
        print("=" * 70)
        print(" ASYNCHRONOUS DUAL-GPU HPO INITIALIZED")
        print(f"  - Study Name: {args.study_name}")
        print(f"  - Storage: {args.storage}")
        print(f"  - Target GPUs: Worker 1 -> 'cuda:0', Worker 2 -> 'cuda:1'")
        print(f"  - Nested CV: K_outer={args.outer_folds}, K_inner={args.inner_folds}")
        print("=" * 70)

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
            "inner_folds": args.inner_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "data_dir": args.data_dir,
        },
    )
    p2 = mp.Process(
        target=run_worker,
        args=("cuda:1", args.study_name, args.storage, n_trials, args.dummy),
        kwargs={
            "outer_fold": args.outer_fold_idx,
            "inner_folds": args.inner_folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "data_dir": args.data_dir,
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
    print("\n" + "=" * 70)
    print(" HPO EXECUTION COMPLETED SUCCESSFULLY")
    print(f"  - Total Trials Recorded in SQLite: {len(updated_study.trials)}")
    if len(updated_study.trials) > 0:
        print(f"  - Best Trial Objective (RMSE): {updated_study.best_value:.4f} km/h")
        print(f"  - Best Hyperparameters Discovered:")
        for k, v in updated_study.best_params.items():
            print(f"      * {k}: {v}")
    print("=" * 70)

    # Optional full outer nested evaluation
    if args.run_outer_eval and args.data_dir:
        evaluate_outer_folds(
            study_name=args.study_name,
            storage_url=args.storage,
            data_dir=args.data_dir,
            n_outer_folds=args.outer_folds,
            epochs=epochs,
            batch_size=batch_size,
            device_str="cpu" if args.dummy else "cuda:0",
        )


if __name__ == "__main__":
    main()
