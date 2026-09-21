import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

"""
Systematic Component-Wise Ablation Studies Runner for SE-ResNet Acoustic Speed Estimation.

Isolates individual architectural and operational mechanisms across three core groups:
    1. Group 1 (SE Attention Recalibration):
       - Baseline ResNet (No SE blocks, use_se=False)
       - SE-ResNet (r=8, high capacity attention bottleneck)
       - SE-ResNet (r=16, balanced standard attention bottleneck)
       - SE-ResNet (r=32, ultra-compact attention bottleneck)
    2. Group 2 (Network Depth and Residual Stage Hierarchy):
       - Shallow SE-ResNet (2 stages: [96, 192], 4 residual blocks)
       - Standard SE-ResNet (3 stages: [96, 192, 384], 6 residual blocks)
       - Deep SE-ResNet (4 stages: [96, 192, 384, 512], 8 residual blocks)
    3. Group 3 (Stochastic Environmental Augmentations):
       - No Augmentation (clean raw Mel-spectrograms, p=0.0)
       - Noise Only (Additive White Gaussian Noise SNR in [10, 25] dB, Gain disabled)
       - Gain Only (Random gain scaling in [-6, +6] dB, Noise disabled)
       - Full Augmentations (Gain + Noise, p=0.8)

Supported Features:
    - Modular PyTorch model building via `build_ablation_model` and `count_parameters` from `src.models_torch`
    - Systematic training and evaluation loops with Cosine Annealing learning rate scheduler
    - Structured CSV output logging:
      (experiment_name, variant_type, parameter_count, val_rmse, val_mae, latency_ms, group, etc.)
    - Fast synthetic CPU verification mode (`--dummy`):
      Strictly forces 'cpu', generates random synthetic tensors (batch_size=2, shape (2, 1, 128, 313)),
      and executes 1 fast epoch per variant without touching CUDA or disk.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Ensure repository root is in sys.path for direct CLI execution
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.config import Config
from src.models_torch import build_ablation_model, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ablation_runner")


@dataclass
class AblationConfig:
    """Configuration specification for a single ablation experiment variant."""

    experiment_name: str
    group: str  # 'se', 'depth', or 'aug'
    variant_type: str  # Short description of configuration
    variant_name: str  # Publication-facing label
    use_se: bool = True
    se_ratio: int = 16
    stages: int = 3
    base_filters: int = 96
    dropout: float = 0.3
    noise_snr_db: Tuple[float, float] = (10.0, 25.0)
    use_noise: bool = True
    augment_prob: float = 0.8


@dataclass
class AblationResult:
    """Container for ablation experiment evaluation metrics."""

    experiment_name: str
    variant_type: str
    variant_name: str
    group: str
    parameter_count: int
    val_rmse: float
    val_mae: float
    latency_ms: float
    use_se: bool
    se_ratio: int
    stages: int
    augmentation: str


CSV_FIELDNAMES = [
    "experiment_name",
    "variant_type",
    "variant_name",
    "group",
    "parameter_count",
    "val_rmse",
    "val_mae",
    "latency_ms",
    "use_se",
    "se_ratio",
    "stages",
    "augmentation",
]


# ---------------------------------------------------------------------------
# Ablation Experiment Definition Matrices
# ---------------------------------------------------------------------------

GROUP_SE: List[AblationConfig] = [
    AblationConfig(
        experiment_name="exp_se_none_baseline",
        group="se",
        variant_type="No SE Blocks",
        variant_name="Baseline ResNet (No SE)",
        use_se=False,
        se_ratio=16,
        stages=3,
        use_noise=True,
    ),
    AblationConfig(
        experiment_name="exp_se_ratio_8",
        group="se",
        variant_type="SE Ratio r=8",
        variant_name="SE-ResNet (r=8)",
        use_se=True,
        se_ratio=8,
        stages=3,
        use_noise=True,
    ),
    AblationConfig(
        experiment_name="exp_se_ratio_16",
        group="se",
        variant_type="SE Ratio r=16 (Standard)",
        variant_name="SE-ResNet (r=16)",
        use_se=True,
        se_ratio=16,
        stages=3,
        use_noise=True,
    ),
    AblationConfig(
        experiment_name="exp_se_ratio_32",
        group="se",
        variant_type="SE Ratio r=32",
        variant_name="SE-ResNet (r=32)",
        use_se=True,
        se_ratio=32,
        stages=3,
        use_noise=True,
    ),
]

GROUP_DEPTH: List[AblationConfig] = [
    AblationConfig(
        experiment_name="exp_depth_2stages_shallow",
        group="depth",
        variant_type="2 Stages [96, 192]",
        variant_name="Shallow SE-ResNet (2 stages)",
        use_se=True,
        se_ratio=16,
        stages=2,
        use_noise=True,
    ),
    AblationConfig(
        experiment_name="exp_depth_3stages_standard",
        group="depth",
        variant_type="3 Stages [96, 192, 384]",
        variant_name="Standard SE-ResNet (3 stages)",
        use_se=True,
        se_ratio=16,
        stages=3,
        use_noise=True,
    ),
    AblationConfig(
        experiment_name="exp_depth_4stages_deep",
        group="depth",
        variant_type="4 Stages [96, 192, 384, 512]",
        variant_name="Deep SE-ResNet (4 stages)",
        use_se=True,
        se_ratio=16,
        stages=4,
        use_noise=True,
    ),
]

GROUP_AUG: List[AblationConfig] = [
    AblationConfig(
        experiment_name="exp_aug_none",
        group="aug",
        variant_type="Clean Spectrograms (p=0.0)",
        variant_name="No Augmentation (Clean)",
        use_se=True,
        se_ratio=16,
        stages=3,
        use_noise=False,
        augment_prob=0.0,
    ),
    AblationConfig(
        experiment_name="exp_aug_noise_only",
        group="aug",
        variant_type="AWGN SNR 10-25 dB",
        variant_name="Noise Only (AWGN)",
        use_se=True,
        se_ratio=16,
        stages=3,
        use_noise=True,
        augment_prob=0.8,
    ),
    AblationConfig(
        experiment_name="exp_aug_gain_only",
        group="aug",
        variant_type="Random Gain [-6, +6] dB",
        variant_name="Gain Only",
        use_se=True,
        se_ratio=16,
        stages=3,
        use_noise=False,
        augment_prob=0.8,
    ),
    AblationConfig(
        experiment_name="exp_aug_full",
        group="aug",
        variant_type="Gain + AWGN (p=0.8)",
        variant_name="Full Augmentations (Gain + Noise)",
        use_se=True,
        se_ratio=16,
        stages=3,
        use_noise=True,
        augment_prob=0.8,
    ),
]


def get_ablation_configs(group_name: str) -> List[AblationConfig]:
    """Resolves CLI ablation group choice to list of configurations."""
    cleaned = group_name.lower().strip()
    if cleaned == "se":
        return list(GROUP_SE)
    elif cleaned == "depth":
        return list(GROUP_DEPTH)
    elif cleaned == "aug":
        return list(GROUP_AUG)
    elif cleaned == "all":
        return list(GROUP_SE) + list(GROUP_DEPTH) + list(GROUP_AUG)
    else:
        raise ValueError(
            f"Unknown ablation group '{group_name}'. Must be one of: 'all', 'se', 'depth', 'aug'."
        )


# ---------------------------------------------------------------------------
# Audio Augmentations & Dataset
# ---------------------------------------------------------------------------


def apply_augmentations(
    audio: np.ndarray,
    noise_snr_db: Tuple[float, float] = (10.0, 25.0),
    use_noise: bool = True,
    augment_prob: float = 0.8,
) -> np.ndarray:
    """
    Applies stochastic environmental acoustic augmentations:
    1. Random Gain Scaling in [-6, +6] dB
    2. Additive White Gaussian Noise (AWGN) with SNR in [10, 25] dB
    """
    if not (use_gain or use_noise):
        return audio

    if random.random() > augment_prob:
        return audio

    # 1. Gain Augmentation
    if use_gain:
        gain_db = random.uniform(*Config.GAIN_DB)
        audio = audio * (10.0 ** (gain_db / 20.0))

    # 2. Additive White Gaussian Noise Augmentation
    if use_noise:
        snr_db = random.uniform(*Config.NOISE_SNR_DB)
        power = np.sum(audio ** 2) / max(1, len(audio))
        if power > 1e-6:
            noise_power = power / (10.0 ** (snr_db / 10.0))
            noise = np.random.normal(0.0, np.sqrt(noise_power), len(audio))
            audio = audio + noise

    return audio


class VS13AblationDataset(Dataset):
    """
    PyTorch Dataset wrapper for acoustic speed estimation with modular augmentations.
    Caches raw audio waveforms in memory to avoid redundant disk I/O every epoch.
    """

    def __init__(
        self,
        audio_paths: List[str],
        speeds: np.ndarray,
        stats_mean: Optional[np.ndarray] = None,
        stats_std: Optional[np.ndarray] = None,
        is_training: bool = False,
        noise_snr_db: Tuple[float, float] = (10.0, 25.0),
        use_noise: bool = True,
        augment_prob: float = 0.8,
        preloaded_audio: Optional[List[np.ndarray]] = None,
    ):
        self.speeds = torch.tensor(speeds, dtype=torch.float32)
        self.stats_mean = stats_mean
        self.stats_std = stats_std
        self.is_training = is_training
        self.use_gain = use_gain
        self.use_noise = use_noise
        self.augment_prob = augment_prob

        # Cache raw audio waveforms in memory (expensive librosa.load only once)
        import librosa
        
        if preloaded_audio is not None:
            self.cached_audio = preloaded_audio
        else:
            self.cached_audio = []
            for path in audio_paths:
                try:
                    audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
                    if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
                        audio = audio[: Config.AUDIO_LENGTH_SAMPLES]
                    else:
                        audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)), "constant")
                except Exception:
                    audio = np.zeros(Config.AUDIO_LENGTH_SAMPLES, dtype=np.float32)
                self.cached_audio.append(audio)

        # For validation (no augmentation), pre-compute mel tensors for max speed
        if not is_training:
            self.cached_tensors = []
            for audio in self.cached_audio:
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=Config.SAMPLE_RATE,
                    n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS,
                )
                mel_db = librosa.power_to_db(mel, ref=1.0)
                if self.stats_mean is not None and self.stats_std is not None:
                    mel_db = (mel_db - self.stats_mean) / self.stats_std
                self.cached_tensors.append(torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0))
        else:
            self.cached_tensors = None

    def __len__(self) -> int:
        return len(self.speeds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        speed = self.speeds[idx]

        # Validation: fully cached
        if self.cached_tensors is not None:
            return self.cached_tensors[idx], speed

        # Training: apply stochastic augmentations on cached audio
        import librosa
        audio = self.cached_audio[idx].copy()

        audio = apply_augmentations(
            audio,
            use_gain=self.use_gain,
            use_noise=self.use_noise,
            augment_prob=self.augment_prob,
        )


        mel = librosa.feature.melspectrogram(
            y=audio, sr=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS,
        )
        mel_db = librosa.power_to_db(mel, ref=1.0)

        if self.stats_mean is not None and self.stats_std is not None:
            mel_db = (mel_db - self.stats_mean) / self.stats_std

        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        return tensor, speed


# ---------------------------------------------------------------------------
# Benchmark & Evaluation Helpers
# ---------------------------------------------------------------------------


def benchmark_latency(
    model: nn.Module,
    sample_tensor: torch.Tensor,
    device: torch.device,
    repetitions: int = 20,
    warmup: int = 5,
) -> float:
    """
    Measures mean inference latency per sample in milliseconds.
    """
    model.eval()
    batch_size = sample_tensor.shape[0]

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(repetitions):
            _ = model(sample_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t1 = time.perf_counter()

    total_samples = repetitions * batch_size
    latency_ms = ((t1 - t0) / total_samples) * 1000.0
    return latency_ms


def format_aug_string(cfg: AblationConfig) -> str:
    if not getattr(cfg, 'use_noise', True):
        return 'Clean'
    return f'Noise SNR {getattr(cfg, "noise_snr_db", "N/A")}'


# ---------------------------------------------------------------------------
# Fast Dummy Verification Mode (CPU-only, Synthetic Data)
# ---------------------------------------------------------------------------


def run_dummy_variant(
    cfg: AblationConfig,
    device: torch.device,
    batch_size: int = 2,
) -> AblationResult:
    """
    Fast dummy verification run for a single ablation variant:
    - Strictly forces CPU device.
    - Uses synthetically generated random tensors (batch_size=2, shape (2, 1, 128, 313)).
    - Runs 1 fast dummy training epoch (forward, backward, optimizer step).
    - Runs 1 fast dummy validation evaluation (RMSE, MAE, inference latency).
    - Never accesses CUDA or the disk.
    """
    # 1. Instantiate model via modular factory
    model = build_ablation_model(
        input_shape=(1, 128, 313),
        use_se=cfg.use_se,
        se_ratio=cfg.se_ratio,
        stages=cfg.stages,
        base_filters=cfg.base_filters,
        dropout=cfg.dropout,
    ).to(device)

    param_count = count_parameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Synthetic batch of size 2 matching Mel-spectrogram shape: (2, 1, 128, 313)
    x_train = torch.randn(batch_size, 1, 128, 313, device=device)
    y_train = torch.tensor([50.0, 75.0], dtype=torch.float32, device=device)[:batch_size]

    # Verify augmentation code path with synthetic audio waveform
    if cfg.use_noise:
        dummy_audio = np.random.randn(Config.AUDIO_LENGTH_SAMPLES).astype(np.float32)
        _ = apply_augmentations(
            dummy_audio,
            noise_snr_db=getattr(cfg, "noise_snr_db", (10.0, 25.0)),
            use_noise=cfg.use_noise,
            augment_prob=1.0,
        )

    # 1. Training step (1 epoch)
    model.train()
    optimizer.zero_grad()
    train_pred = model(x_train).view(-1)
    train_loss = criterion(train_pred, y_train)
    train_loss.backward()
    optimizer.step()

    # 2. Validation evaluation step
    model.eval()
    x_val = torch.randn(batch_size, 1, 128, 313, device=device)
    y_val = torch.tensor([52.0, 72.0], dtype=torch.float32, device=device)[:batch_size]

    with torch.no_grad():
        val_pred = model(x_val).view(-1)
        val_mse = criterion(val_pred, y_val).item()
        val_rmse = float(np.sqrt(val_mse))
        val_mae = float(torch.mean(torch.abs(val_pred - y_val)).item())

    # Measure latency on CPU (fast benchmark in dummy mode)
    latency_ms = benchmark_latency(model, x_val, device, repetitions=2, warmup=1)

    return AblationResult(
        experiment_name=cfg.experiment_name,
        variant_type=cfg.variant_type,
        variant_name=cfg.variant_name,
        group=cfg.group,
        parameter_count=param_count,
        val_rmse=round(val_rmse, 4),
        val_mae=round(val_mae, 4),
        latency_ms=round(latency_ms, 3),
        use_se=cfg.use_se,
        se_ratio=cfg.se_ratio,
        stages=cfg.stages,
        augmentation=format_aug_string(cfg),
    )


# ---------------------------------------------------------------------------
# Real Training and Evaluation Loop (for real dataset runs)
# ---------------------------------------------------------------------------


def train_ablation_variant(
    cfg: AblationConfig,
    train_paths: List[str],
    train_speeds: np.ndarray,
    val_paths: List[str],
    val_speeds: np.ndarray,
    stats: Dict[str, Any],
    device: torch.device,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = Config.INIT_LR,
    weight_decay: float = Config.WEIGHT_DECAY,
    patience: int = 30,
    preloaded_audio_train: Optional[List[np.ndarray]] = None,
    preloaded_audio_val: Optional[List[np.ndarray]] = None,
) -> AblationResult:
    """
    Trains and evaluates a single ablation model variant on real dataset partitions.
    """
    # 1. Split Training Data into 80% Train / 20% Val for Early Stopping
    # val_paths here represents the pure Test Set
    sub_train_paths, sub_val_paths, sub_train_speeds, sub_val_speeds, sub_train_audio, sub_val_audio = train_test_split(
        train_paths, train_speeds, preloaded_audio_train, test_size=0.2, random_state=42
    )

    # 2. Dataset & DataLoaders
    mean_val = np.array(stats["mean"], dtype=np.float32)
    std_val = np.array(stats["std"], dtype=np.float32)

    train_ds = VS13AblationDataset(
        audio_paths=sub_train_paths,
        speeds=sub_train_speeds,
        stats_mean=mean_val,
        stats_std=std_val,
        is_training=True,
        noise_snr_db=getattr(cfg, "noise_snr_db", (10.0, 25.0)),
        use_noise=cfg.use_noise,
        augment_prob=cfg.augment_prob,
        preloaded_audio=sub_train_audio,
    )
    val_ds = VS13AblationDataset(
        audio_paths=sub_val_paths,
        speeds=sub_val_speeds,
        stats_mean=mean_val,
        stats_std=std_val,
        is_training=False,
        use_noise=False,
        preloaded_audio=sub_val_audio,
    )
    test_ds = VS13AblationDataset(
        audio_paths=val_paths,
        speeds=val_speeds,
        stats_mean=mean_val,
        stats_std=std_val,
        is_training=False,
        use_noise=False,
        preloaded_audio=preloaded_audio_val,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # 2. Build model
    model = build_ablation_model(
        input_shape=(1, 128, 313),
        use_se=cfg.use_se,
        se_ratio=cfg.se_ratio,
        stages=cfg.stages,
        base_filters=cfg.base_filters,
        dropout=cfg.dropout,
    ).to(device)

    param_count = count_parameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_val_mae = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b).view(-1)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation pass
        model.eval()
        sq_errors = []
        abs_errors = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v = model(x_v).view(-1)
                sq_errors.extend((pred_v - y_v).pow(2).cpu().numpy().tolist())
                abs_errors.extend((pred_v - y_v).abs().cpu().numpy().tolist())

        epoch_rmse = float(np.sqrt(np.mean(sq_errors)))
        epoch_mae = float(np.mean(abs_errors))

        if epoch_rmse < best_val_rmse:
            best_val_rmse = epoch_rmse
            best_val_mae = epoch_mae
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == epochs:
            logger.info(f"    Epoch {epoch:3d}/{epochs} - Val RMSE: {epoch_rmse:.2f} km/h (Best: {best_val_rmse:.2f} km/h)")

        if patience_counter >= patience:
            logger.info(f"    Early stopping triggered at epoch {epoch} (best RMSE: {best_val_rmse:.2f} km/h)")
            break

    # Restore best weights for latency benchmark
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    # Benchmark latency on a batch from validation loader
    first_val_batch, _ = next(iter(val_loader))
    first_val_batch = first_val_batch.to(device)
    latency_ms = benchmark_latency(model, first_val_batch, device, repetitions=20, warmup=5)

    return AblationResult(
        experiment_name=cfg.experiment_name,
        variant_type=cfg.variant_type,
        variant_name=cfg.variant_name,
        group=cfg.group,
        parameter_count=param_count,
        val_rmse=round(best_val_rmse, 4),
        val_mae=round(best_val_mae, 4),
        latency_ms=round(latency_ms, 3),
        use_se=cfg.use_se,
        se_ratio=cfg.se_ratio,
        stages=cfg.stages,
        augmentation=format_aug_string(cfg),
    )


# ---------------------------------------------------------------------------
# CSV Output and Table Formatting
# ---------------------------------------------------------------------------


def save_results_to_csv(results: List[AblationResult], output_csv_path: str) -> None:
    """Writes ablation experiment results to target CSV file."""
    output_dir = os.path.dirname(os.path.abspath(output_csv_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    logger.info(f"Ablation results successfully saved to: {output_csv_path}")


def print_results_table(results: List[AblationResult]) -> None:
    """Formats and prints summary table of ablation study results."""
    header = f"{'Group':<8} | {'Variant':<32} | {'Params':<11} | {'RMSE (km/h)':<13} | {'MAE (km/h)':<12} | {'Latency (ms)':<12}"
    separator = "-" * len(header)
    print("\n" + separator)
    print("                      ABLATION STUDY RESULTS SUMMARY")
    print(separator)
    print(header)
    print(separator)
    for r in results:
        print(
            f"{r.group:<8} | {r.variant_name:<32} | {r.parameter_count:>11,} | {r.val_rmse:>13.2f} | {r.val_mae:>12.2f} | {r.latency_ms:>12.2f}"
        )
    print(separator + "\n")


# ---------------------------------------------------------------------------
# CLI & Main Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Systematic Component-Wise Ablation Studies Runner for SE-ResNet"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=["all", "se", "depth", "aug"],
        help="Ablation group to run: 'se', 'depth', 'aug', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ablation_results.csv",
        help="Output CSV file path for experiment results (default: 'ablation_results.csv')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Training epochs per variant (default: 150; overridden to 1 in dummy mode)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation (default: 32; overridden to 2 in dummy mode)",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Run fast synthetic CPU verification without touching CUDA or disk (Acceptance Criteria)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to VS13 dataset root directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target execution device ('cuda' or 'cpu'; strictly forced to 'cpu' in dummy mode)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience in epochs (default: 30)",
    )
    return parser.parse_args()



def ablation_worker_top(gpu_id, q_task, q_res, train_paths, train_speeds, val_paths, val_speeds, stats, epochs, batch_size, patience, preloaded_train, preloaded_val):
    import logging
    import torch
    dev = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    while not q_task.empty():
        try:
            task = q_task.get(timeout=3)
        except Exception:
            break
        i, total, c = task
        logger.info(f"[{dev}] [{i}/{total}] Starting ablation variant: {c.variant_name}")
        try:
            res = train_ablation_variant(
                cfg=c, train_paths=train_paths, train_speeds=train_speeds,
                val_paths=val_paths, val_speeds=val_speeds, stats=stats,
                device=dev, epochs=epochs, batch_size=batch_size,
                patience=patience, preloaded_audio_train=preloaded_train, preloaded_audio_val=preloaded_val
            )
            logger.info(f"[{dev}] [{i}/{total}] Completed {c.variant_name} - Val RMSE={res.val_rmse:.2f} km/h")
            q_res.put(res)
        except Exception as e:
            logger.error(f"[{dev}] Error in {c.variant_name}: {e}")

def main():
    args = parse_args()

    # Seed RNGs for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    configs = get_ablation_configs(args.ablation)

    if args.dummy:
        # STRICTLY force 'cpu' device; never access CUDA or disk
        device = torch.device("cpu")
        batch_size = 2
        print("=" * 75)
        print(" FAST DUMMY VERIFICATION MODE ACTIVE (Milestone M3)")
        print("  - Device: STRICTLY FORCED TO 'cpu'")
        print("  - Data: Synthetically generated random tensors (batch_size=2)")
        print(f"  - Target Ablation Group: {args.ablation} ({len(configs)} variant(s))")
        print(f"  - Output CSV: {args.output_csv}")
        print("=" * 75)

        results: List[AblationResult] = []
        for idx, cfg in enumerate(configs, 1):
            logger.info(
                f"[{idx}/{len(configs)}] Running dummy epoch for: {cfg.variant_name} ({cfg.variant_type})"
            )
            res = run_dummy_variant(cfg, device=device, batch_size=batch_size)
            results.append(res)
            logger.info(
                f"  -> Finished {cfg.variant_name}: Params={res.parameter_count:,}, "
                f"RMSE={res.val_rmse:.2f} km/h, MAE={res.val_mae:.2f} km/h, Latency={res.latency_ms:.2f} ms"
            )

        save_results_to_csv(results, args.output_csv)
        print_results_table(results)
        print("Fast dummy verification completed successfully with exit code 0.")
        return

    # Real Dataset Mode
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(" SYSTEMATIC ABLATION STUDIES RUNNER INITIALIZED")
    print(f"  - Target Group: {args.ablation} ({len(configs)} variant(s))")
    print(f"  - Execution Device: {device}")
    print(f"  - Epochs per Variant: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"  - Output CSV: {args.output_csv}")
    print("=" * 75)

    if not args.data_dir or not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: '{args.data_dir}'. "
            "Please provide a valid path via --data_dir, or run with --dummy for fast verification."
        )

    from src.utils import calculate_global_stats, get_official_train_test_split

    train_paths, train_speeds, _, val_paths, val_speeds, _ = get_official_train_test_split(args.data_dir)
    
    if len(train_paths) == 0:
        raise ValueError(f"No audio files discovered in dataset directory: {args.data_dir}")

    all_paths = np.concatenate([train_paths, val_paths])
    logger.info(
        f"Dataset loaded: {len(all_paths)} clips total ({len(train_paths)} train, {len(val_paths)} val (test))"
    )

    # Calculate normalization statistics on training split only (leakage-free)
    stats = calculate_global_stats(train_paths)

    logger.info(f"Pre-loading {len(all_paths)} audio files into memory...")
    import librosa
    master_audio = []
    for i, path in enumerate(all_paths):
        if i % 100 == 0: logger.info(f"  Loaded {i}/{len(all_paths)}")
        try:
            audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
            if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
                audio = audio[: Config.AUDIO_LENGTH_SAMPLES]
            else:
                audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)), "constant")
        except Exception:
            audio = np.zeros(Config.AUDIO_LENGTH_SAMPLES, dtype=np.float32)
        master_audio.append(audio)

    # Re-split audio mapping directly
    path_to_audio = {p: a for p, a in zip(all_paths, master_audio)}
    preloaded_train = [path_to_audio[p] for p in train_paths]
    preloaded_val = [path_to_audio[p] for p in val_paths]

    # Prepare multiprocessing queues
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    for idx, cfg in enumerate(configs, 1):
        task_queue.put((idx, len(configs), cfg))

    n_gpus = torch.cuda.device_count()
    if n_gpus < 1: n_gpus = 1

    logger.info(f"Starting {n_gpus} dual-GPU workers to train {len(configs)} ablation variants in parallel...")
    
    # We define the worker function directly here or import it
    # But it's easier to just write it inline since it just wraps train_ablation_variant


    processes = []
    for i in range(n_gpus):
        p = mp.Process(target=ablation_worker_top, args=(i % n_gpus, task_queue, result_queue, train_paths, train_speeds, val_paths, val_speeds, stats, args.epochs, args.batch_size, args.patience, preloaded_train, preloaded_val))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Sort results to match original config order roughly
    # (Since we didn't store the exact sort key, we'll sort by group and RMSE)
    results.sort(key=lambda r: (r.group, r.val_rmse))

    save_results_to_csv(results, args.output_csv)
    print_results_table(results)
    logger.info("All ablation experiments completed successfully.")


if __name__ == "__main__":
    main()
