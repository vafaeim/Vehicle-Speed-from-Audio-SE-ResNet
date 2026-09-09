"""
Tier 4: Data Engineering, Splitting & Preprocessing Tests
Tests deterministic sorted dataset discovery, 10-fold cross-validation reproducibility,
zero-overlap split partitions, audio padding/cropping, batch collation, and augmentations.
"""

import os
import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


# --- Reference Data Loading Components (Specification Oracle) ---

def discover_dataset_files(data_root: str):
    """
    Discovers all audio files deterministically sorted lexicographically.
    Ensures identical file order across systems.
    """
    file_list = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.endswith(".wav"):
                file_list.append(os.path.join(root, f))
    file_list.sort()
    return file_list


def create_10fold_splits(num_samples: int, n_folds: int = 10, seed: int = 42):
    """
    Creates deterministic KFold cross-validation train/val splits.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(num_samples)
    return list(kf.split(indices))


def pad_or_crop_audio(audio: np.ndarray, target_length: int = 160000) -> np.ndarray:
    """
    Pads with zeros or crops audio to exact target_length.
    """
    length = len(audio)
    if length > target_length:
        return audio[:target_length]
    elif length < target_length:
        return np.pad(audio, (0, target_length - length), mode="constant")
    return audio


def apply_gain_augmentation(audio: np.ndarray, gain_db_range: tuple = (-6.0, 6.0)) -> np.ndarray:
    """Scales audio amplitude within given decibel gain range."""
    gain_db = np.random.uniform(gain_db_range[0], gain_db_range[1])
    scale = 10.0 ** (gain_db / 20.0)
    return audio * scale


def apply_additive_noise(audio: np.ndarray, snr_db_range: tuple = (10.0, 25.0)) -> np.ndarray:
    """Adds white Gaussian noise with target SNR in decibels."""
    power = np.mean(audio**2)
    if power < 1e-8:
        return audio
    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=len(audio)).astype(audio.dtype)
    return audio + noise


class SyntheticAudioDataset(Dataset):
    """PyTorch Dataset for audio speed regression."""
    def __init__(self, waveforms: list, speeds: list, target_length: int = 160000, is_training: bool = False):
        self.waveforms = waveforms
        self.speeds = speeds
        self.target_length = target_length
        self.is_training = is_training
        
    def __len__(self):
        return len(self.waveforms)
        
    def __getitem__(self, idx):
        wave = self.waveforms[idx].copy()
        if self.is_training:
            wave = apply_gain_augmentation(wave)
            wave = apply_additive_noise(wave)
        wave = pad_or_crop_audio(wave, self.target_length)
        
        # Max normalization
        max_val = np.max(np.abs(wave))
        if max_val > 1e-6:
            wave = wave / max_val
            
        x = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)  # (1, target_length)
        y = torch.tensor([self.speeds[idx]], dtype=torch.float32)  # (1,)
        return x, y


# --- Unit Tests ---

def test_deterministic_sorted_discovery(temp_dataset_dir):
    """Verifies that dataset discovery is deterministic and lexicographically sorted."""
    files1 = discover_dataset_files(temp_dataset_dir)
    files2 = discover_dataset_files(temp_dataset_dir)
    
    assert len(files1) == 20, f"Expected 20 discovered files, got {len(files1)}"
    assert files1 == files2, "File lists between repeated discoveries are not identical"
    assert files1 == sorted(files1), "Discovered files are not strictly sorted"


def test_10fold_split_reproducibility():
    """Verifies that KFold splits with seed=42 are 100% reproducible across separate calls."""
    splits_run1 = create_10fold_splits(num_samples=100, n_folds=10, seed=42)
    splits_run2 = create_10fold_splits(num_samples=100, n_folds=10, seed=42)
    
    for fold in range(10):
        train1, val1 = splits_run1[fold]
        train2, val2 = splits_run2[fold]
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)


def test_10fold_split_disjointness_and_completeness():
    r"""
    Verifies that for each fold:
    1. train_indices and val_indices have zero intersection (train \cap val == \emptyset).
    2. Union equals all samples.
    3. Union of all validation folds covers the entire dataset exactly once.
    """
    num_samples = 130  # Representative VS13 dataset size
    splits = create_10fold_splits(num_samples=num_samples, n_folds=10, seed=42)
    
    all_val_indices = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        # 1. Zero overlap
        intersection = np.intersect1d(train_idx, val_idx)
        assert len(intersection) == 0, f"Fold {fold} has non-empty intersection: {intersection}"
        
        # 2. Completeness
        union = np.union1d(train_idx, val_idx)
        assert len(union) == num_samples, f"Fold {fold} does not cover all samples"
        
        all_val_indices.extend(val_idx.tolist())
        
    # 3. All validation folds together partition the dataset
    all_val_sorted = sorted(all_val_indices)
    expected = list(range(num_samples))
    assert all_val_sorted == expected, "Validation folds do not partition dataset cleanly"


def test_audio_padding_short_signal():
    """Verifies that audio shorter than target_length is right-padded with zeros."""
    short_audio = np.ones(80000, dtype=np.float32)
    padded = pad_or_crop_audio(short_audio, target_length=160000)
    
    assert len(padded) == 160000, f"Expected length 160000, got {len(padded)}"
    assert np.all(padded[:80000] == 1.0), "Original content was altered during padding"
    assert np.all(padded[80000:] == 0.0), "Padding region is not zero"


def test_audio_cropping_long_signal():
    """Verifies that audio longer than target_length is truncated to target_length."""
    long_audio = np.arange(200000, dtype=np.float32)
    cropped = pad_or_crop_audio(long_audio, target_length=160000)
    
    assert len(cropped) == 160000, f"Expected length 160000, got {len(cropped)}"
    assert np.all(cropped == long_audio[:160000]), "Cropping produced corrupted audio"


def test_audio_exact_length_unchanged():
    """Verifies that audio already at target_length is unmodified."""
    exact_audio = np.random.randn(160000).astype(np.float32)
    processed = pad_or_crop_audio(exact_audio, target_length=160000)
    assert np.array_equal(exact_audio, processed)


def test_batch_collation_shapes_and_types(synthetic_doppler_audio):
    """
    Verifies DataLoader batch collation produces tensors of shape:
    x: (B, 1, 160000) float32
    y: (B, 1) float32
    """
    wave = synthetic_doppler_audio["waveform"]
    speed = synthetic_doppler_audio["speed_kmh"]
    
    waveforms = [wave for _ in range(8)]
    speeds = [speed + float(i) for i in range(8)]
    
    dataset = SyntheticAudioDataset(waveforms, speeds, is_training=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    batch_x, batch_y = next(iter(loader))
    
    assert batch_x.shape == (4, 1, 160000), f"Expected x shape (4, 1, 160000), got {batch_x.shape}"
    assert batch_y.shape == (4, 1), f"Expected y shape (4, 1), got {batch_y.shape}"
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.float32


def test_gain_augmentation_preserves_finite_values():
    """Verifies that gain augmentation produces finite values without overflow or NaN."""
    np.random.seed(42)
    audio = np.random.randn(1000).astype(np.float32)
    augmented = apply_gain_augmentation(audio, gain_db_range=(-6.0, 6.0))
    
    assert np.all(np.isfinite(augmented)), "Gain augmentation produced NaN or Inf"
    # Max possible gain is 10^(6/20) \approx 1.995
    max_scale = 10.0 ** (6.0 / 20.0)
    assert np.max(np.abs(augmented)) <= np.max(np.abs(audio)) * max_scale + 1e-4


def test_additive_noise_preserves_finite_values():
    """Verifies that additive white noise preserves audio shape and finite values."""
    np.random.seed(42)
    audio = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
    noisy = apply_additive_noise(audio, snr_db_range=(10.0, 25.0))
    
    assert len(noisy) == len(audio)
    assert np.all(np.isfinite(noisy))
    assert not np.array_equal(audio, noisy), "Additive noise did not modify audio"
