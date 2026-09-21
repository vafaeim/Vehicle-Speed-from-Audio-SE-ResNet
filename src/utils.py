import os
import re
import numpy as np
import librosa
import json
from .config import Config

def get_official_train_test_split(data_root):
    """
    Parses the VS13 dataset directory structure and strictly honors the 
    official Train_valid_split.txt file provided by the dataset authors.
    Returns: (train_paths, train_speeds, train_classes, test_paths, test_speeds, test_classes)
    """
    train_paths, train_speeds, train_classes = [], [], []
    test_paths, test_speeds, test_classes = [], [], []
    
    vehicle_folders = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    for vehicle_folder in vehicle_folders:
        vehicle_path = os.path.join(data_root, vehicle_folder)
        split_file_path = os.path.join(vehicle_path, 'Train_valid_split.txt')
        
        if not os.path.exists(split_file_path): 
            # Fallback for incorrectly formatted folders
            for fname in os.listdir(vehicle_path):
                if fname.endswith(".wav"):
                    match = re.match(r"([a-zA-Z0-9]+)_(\d+)\.wav", fname)
                    if match:
                        train_paths.append(os.path.join(vehicle_path, fname))
                        train_speeds.append(int(match.group(2)))
                        train_classes.append(vehicle_folder)
            continue

        with open(split_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    base_name = parts[0]
                    split_type = parts[1].lower()
                    wav_file = os.path.join(vehicle_path, base_name + '.wav')
                    
                    if not os.path.exists(wav_file): 
                        continue
                    
                    match = re.match(r'([a-zA-Z0-9]+)_(\d+)\.wav', os.path.basename(wav_file))
                    if match:
                        speed = int(match.group(2))
                        if split_type == 'train':
                            train_paths.append(wav_file)
                            train_speeds.append(speed)
                            train_classes.append(vehicle_folder)
                        elif split_type == 'valid':
                            test_paths.append(wav_file)
                            test_speeds.append(speed)
                            test_classes.append(vehicle_folder)

    return (
        np.array(train_paths), np.array(train_speeds, dtype=np.float32), np.array(train_classes),
        np.array(test_paths), np.array(test_speeds, dtype=np.float32), np.array(test_classes)
    )

class SortedKFold:
    """
    Implements continuous stratification by sorting samples by target value (speed),
    chunking them into batches of size K, and distributing one sample per batch to each fold.
    This eliminates covariate shift across folds for continuous targets.
    """
    def __init__(self, n_splits=10):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("SortedKFold requires the target variable (y) for stratification.")
        
        # Sort indices by speed ascending
        sorted_idx = np.argsort(y)
        
        # Initialize folds
        folds = [[] for _ in range(self.n_splits)]
        
        # Distribute sequentially
        for i, idx in enumerate(sorted_idx):
            fold_idx = i % self.n_splits
            folds[fold_idx].append(idx)
            
        # Yield (train_idx, val_idx)
        for i in range(self.n_splits):
            val_idx = np.array(folds[i])
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield train_idx, val_idx


def calculate_global_stats(audio_paths, save_path=None):
    """
    Calculates mean and std of Mel Spectrograms across the dataset 
    for Z-score normalization.
    """
    print(f"Calculating stats for {len(audio_paths)} files...")
    mel_sums = np.zeros((Config.N_MELS, 1), dtype=np.float64)
    mel_sum_sqs = np.zeros((Config.N_MELS, 1), dtype=np.float64)
    total_frames = 0

    for i, path in enumerate(audio_paths):
        if i % 50 == 0: print(f"Processing {i}/{len(audio_paths)}...")
        try:
            audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
            # Pad/Crop
            if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
                audio = audio[:Config.AUDIO_LENGTH_SAMPLES]
            else:
                audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)), 'constant')
            
            mel = librosa.feature.melspectrogram(
                y=audio, sr=Config.SAMPLE_RATE, 
                n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS
            )
            mel_db = librosa.power_to_db(mel, ref=1.0)
            
            mel_sums += np.sum(mel_db, axis=1, keepdims=True)
            mel_sum_sqs += np.sum(mel_db**2, axis=1, keepdims=True)
            total_frames += mel_db.shape[1]
        except Exception as e:
            print(f"Error processing {path}: {e}")

    mel_mean = (mel_sums / total_frames).astype(np.float32)
    mel_std = np.sqrt(mel_sum_sqs / total_frames - mel_mean**2).astype(np.float32)
    mel_std[mel_std < 1e-8] = 1e-8 # Prevent divide by zero

    stats = {
        "mean": mel_mean.tolist(),
        "std": mel_std.tolist()
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(stats, f)
            
    return stats
import random
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
