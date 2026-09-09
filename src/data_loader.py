# dataset loading and preprocessing routines

import os
import re
import wave
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from .config import Config
from .utils import parse_speed_from_filename

# load audio from disk to float32 tensor
def load_audio(file_path):
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, sample_rate
    except Exception:
        pass

    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        
        if sampwidth == 2:
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            audio = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
            
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        return waveform, framerate

# pad or crop audio to target sample length
def pad_or_crop_audio(audio, target_length=160000):
    if isinstance(audio, np.ndarray):
        length = len(audio)
        if length > target_length:
            return audio[:target_length]
        elif length < target_length:
            return np.pad(audio, (0, target_length - length), mode='constant')
        return audio
    elif isinstance(audio, torch.Tensor):
        length = audio.shape[-1]
        if length > target_length:
            return audio[..., :target_length]
        elif length < target_length:
            return F.pad(audio, (0, target_length - length), mode='constant', value=0.0)
        return audio
    return audio

# apply random amplitude gain in decibels
def apply_gain_augmentation(audio, gain_db_range=(-6.0, 6.0)):
    gain_db = np.random.uniform(gain_db_range[0], gain_db_range[1])
    scale = 10.0 ** (gain_db / 20.0)
    if isinstance(audio, np.ndarray):
        return audio * scale
    return audio * float(scale)

# add white gaussian noise at target signal to noise ratio
def apply_additive_noise(audio, snr_db_range=(10.0, 25.0)):
    if isinstance(audio, torch.Tensor):
        audio_np = audio.cpu().numpy()
        is_tensor = True
    else:
        audio_np = audio
        is_tensor = False

    power = np.mean(audio_np ** 2)
    if power < 1e-8:
        return audio
    snr_db = np.random.uniform(snr_db_range[0], snr_db_range[1])
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=audio_np.shape).astype(audio_np.dtype)
    noisy = audio_np + noise

    if is_tensor:
        return torch.tensor(noisy, dtype=audio.dtype, device=audio.device)
    return noisy

# extract cartesian complex short time fourier transform
def compute_cartesian_stft(waveform, n_fft=2048, hop_length=512):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)
        
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    real = stft.real.unsqueeze(1)
    imag = stft.imag.unsqueeze(1)
    return torch.cat([real, imag], dim=1)

# extract differentiable instantaneous frequency
def compute_instantaneous_frequency(waveform, n_fft=2048, hop_length=512, sample_rate=16000):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)
        
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    s_curr = stft[..., 1:]
    s_prev = stft[..., :-1]
    cross = s_curr * torch.conj(s_prev)
    delta_phi = torch.atan2(cross.imag, cross.real)
    delta_phi = F.pad(delta_phi, (1, 0), mode='replicate')
    dt = hop_length / sample_rate
    inst_freq = delta_phi / (2.0 * np.pi * dt)
    return inst_freq.unsqueeze(1)

# vehicle speed estimation audio dataset
class VS13Dataset(Dataset):
    def __init__(self, file_paths, labels, target_length=160000, is_training=False, return_complex=False, use_cache=True):
        self.file_paths = file_paths
        self.labels = labels
        self.target_length = target_length
        self.is_training = is_training
        self.return_complex = return_complex
        self.use_cache = use_cache
        self.cache = {}
        
        # Pre-load entire dataset into RAM if cache is enabled
        if self.use_cache:
            for idx in range(len(self.file_paths)):
                path = self.file_paths[idx]
                waveform, _ = load_audio(path)
                if waveform.ndim == 2:
                    waveform = waveform.squeeze(0)
                waveform_np = waveform.numpy()
                self.cache[idx] = pad_or_crop_audio(waveform_np, self.target_length)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        speed = float(self.labels[idx])
        
        if self.use_cache and idx in self.cache:
            waveform_np = self.cache[idx].copy()
        else:
            path = self.file_paths[idx]
            waveform, _ = load_audio(path)
            if waveform.ndim == 2:
                waveform = waveform.squeeze(0)
                
            waveform_np = waveform.numpy()
            waveform_np = pad_or_crop_audio(waveform_np, self.target_length)
        
        # augmentations in training mode
        if self.is_training:
            if random.random() < Config.AUGMENT_PROB:
                waveform_np = apply_gain_augmentation(waveform_np, Config.GAIN_DB)
                waveform_np = apply_additive_noise(waveform_np, Config.NOISE_SNR_DB)
                
        # amplitude normalization
        max_val = np.max(np.abs(waveform_np))
        if max_val > 1e-6:
            waveform_np = waveform_np / max_val
            
        audio_tensor = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor([speed], dtype=torch.float32)
        
        # optional complex representation
        if self.return_complex:
            audio_tensor = compute_cartesian_stft(audio_tensor, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH).squeeze(0)
            
        return audio_tensor, label_tensor

# generate deterministic k-fold splits
def create_10fold_splits(num_samples, n_folds=10, seed=42):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(num_samples)
    return list(kf.split(indices))

# build dataloaders for training validation and test
def get_vs13_datasets(data_root, fold=None, n_folds=10, seed=42, batch_size=32, num_workers=2, return_complex=False):
    all_paths = []
    all_speeds = []
    official_train_paths = []
    official_train_speeds = []
    official_val_paths = []
    official_val_speeds = []
    has_official_split = False

    # scan vehicle directories deterministically
    subdirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    
    for subdir in subdirs:
        vehicle_path = os.path.join(data_root, subdir)
        split_file = os.path.join(vehicle_path, 'Train_valid_split.txt')
        if os.path.exists(split_file):
            has_official_split = True
            with open(split_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        base_name, split_type = parts[0], parts[1].lower()
                        wav_path = os.path.join(vehicle_path, base_name + '.wav')
                        if not os.path.exists(wav_path):
                            continue
                        spd = parse_speed_from_filename(wav_path)
                        if spd is not None:
                            all_paths.append(wav_path)
                            all_speeds.append(spd)
                            if 'train' in split_type:
                                official_train_paths.append(wav_path)
                                official_train_speeds.append(spd)
                            else:
                                official_val_paths.append(wav_path)
                                official_val_speeds.append(spd)
        else:
            wavs = sorted([f for f in os.listdir(vehicle_path) if f.endswith('.wav')])
            for w in wavs:
                wav_path = os.path.join(vehicle_path, w)
                spd = parse_speed_from_filename(wav_path)
                if spd is not None:
                    all_paths.append(wav_path)
                    all_speeds.append(spd)

    if not all_paths:
        wavs = sorted([f for f in os.listdir(data_root) if f.endswith('.wav')])
        for w in wavs:
            wav_path = os.path.join(data_root, w)
            spd = parse_speed_from_filename(wav_path)
            if spd is not None:
                all_paths.append(wav_path)
                all_speeds.append(spd)

    all_paths_np = np.array(all_paths)
    all_speeds_np = np.array(all_speeds, dtype=np.float32)

    pin_memory = torch.cuda.is_available()

    # handle fold-based cross-validation
    if fold is not None:
        splits = create_10fold_splits(len(all_paths_np), n_folds=n_folds, seed=seed)
        train_idx, val_idx = splits[fold]
        
        train_ds = VS13Dataset(all_paths_np[train_idx].tolist(), all_speeds_np[train_idx].tolist(), is_training=True, return_complex=return_complex)
        val_ds = VS13Dataset(all_paths_np[val_idx].tolist(), all_speeds_np[val_idx].tolist(), is_training=False, return_complex=return_complex)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader

    # handle official dataset split
    if has_official_split and official_train_paths and official_val_paths:
        train_ds = VS13Dataset(official_train_paths, official_train_speeds, is_training=True, return_complex=return_complex)
        val_ds = VS13Dataset(official_val_paths, official_val_speeds, is_training=False, return_complex=return_complex)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader, val_loader

    # default fallback split
    splits = create_10fold_splits(len(all_paths_np), n_folds=n_folds, seed=seed)
    train_idx, val_idx = splits[0]
    
    train_ds = VS13Dataset(all_paths_np[train_idx].tolist(), all_speeds_np[train_idx].tolist(), is_training=True, return_complex=return_complex)
    val_ds = VS13Dataset(all_paths_np[val_idx].tolist(), all_speeds_np[val_idx].tolist(), is_training=False, return_complex=return_complex)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader
