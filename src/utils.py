# utility functions for speed estimation pipeline

import os
import re
import random
import numpy as np
import torch

# set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# parse speed value from audio filename
def parse_speed_from_filename(file_path):
    basename = os.path.basename(file_path)
    match = re.search(r'([a-zA-Z0-9]+)_(\d+)(?:_\d+)?\.wav', basename)
    if match:
        return float(match.group(2))
    return None

# deterministically discover all wav files
def discover_dataset_files(data_root):
    file_list = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.endswith('.wav'):
                file_list.append(os.path.join(root, f))
    file_list.sort()
    return file_list

# parse paths and ground truth speed labels
def get_all_audio_paths_and_labels(data_root):
    all_paths = []
    all_speeds = []
    
    # check for vehicle subdirectories
    subdirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    
    if subdirs:
        for vehicle_folder in subdirs:
            vehicle_path = os.path.join(data_root, vehicle_folder)
            split_file_path = os.path.join(vehicle_path, 'Train_valid_split.txt')
            
            # read official split file if present
            if os.path.exists(split_file_path):
                with open(split_file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            base_name = parts[0]
                            wav_file = os.path.join(vehicle_path, base_name + '.wav')
                            if not os.path.exists(wav_file):
                                continue
                            speed = parse_speed_from_filename(wav_file)
                            if speed is not None:
                                all_paths.append(wav_file)
                                all_speeds.append(speed)
            else:
                # scan direct wav files in vehicle folder
                wav_files = sorted([f for f in os.listdir(vehicle_path) if f.endswith('.wav')])
                for wf in wav_files:
                    wav_file = os.path.join(vehicle_path, wf)
                    speed = parse_speed_from_filename(wav_file)
                    if speed is not None:
                        all_paths.append(wav_file)
                        all_speeds.append(speed)
    else:
        # scan files directly in data root
        wav_files = sorted([f for f in os.listdir(data_root) if f.endswith('.wav')])
        for wf in wav_files:
            wav_file = os.path.join(data_root, wf)
            speed = parse_speed_from_filename(wav_file)
            if speed is not None:
                all_paths.append(wav_file)
                all_speeds.append(speed)
                
    return all_paths, np.array(all_speeds, dtype=np.float32)

# compute root mean squared error
def compute_rmse(predictions, targets):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))

# compute mean absolute error
def compute_mae(predictions, targets):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    return float(np.mean(np.abs(predictions - targets)))

# profile peak hardware memory allocation
def profile_peak_memory(device='cpu'):
    if device == 'cuda' and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        return 0.0
