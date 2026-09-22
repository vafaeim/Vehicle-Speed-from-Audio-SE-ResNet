import torch
import numpy as np
import random
import argparse
import os
import glob
from torch.utils.data import Dataset, DataLoader
from src.config import Config
from src.models_torch import build_se_resnet
import librosa
import pandas as pd
import json
import re

class NoiseTestDataset(Dataset):
    def __init__(self, audio_paths, speeds, noise_snr_db, mean, std):
        self.audio_paths = audio_paths
        self.speeds = torch.tensor(speeds, dtype=torch.float32)
        self.noise_snr_db = noise_snr_db
        self.mean = mean
        self.std = std
        
        # Load audio once
        self.audio_data = []
        for path in audio_paths:
            try:
                audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
                if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
                    audio = audio[:Config.AUDIO_LENGTH_SAMPLES]
                else:
                    audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)), "constant")
            except:
                audio = np.zeros(Config.AUDIO_LENGTH_SAMPLES, dtype=np.float32)
            self.audio_data.append(audio)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio = self.audio_data[idx].copy()
        speed = self.speeds[idx]
        
        # Add precise noise level
        if self.noise_snr_db is not None:
            power = np.sum(audio ** 2) / len(audio)
            if power > 1e-6:
                noise_power = power / (10 ** (self.noise_snr_db / 10))
                noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
                audio += noise

        mel = librosa.feature.melspectrogram(
            y=audio, sr=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS,
        )
        mel_db = librosa.power_to_db(mel, ref=1.0)
        mel_db = (mel_db - self.mean) / self.std
        
        return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0), speed

def evaluate_ensemble_noise_curve(model_dir: str, data_dir: str, output_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_paths = glob.glob(os.path.join(model_dir, "*.pt"))
    if not model_paths:
        print(f"Error: No .pt files found in {model_dir}")
        return
        
    print(f"Found {len(model_paths)} models for Ensemble Evaluation.")
    
    
    try:
        import json
        with open("dataset_splits.json", "r") as f:
            splits = json.load(f)
        test_paths_json = splits["test_paths"]
        stats = splits["stats"]
        mean_val = np.array(stats['mean'], dtype=np.float32)
        std_val = np.array(stats['std'], dtype=np.float32)
        
        from src.utils import get_official_train_test_split
        _, _, _, test_paths_official, test_speeds_official, _ = get_official_train_test_split(data_dir)
        
        # We need to map basename to speed, because the absolute paths might be different
        # between Kaggle environments
        basename_to_speed = {os.path.basename(p): s for p, s in zip(test_paths_official, test_speeds_official)}
        
        final_test_paths = []
        final_test_speeds = []
        
        for p in test_paths_json:
            b = os.path.basename(p)
            if b in basename_to_speed:
                final_test_speeds.append(basename_to_speed[b])
                
                # Find correct path in current data_dir
                found = glob.glob(f"{data_dir}/**/{b}", recursive=True)
                if found:
                    final_test_paths.append(found[0])
                else:
                    final_test_paths.append(p)
            else:
                print(f"Warning: could not find speed for {b}")
                
        test_speeds = np.array(final_test_speeds)
        
    except Exception as e:
        print(f"Error loading dataset_splits.json or matching data: {e}")
        return
ensemble = []
    for p in model_paths:
        # Load the architecture with SE (this is the SOTA model)
        # Note: if they pass a baseline no-se model directory, this hardcoded use_se=True will fail.
        # Let's dynamically infer from config or just default to True and let the user override if needed.
        # But this is primarily for the main SOTA model.
        model = build_se_resnet(
            input_shape=(1, Config.N_MELS, 313), 
            use_se=True, 
            se_ratio=getattr(Config, 'SE_RATIO', 16),
            stages=getattr(Config, 'STAGES', 3),
            base_filters=getattr(Config, 'BASE_FILTERS', 96),
            dropout=getattr(Config, 'DROPOUT', 0.3)
        ).to(device)
        model.load_state_dict(torch.load(p, map_location=device))
        model.eval()
        ensemble.append(model)

    snr_levels = [None, 40, 30, 20, 10, 5, 0]
    csv_results = []
    
    print("\n" + "="*70)
    print("CONTINUOUS NOISE ROBUSTNESS CURVE (10-FOLD SOTA ENSEMBLE)")
    print("="*70)
    print(f"{'SNR (dB)':<12} | {'Avg Single RMSE':<20} | {'Ensemble RMSE':<15}")
    print("-" * 70)
    
    for snr in snr_levels:
        dataset = NoiseTestDataset(final_test_paths, test_speeds, snr, mean_val, std_val)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        single_sq_errs = [[] for _ in range(len(ensemble))]
        ens_sq_errs = []
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                
                preds = []
                for i, m in enumerate(ensemble):
                    pred = m(x).view(-1)
                    preds.append(pred)
                    single_sq_errs[i].extend((pred - y).pow(2).cpu().numpy().tolist())
                
                ens_pred = torch.stack(preds).mean(dim=0)
                ens_sq_errs.extend((ens_pred - y).pow(2).cpu().numpy().tolist())
                
        rmses = [float(np.sqrt(np.mean(errs))) for errs in single_sq_errs]
        avg_single_rmse = float(np.mean(rmses))
        ens_rmse = float(np.sqrt(np.mean(ens_sq_errs)))
        
        snr_label = "Clean" if snr is None else f"{snr} dB"
        print(f"{snr_label:<12} | {avg_single_rmse:<20.2f} | {ens_rmse:<15.2f}")
        csv_results.append([snr_label, avg_single_rmse, ens_rmse])

    pd.DataFrame(csv_results, columns=['SNR_dB', 'Single_RMSE', 'Ens_RMSE']).to_csv(output_csv, index=False)
    print(f"\nSaved robustness curve to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/vs13", help="Dataset directory")
    parser.add_argument("--output", type=str, default="noise_robustness.csv", help="Output CSV file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the 10-fold .pt weights")
    args = parser.parse_args()
    
    evaluate_ensemble_noise_curve(args.model_dir, args.data_dir, args.output)
