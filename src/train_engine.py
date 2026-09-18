import os
import copy
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from .config import Config
from .models_torch import build_se_resnet
from .ablation_runner import VS13AblationDataset

def run_cross_validation(all_paths, all_speeds, stats: Dict[str, Any]):
    """
    Executes 10-Fold Cross Validation PyTorch training loop with full in-memory caching.
    Saves the best model for each fold to the checkpoints directory.
    """
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    fold_results = []
    
    paths_np = np.array(all_paths)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mean_val = np.array(stats["mean"], dtype=np.float32)
    std_val = np.array(stats["std"], dtype=np.float32)

    n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    input_shape = (1, Config.N_MELS, n_frames)

    print(f"Pre-loading {len(all_paths)} audio files into memory...", flush=True)
    import librosa
    master_audio = []
    for i, path in enumerate(all_paths):
        if i % 100 == 0: print(f"  Loaded {i}/{len(all_paths)}", flush=True)
        try:
            audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
            if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
                audio = audio[: Config.AUDIO_LENGTH_SAMPLES]
            else:
                audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)), "constant")
        except Exception:
            audio = np.zeros(Config.AUDIO_LENGTH_SAMPLES, dtype=np.float32)
        master_audio.append(audio)

    for fold, (train_idx, val_idx) in enumerate(kf.split(paths_np, all_speeds)):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}", flush=True)
        
        X_train, y_train = paths_np[train_idx].tolist(), all_speeds[train_idx]
        X_val, y_val = paths_np[val_idx].tolist(), all_speeds[val_idx]
        
        audio_train = [master_audio[i] for i in train_idx]
        audio_val = [master_audio[i] for i in val_idx]
        
        # We reuse the highly optimized VS13AblationDataset which caches raw audio
        train_ds = VS13AblationDataset(
            audio_paths=X_train, speeds=y_train, stats_mean=mean_val, stats_std=std_val,
            is_training=True, use_gain=True, use_noise=True, augment_prob=Config.AUGMENT_PROB, preloaded_audio=audio_train
        )
        val_ds = VS13AblationDataset(
            audio_paths=X_val, speeds=y_val, stats_mean=mean_val, stats_std=std_val,
            is_training=False, preloaded_audio=audio_val
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        model = build_se_resnet(
            input_shape=input_shape,
            dropout=Config.DROPOUT_RATE,
            se_ratio=Config.SE_RATIO
        ).to(device)
        
        # HPO optimized defaults are used here
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=Config.INIT_LR, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=Config.EPOCHS * len(train_loader)
        )
        criterion = nn.MSELoss()

        best_val_rmse = float('inf')
        best_state_dict = None
        patience_counter = 0
        
        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0.0
            
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                
                optimizer.zero_grad()
                preds = model(X_b).squeeze(-1)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * X_b.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    preds = model(X_b).squeeze(-1)
                    loss = criterion(preds, y_b)
                    val_loss += loss.item() * X_b.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_rmse = np.sqrt(val_loss)
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 10 == 0 or epoch == Config.EPOCHS - 1:
                print(f"  Epoch {epoch:3d}/{Config.EPOCHS} - Train MSE: {train_loss:.2f} - Val RMSE: {val_rmse:.2f} (Best: {best_val_rmse:.2f})")
                
            if patience_counter >= Config.PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch}")
                break
                
        # Save best fold model
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold+1}_best.pt")
        torch.save({"model_state_dict": best_state_dict}, checkpoint_path)
        
        print(f"Fold {fold+1} Result - Best RMSE: {best_val_rmse:.4f}")
        fold_results.append(best_val_rmse)
        
    print(f"\n{'='*40}")
    print(f"Final Ensemble RMSE: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    print(f"{'='*40}")