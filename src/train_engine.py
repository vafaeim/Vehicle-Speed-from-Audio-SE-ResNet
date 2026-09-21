import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch.multiprocessing as mp

from .config import Config
from .models import build_se_resnet, Factorized1DNet, build_model
from .losses import PhysicsInformedLoss
from .ablation_runner import VS13AblationDataset

def fold_worker(gpu_id, fold_queue, result_queue, paths_np, all_speeds, master_audio, mean_val, std_val, input_shape):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    # Reduce num_workers since we have multiple active processes
    # Kaggle has 4 cores. 2 workers per process = 4 total workers.
    loaders_workers = 2 
    
    while not fold_queue.empty():
        try:
            task = fold_queue.get(timeout=3)
        except Exception:
            break
            
        fold, train_idx, val_idx = task
        print(f"[{device}] Starting Fold {fold+1}/{Config.N_FOLDS}", flush=True)
        
        X_train, y_train = paths_np[train_idx].tolist(), all_speeds[train_idx]
        X_val, y_val = paths_np[val_idx].tolist(), all_speeds[val_idx]
        
        audio_train = [master_audio[i] for i in train_idx]
        audio_val = [master_audio[i] for i in val_idx]
        
        train_ds = VS13AblationDataset(
            audio_paths=X_train, speeds=y_train, stats_mean=mean_val, stats_std=std_val,
            is_training=True, use_noise=True, augment_prob=Config.AUGMENT_PROB, preloaded_audio=audio_train
        )
        val_ds = VS13AblationDataset(
            audio_paths=X_val, speeds=y_val, stats_mean=mean_val, stats_std=std_val,
            is_training=False, preloaded_audio=audio_val
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
            num_workers=loaders_workers, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
            num_workers=loaders_workers, pin_memory=True, persistent_workers=True
        )
        
        model = build_se_resnet(
            input_shape=input_shape,
            dropout=getattr(Config, 'DROPOUT', getattr(Config, 'DROPOUT_RATE', 0.2)),
            se_ratio=getattr(Config, 'SE_REDUCTION', getattr(Config, 'SE_RATIO', 8))
        ).to(device)
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=Config.INIT_LR, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=Config.EPOCHS * len(train_loader)
        )
        criterion_physics = PhysicsInformedLoss(
            gamma=getattr(Config, "CAUCHY_GAMMA", 5.0),
            physics_weight=getattr(Config, "PHYSICS_LOSS_WEIGHT", getattr(Config, "PHYSICS_WEIGHT", 0.10)),
            bound_weight=getattr(Config, "BOUND_WEIGHT", 0.05),
            max_accel=getattr(Config, "MAX_ACCELERATION", 30.0),
            loss_type=getattr(Config, "LOSS_TYPE", "cauchy"),
            huber_delta=getattr(Config, "HUBER_DELTA", 10.0),
            speed_min=getattr(Config, "SPEED_MIN", 10.0),
            speed_max=getattr(Config, "SPEED_MAX", 140.0),
        ).to(device)
        criterion_eval_mse = nn.MSELoss()

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
                loss = criterion_physics(preds, y_b)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * X_b.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            val_mse_sum = 0.0
            val_samples = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    preds = model(X_b).squeeze(-1)
                    phys_loss = criterion_physics(preds, y_b)
                    val_loss += phys_loss.item() * X_b.size(0)
                    mse_batch = criterion_eval_mse(preds, y_b)
                    val_mse_sum += mse_batch.item() * X_b.size(0)
                    val_samples += X_b.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_rmse = float(np.sqrt(val_mse_sum / max(1, val_samples)))
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 10 == 0 or epoch == Config.EPOCHS - 1:
                print(f"[{device}] Fold {fold+1} Epoch {epoch:3d}/{Config.EPOCHS} - Train PhysLoss: {train_loss:.2f} - Val PhysLoss: {val_loss:.2f} - Val RMSE: {val_rmse:.2f} km/h (Best: {best_val_rmse:.2f} km/h)", flush=True)
                
            if patience_counter >= Config.PATIENCE:
                print(f"[{device}] Fold {fold+1} Early stopping at epoch {epoch}", flush=True)
                break
                
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold+1}_best.pt")
        torch.save({"model_state_dict": best_state_dict}, checkpoint_path)
        print(f"[{device}] Fold {fold+1} Completed - Best RMSE: {best_val_rmse:.4f}", flush=True)
        result_queue.put((fold, best_val_rmse))


def run_cross_validation(all_paths, all_speeds, stats: Dict[str, Any]):
    # Setup MP
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from src.utils import SortedKFold
    kf = SortedKFold(n_splits=Config.N_FOLDS)
    paths_np = np.array(all_paths)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

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

    # Prepare multiprocessing queues
    manager = mp.Manager()
    fold_queue = manager.Queue()
    result_queue = manager.Queue()

    for fold, (train_idx, val_idx) in enumerate(kf.split(paths_np, all_speeds)):
        fold_queue.put((fold, train_idx, val_idx))

    # Detect number of GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        n_gpus = 1 # Fallback
        
    print(f"Starting {n_gpus} dual-GPU workers to train {Config.N_FOLDS} folds in parallel...", flush=True)

    processes = []
    for i in range(n_gpus):
        p = mp.Process(target=fold_worker, args=(
            i % n_gpus, fold_queue, result_queue, paths_np, all_speeds, master_audio, mean_val, std_val, input_shape
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    fold_results_dict = {}
    while not result_queue.empty():
        fold, rmse = result_queue.get()
        fold_results_dict[fold] = rmse

    fold_results = [fold_results_dict[i] for i in range(Config.N_FOLDS)]

    print(f"\n{'='*40}")
    print(f"Final Ensemble RMSE: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    print(f"{'='*40}")