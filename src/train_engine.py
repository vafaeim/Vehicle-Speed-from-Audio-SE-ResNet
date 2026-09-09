# training and validation execution routines

import os
import time
import numpy as np
import torch
import torch.nn as nn
from .config import Config
from .models import build_model
from .losses import PhysicsInformedLoss
from .data_loader import get_vs13_datasets, VS13Dataset
from .utils import compute_rmse, compute_mae, profile_peak_memory

# run one full training epoch
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, device, clip_grad_norm=5.0, use_amp=True):
    model.train()
    total_loss = 0.0
    count = 0
    
    device_obj = torch.device(device)
    device_type = device_obj.type
    amp_enabled = use_amp and (device_type == 'cuda')
    amp_dtype = torch.float16 if device_type == 'cuda' else torch.bfloat16
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        
        # mixed precision forward pass
        with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            pred = model(x)
            loss = criterion(pred, y)
            
        # backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # gradient clipping
        if clip_grad_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            
        # optimizer update
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item() * len(y)
        count += len(y)
        
    return total_loss / max(1, count)

# evaluate model performance on validation set
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    device_obj = torch.device(device)
    device_type = device_obj.type
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            with torch.amp.autocast(device_type=device_type, enabled=False):
                pred = model(x)
                loss = criterion(pred, y)
                
            total_loss += loss.item() * len(y)
            all_preds.append(pred.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    val_loss = total_loss / max(1, len(all_targets))
    val_rmse = compute_rmse(all_preds, all_targets)
    val_mae = compute_mae(all_preds, all_targets)
    
    return {
        'loss': val_loss,
        'rmse': val_rmse,
        'mae': val_mae,
        'predictions': all_preds,
        'targets': all_targets
    }

# configure optimizer parameter groups and learning rate scheduler
def build_optimizer_and_scheduler(model, config=Config, steps_per_epoch=1):
    """
    Constructs an AdamW optimizer with decoupled parameter groups and calibrated OneCycleLR scheduler.
    - SincNet frontend parameters: zero weight decay, differential LR (0.2x).
    - GRU parameters: reduced weight decay (0.5x), differential LR (0.6x).
    - 2D Conv/Linear weights: standard weight decay (1.0x), base LR (1.0x).
    - BatchNorm and biases: zero weight decay, base LR (1.0x).
    """
    sinc_params = []
    gru_params = []
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'frontend' in name or 'sinc' in name:
            sinc_params.append(param)
        elif 'rnn' in name or 'gru' in name:
            gru_params.append(param)
        elif param.ndim <= 1 or name.endswith('.bias') or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    param_groups = [
        {'params': sinc_params, 'lr': config.LEARNING_RATE * 0.2, 'weight_decay': 0.0},
        {'params': gru_params, 'lr': config.LEARNING_RATE * 0.6, 'weight_decay': config.WEIGHT_DECAY * 0.5},
        {'params': decay_params, 'lr': config.LEARNING_RATE, 'weight_decay': config.WEIGHT_DECAY},
        {'params': no_decay_params, 'lr': config.LEARNING_RATE, 'weight_decay': 0.0},
    ]
    
    active_groups = [g for g in param_groups if len(g['params']) > 0]
    optimizer = torch.optim.AdamW(
        active_groups if len(active_groups) > 0 else model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    max_lrs = []
    for g in active_groups:
        if g['params'] is sinc_params:
            max_lrs.append(config.LEARNING_RATE * 0.4)
        elif g['params'] is gru_params:
            max_lrs.append(config.LEARNING_RATE * 1.0)
        else:
            max_lrs.append(config.LEARNING_RATE * 1.5)
            
    if len(max_lrs) == 0:
        max_lrs = config.LEARNING_RATE * 1.5
        
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=config.EPOCHS,
        steps_per_epoch=max(1, steps_per_epoch),
        pct_start=0.10,          # 10% warmup
        div_factor=10.0,         # gentle initial rate (base / 10)
        final_div_factor=100.0,  # final floor prevents gradient freezing
        cycle_momentum=False     # protects AdamW momentum
    )
    
    return optimizer, scheduler

# train single cross-validation fold
def train_fold(fold, train_loader, val_loader, config=Config):
    device = config.DEVICE
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"fold_{fold}_best.pt")
    
    # initialize model and loss
    model = build_model(config).to(device)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    huber_delta = getattr(config, 'HUBER_DELTA', 10.0)
    criterion = PhysicsInformedLoss(
        gamma=config.CAUCHY_GAMMA,
        physics_weight=config.PHYSICS_WEIGHT,
        max_accel=config.MAX_ACCELERATION,
        loss_type=config.LOSS_TYPE,
        huber_delta=huber_delta
    ).to(device)
    
    # configure optimizer and scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        config=config,
        steps_per_epoch=len(train_loader)
    )
    
    device_obj = torch.device(device)
    scaler_device = device_obj.type if device_obj.type == 'cuda' else 'cpu'
    scaler = torch.amp.GradScaler(
        device=scaler_device,
        enabled=(config.USE_AMP and device_obj.type == 'cuda')
    )
    
    best_val_rmse = float('inf')
    epochs_no_improve = 0
    
    # training loop over epochs
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()
        
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            clip_grad_norm=config.CLIP_GRAD_NORM,
            use_amp=config.USE_AMP
        )
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_rmse = val_metrics['rmse']
        
        epoch_time = time.time() - epoch_start
        total_remaining_epochs = (config.N_FOLDS - fold) * config.EPOCHS + (config.EPOCHS - epoch)
        eta_seconds = epoch_time * total_remaining_epochs
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
        
        print(f"  [Epoch {epoch:03d}/{config.EPOCHS:03d}] Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} km/h | Time: {epoch_time:.1f}s | ETA: {eta_str}")
        
        # save best model checkpoint
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.PATIENCE:
                break
                
    # reload best model state
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        
    return model, best_val_rmse

# evaluate ensemble of models across test dataset
def evaluate_ensemble(models, test_loader, config=Config):
    device = config.DEVICE
    fold_predictions = []
    targets = None
    
    for model in models:
        model.eval()
        preds = []
        target_list = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = model(x)
                preds.append(out.detach().cpu().numpy())
                if targets is None:
                    target_list.append(y.detach().cpu().numpy())
        fold_predictions.append(np.concatenate(preds, axis=0))
        if targets is None:
            targets = np.concatenate(target_list, axis=0)
            
    # ensemble predictions across all models
    ensemble_preds = np.mean(np.array(fold_predictions), axis=0)
    ensemble_rmse = compute_rmse(ensemble_preds, targets)
    vram_usage = profile_peak_memory(device)
    
    return ensemble_rmse, vram_usage, ensemble_preds

# run complete 10-fold cross validation pipeline
def run_cross_validation(data_root, config=Config):
    fold_results = []
    models = []
    
    for fold in range(config.N_FOLDS):
        train_loader, val_loader = get_vs13_datasets(
            data_root=data_root,
            fold=fold,
            n_folds=config.N_FOLDS,
            seed=config.SEED,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS
        )
        model, best_rmse = train_fold(fold + 1, train_loader, val_loader, config)
        fold_results.append(best_rmse)
        models.append(model)
        
    final_rmse = float(np.mean(fold_results))
    return models, fold_results, final_rmse
