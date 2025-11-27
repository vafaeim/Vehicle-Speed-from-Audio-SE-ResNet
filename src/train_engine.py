import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from .config import Config
from .data_loader import get_tf_dataset
from .models import build_se_resnet
import os

def run_cross_validation(all_paths, all_speeds, stats):
    """
    Executes 10-Fold Cross Validation training loop.
    """
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    fold_results = []
    
    paths_np = np.array(all_paths)
    
    # Determine input shape based on config
    n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    input_shape = (Config.N_MELS, n_frames, 1)
    
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(paths_np, all_speeds)):
        print(f"\n{'='*20} Fold {fold+1}/{Config.N_FOLDS} {'='*20}")
        
        # Split data
        X_train, y_train = paths_np[train_idx], all_speeds[train_idx]
        X_val, y_val = paths_np[val_idx], all_speeds[val_idx]
        
        # Create Datasets
        train_ds = get_tf_dataset(X_train, y_train, stats, is_training=True)
        val_ds = get_tf_dataset(X_val, y_val, stats, is_training=False)
        
        # Build Model within Strategy Scope (for GPU acceleration)
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            model = build_se_resnet(input_shape)
            
            # Cosine Decay Scheduler
            steps_per_epoch = len(X_train) // Config.BATCH_SIZE
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=Config.INIT_LR,
                decay_steps=steps_per_epoch * Config.EPOCHS,
                alpha=0.0
            )
            
            # AdamW Optimizer
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=Config.WEIGHT_DECAY
            )
            
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
            )
        
        # Callbacks
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold+1}_best.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=Config.PATIENCE, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0
            )
        ]
        
        # Train
        history = model.fit(
            train_ds,
            epochs=Config.EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=2 # Less noisy output
        )
        
        # Evaluate best model
        loss, rmse = model.evaluate(val_ds, verbose=0)
        print(f"Fold {fold+1} Result - RMSE: {rmse:.4f}")
        fold_results.append(rmse)
        
        # Clear session to free GPU memory
        tf.keras.backend.clear_session()
        
    print(f"\n{'='*40}")
    print(f"Final Ensemble RMSE: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    print(f"{'='*40}")