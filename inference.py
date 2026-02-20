import argparse
import os
import time
import numpy as np
import tensorflow as tf
from src.config import Config
from src.data_loader import get_tf_dataset
from src.models import build_se_resnet
from src.utils import get_all_audio_paths_and_labels, calculate_global_stats

def run_ensemble_inference(data_dir, weights_dir):
    # 1. data ingestion
    print(f"[INFO] Scanning dataset at {data_dir}...")
    paths, speeds = get_all_audio_paths_and_labels(data_dir)
    
    if len(paths) == 0:
        raise ValueError("Dataset empty or path incorrect.")

    print(f"[INFO] Processing {len(paths)} files...")
    stats = calculate_global_stats(paths)
    
    # create optimized dataset (parallel prefetch)
    ds = get_tf_dataset(paths, speeds, stats, is_training=False)
    
    # 2. architecture setup
    n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    input_shape = (Config.N_MELS, n_frames, 1)
    
    # 3. ensemble loop
    fold_predictions = []
    fold_times = []
    found_weights = False

    print(f"[INFO] Executing Ensemble Inference (weights: {weights_dir})")
    
    for fold in range(1, Config.N_FOLDS + 1):
        weight_path = os.path.join(weights_dir, f"se-resnet-fold-{fold:02d}.weights.h5")
        
        if not os.path.exists(weight_path):
            print(f"[WARN] Fold {fold} missing at {weight_path}. Skipping.")
            continue
            
        found_weights = True
        start_fold = time.perf_counter()
        
        # build architecture and load weights
        model = build_se_resnet(input_shape)
        model.load_weights(weight_path)
        
        # execution
        preds = model.predict(ds, verbose=0)
        fold_predictions.append(preds.flatten())
        
        fold_duration = time.perf_counter() - start_fold
        fold_times.append(fold_duration)
        
        print(f"   -> Fold {fold:02d} Loaded | Processing Time: {fold_duration:.2f}s")
        
        # manual gc and session clear to prevent O(N) memory leak
        del model
        tf.keras.backend.clear_session()

    if not found_weights:
        raise FileNotFoundError(f"Checkpoints not found in {weights_dir}")

    # 4. aggregation and latency metrics
    fold_predictions = np.array(fold_predictions)
    ensemble_preds = np.mean(fold_predictions, axis=0)
    
    total_inf_time = sum(fold_times)
    avg_per_sample_ms = (total_inf_time / (len(speeds) * len(fold_times))) * 1000
    
    # 5. metrics calculation
    rmse = np.sqrt(np.mean((speeds - ensemble_preds) ** 2))
    
    print("\n" + "="*45)
    print(f"ENSEMBLE EVALUATION COMPLETE")
    print(f"Total Samples:    {len(speeds)}")
    print(f"Avg Latency/File: {avg_per_sample_ms:.2f} ms (CPU)")
    print(f"Final RMSE:       {rmse:.4f} km/h")
    print("="*45 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SE-ResNet Ensemble Inference")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--weights_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    run_ensemble_inference(args.data_dir, args.weights_dir)
