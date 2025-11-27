import argparse
import os
import numpy as np
import tensorflow as tf
from src.config import Config
from src.data_loader import get_tf_dataset
from src.models import build_se_resnet
from src.utils import get_all_audio_paths_and_labels, calculate_global_stats

def run_ensemble_inference(data_dir, weights_dir):
    """
    Loads 10 SE-ResNet models, predicts on data, and computes Ensemble RMSE.
    """
    # 1. Setup Data
    print(f"[INFO] Scanning dataset at {data_dir}...")
    paths, speeds = get_all_audio_paths_and_labels(data_dir)
    
    if len(paths) == 0:
        raise ValueError("No audio files found. Check data directory.")

    # Note: In a real scenario, use pre-calculated stats from training. 
    # Here we recalculate to ensure pipeline consistency for the user.
    print("[INFO] Calculating normalization statistics...")
    stats = calculate_global_stats(paths)
    
    # Create Dataset (No shuffle, No augmentation)
    ds = get_tf_dataset(paths, speeds, stats, is_training=False)
    
    # 2. Prepare Model Architecture
    n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    input_shape = (Config.N_MELS, n_frames, 1)
    
    # 3. Ensemble Prediction Loop
    fold_predictions = []
    found_weights = False

    print(f"[INFO] Starting Ensemble Inference using weights from {weights_dir}...")
    
    for fold in range(1, Config.N_FOLDS + 1):
        weight_path = os.path.join(weights_dir, f"fold_{fold}_best_weights.weights.h5")
        
        if not os.path.exists(weight_path):
            print(f"[WARN] Weights for Fold {fold} not found at {weight_path}. Skipping.")
            continue
            
        found_weights = True
        print(f"   -> Loading Fold {fold}...")
        
        # Build fresh model and load weights
        model = build_se_resnet(input_shape)
        model.load_weights(weight_path)
        
        # Predict
        preds = model.predict(ds, verbose=0)
        fold_predictions.append(preds.flatten())
        
        # Cleanup to save memory
        del model
        tf.keras.backend.clear_session()

    if not found_weights:
        raise FileNotFoundError("No weight files found. Please check checkpoints directory.")

    # 4. Ensemble Aggregation (Mean)
    print("[INFO] Aggregating predictions...")
    fold_predictions = np.array(fold_predictions)
    ensemble_preds = np.mean(fold_predictions, axis=0)
    
    # 5. Metrics
    mse = np.mean((speeds - ensemble_preds) ** 2)
    rmse = np.sqrt(mse)
    
    print("\n" + "="*40)
    print(f"FINAL ENSEMBLE RESULTS")
    print(f"files processed: {len(speeds)}")
    print(f"RMSE: {rmse:.4f} km/h")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for SE-ResNet Ensemble")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to VS13 dataset")
    parser.add_argument('--weights_dir', type=str, default='checkpoints', help="Directory containing .weights.h5 files")
    
    args = parser.parse_args()
    run_ensemble_inference(args.data_dir, args.weights_dir)