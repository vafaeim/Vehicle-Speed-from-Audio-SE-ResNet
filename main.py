import argparse
import os
import sys
import json
from sklearn.model_selection import train_test_split
from src.utils import get_all_audio_paths_and_labels, calculate_global_stats
from src.train_engine import run_cross_validation
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Train SE-ResNet for Vehicle Speed Estimation")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the VS13 dataset root directory")
    parser.add_argument('--test_split', type=float, default=0.1, help="Fraction of dataset reserved for held-out test set")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found.")
        sys.exit(1)
        
    print(f"Scanning dataset at {args.data_dir}...")
    paths, speeds = get_all_audio_paths_and_labels(args.data_dir)
    
    if len(paths) == 0:
        print("Error: No audio files found. Check directory structure.")
        sys.exit(1)
        
    print(f"Found {len(paths)} samples total.")
    
    # Create the pure held-out Test Set (10%) to prevent ensemble data leakage
    train_paths, test_paths, train_speeds, test_speeds = train_test_split(
        paths, speeds, test_size=args.test_split, random_state=Config.SEED
    )
    
    print(f"Split: {len(train_paths)} for 10-Fold CV Training, {len(test_paths)} for pure Ensemble Inference.")
    
    # Calculate stats for Z-score ONLY on the training split
    stats = calculate_global_stats(train_paths)
    
    # Save the test set paths and the training stats for inference.py
    with open("dataset_splits.json", "w") as f:
        json.dump({
            "test_paths": test_paths,
            "stats": stats
        }, f, indent=4)
    
    # Run 10-Fold CV Training on the 90% Training split
    run_cross_validation(train_paths, train_speeds, stats)

if __name__ == "__main__":
    main()