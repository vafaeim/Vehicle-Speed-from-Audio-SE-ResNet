import argparse
import os
import sys
import json
from src.utils import get_official_train_test_split, calculate_global_stats
from src.train_engine import run_cross_validation
from src.config import Config

def main():
    from src.utils import set_seed
    set_seed(42)
    parser = argparse.ArgumentParser(description="Train SE-ResNet for Vehicle Speed Estimation")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the VS13 dataset root directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} not found.")
        sys.exit(1)
        
    print(f"Scanning dataset at {args.data_dir} for official Train/Valid splits...")
    
    # Use the official split provided by the authors (Train_valid_split.txt)
    train_paths, train_speeds, train_classes, test_paths, test_speeds, test_classes = get_official_train_test_split(args.data_dir)
    
    if len(train_paths) == 0:
        print("Error: No audio files found. Check directory structure.")
        sys.exit(1)
        
    print(f"Dataset Split loaded: {len(train_paths)} Train files, {len(test_paths)} Valid (Test) files.")
    
    # Calculate stats for Z-score ONLY on the training split
    stats = calculate_global_stats(train_paths)
    
    # Save the test set paths and the training stats for inference.py
    with open("dataset_splits.json", "w") as f:
        json.dump({
            "test_paths": test_paths.tolist(),
            "stats": stats
        }, f, indent=4)
    
    # Run 10-Fold CV Training on the Training split
    run_cross_validation(train_paths, train_speeds, stats)

if __name__ == "__main__":
    main()