"""
Training entry point for SE-ResNet acoustic vehicle speed estimation.

Loads the VS13 dataset using the official Train_valid_split.txt partition,
computes per-frequency-bin Z-score normalization statistics on the training
split only, persists split metadata for inference.py, and launches 10-Fold
Nested Cross-Validation training via run_cross_validation.

Usage:
    python main.py --data_dir /path/to/vs13
"""

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

    train_paths, train_speeds, train_classes, test_paths, test_speeds, test_classes = get_official_train_test_split(args.data_dir)

    if len(train_paths) == 0:
        print("Error: No audio files found. Check directory structure.")
        sys.exit(1)

    print(f"Dataset Split loaded: {len(train_paths)} Train files, {len(test_paths)} Valid (Test) files.")

    stats = calculate_global_stats(train_paths)

    with open("dataset_splits.json", "w") as f:
        json.dump({
            "test_paths": test_paths.tolist(),
            "stats": stats
        }, f, indent=4)

    run_cross_validation(train_paths, train_speeds, stats)

if __name__ == "__main__":
    main()