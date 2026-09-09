import argparse
import os
import sys
import torch
import numpy as np
import time

from src.config import Config
from src.data_loader import get_vs13_datasets, create_10fold_splits
from src.models import build_model
from src.train_engine import train_fold, evaluate_ensemble
from src.utils import profile_peak_memory, set_seed

# execute phase 5 verification
def main():
    parser = argparse.ArgumentParser(description="Evaluate Sub-6.5km/h Vehicle Speed Estimation (10-fold CV)")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to VS13 dataset root")
    parser.add_argument('--fast_dev_run', action='store_true', help="Run 1 epoch for profiling")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset not found at {args.data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("PHASE 5: 10-FOLD CROSS-VALIDATION & VRAM PROFILING")
    print("=" * 60)

    # enforce hardware constraints and determinism
    device = Config.DEVICE
    set_seed(Config.SEED)
    print(f"[Info] Executing on Device: {device}")
    
    if args.fast_dev_run:
        print("[Info] Fast Dev Run enabled: Overriding EPOCHS to 1")
        Config.EPOCHS = 1
        Config.N_FOLDS = 2

    start_time = time.time()
    
    # execute 10-fold cv
    fold_results = []
    ensemble_models = []
    
    for fold in range(Config.N_FOLDS):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")
        train_loader, val_loader = get_vs13_datasets(
            data_root=args.data_dir,
            fold=fold,
            n_folds=Config.N_FOLDS,
            seed=Config.SEED,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
        
        # profile memory during first fold
        if fold == 0:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
        model, best_rmse = train_fold(fold, train_loader, val_loader, Config)
        fold_results.append(best_rmse)
        ensemble_models.append(model)
        
        print(f"[Fold {fold + 1}] Best Validation RMSE: {best_rmse:.3f} km/h")
        if device == 'cuda':
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"[Fold {fold + 1}] Peak VRAM Allocated: {vram_mb:.2f} MB")

    # calculate ensembled metrics
    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION & PROFILING")
    print("=" * 60)
    
    # construct final test loader from all data for ensembling
    _, test_loader = get_vs13_datasets(
        data_root=args.data_dir,
        fold=None, 
        n_folds=1, # no split
        seed=Config.SEED,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )

    ensemble_rmse, inference_vram, _ = evaluate_ensemble(ensemble_models, test_loader, Config)
    
    print(f"Total Execution Time: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Average Fold RMSE:    {np.mean(fold_results):.3f} ± {np.std(fold_results):.3f} km/h")
    print(f"Final Ensemble RMSE:  {ensemble_rmse:.3f} km/h (Target: < 6.5 km/h)")
    
    # strict vram verification
    t4_vram_limit_mb = 16 * 1024
    if inference_vram is not None and "MB" in str(inference_vram):
        vram_val = float(str(inference_vram).split()[0])
        print(f"Peak Inference VRAM:  {vram_val:.2f} MB")
        if vram_val < t4_vram_limit_mb:
            print("[✓] PASSED: VRAM well within 16GB T4 limit.")
        else:
            print("[X] FAILED: VRAM exceeded 16GB T4 limit!")

    if ensemble_rmse < 6.5:
        print("\n[✓] PROJECT SUCCESS: Target sub-6.5 km/h RMSE achieved!")
    else:
        print("\n[X] PROJECT FAILURE: Target RMSE not reached.")

if __name__ == "__main__":
    main()
