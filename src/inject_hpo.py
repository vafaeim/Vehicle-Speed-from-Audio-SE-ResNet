import optuna
import re
import os
import sys
import argparse

def inject_best_params(args):
    # If any manual flags are provided, bypass the database
    if args.lr is not None or args.weight_decay is not None or args.dropout is not None or args.se_ratio is not None:
        p = {}
        if args.lr is not None: p['lr'] = args.lr
        if args.weight_decay is not None: p['weight_decay'] = args.weight_decay
        if args.dropout is not None: p['dropout'] = args.dropout
        if args.se_ratio is not None: p['se_ratio'] = args.se_ratio
        print(f"[Inject HPO] Manual mode active. Bypassing DB. Injecting params: {p}")
    else:
        db_path = "sqlite:///optuna_study.db"
        study_name = "se_resnet_vs13_hpo"
        try:
            study = optuna.load_study(study_name=study_name, storage=db_path)
            p = study.best_params
            print(f"[Inject HPO] Found best params in DB: {p}")
        except Exception as e:
            print(f"[Inject HPO] Could not load study from {db_path}: {e}")
            sys.exit(1)

    config_path = "src/config.py"
    with open(config_path, "r") as f:
        content = f.read()

    # Regex replacements to hardcode the new values
    if "lr" in p:
        content = re.sub(r"INIT_LR\s*=\s*[\d\.e\-]+", f"INIT_LR = {p['lr']}", content)
    if "weight_decay" in p:
        content = re.sub(r"WEIGHT_DECAY\s*=\s*[\d\.e\-]+", f"WEIGHT_DECAY = {p['weight_decay']}", content)
    if "dropout" in p:
        content = re.sub(r"DROPOUT_RATE\s*=\s*[\d\.]+", f"DROPOUT_RATE = {p['dropout']}", content)
    if "se_ratio" in p:
        content = re.sub(r"SE_RATIO\s*=\s*\d+", f"SE_RATIO = {p['se_ratio']}", content)

    with open(config_path, "w") as f:
        f.write(content)
    
    print("[Inject HPO] Successfully injected optimal hyperparameters into src/config.py!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject Hyperparameters into config.py")
    parser.add_argument("--lr", type=float, default=None, help="Manual learning rate to inject")
    parser.add_argument("--weight_decay", type=float, default=None, help="Manual weight decay to inject")
    parser.add_argument("--dropout", type=float, default=None, help="Manual dropout rate to inject")
    parser.add_argument("--se_ratio", type=int, default=None, help="Manual SE ratio to inject")
    
    args = parser.parse_args()
    inject_best_params(args)
