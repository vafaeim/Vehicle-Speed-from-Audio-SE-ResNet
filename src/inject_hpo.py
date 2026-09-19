import optuna
import re
import os
import sys

def inject_best_params():
    db_path = "sqlite:///optuna_study.db"
    study_name = "se_resnet_vs13"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=db_path)
        p = study.best_params
        print(f"[Inject HPO] Found best params: {p}")
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
    inject_best_params()
