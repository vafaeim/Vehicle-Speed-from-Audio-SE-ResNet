import optuna
import re
import os
import sys

def inject_best_params(db_path="sqlite:///optuna_study.db", study_name="se_resnet_vs13_hpo", config_path="src/config.py", params=None):
    if params is not None:
        p = params
    else:
        try:
            study = optuna.load_study(study_name=study_name, storage=db_path)
            p = study.best_params
            print(f"[Inject HPO] Found best params: {p}")
        except Exception as e:
            print(f"[Inject HPO] Could not load study from {db_path}: {e}")
            sys.exit(1)

    if not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(__file__), "config.py")
        if os.path.exists(alt_path):
            config_path = alt_path

    with open(config_path, "r") as f:
        content = f.read()

    # Regex replacements to hardcode the new values
    if "lr" in p:
        content = re.sub(r"INIT_LR\s*=\s*[\d\.e\-]+", f"INIT_LR = {p['lr']}", content)
    if "weight_decay" in p:
        content = re.sub(r"WEIGHT_DECAY\s*=\s*[\d\.e\-]+", f"WEIGHT_DECAY = {p['weight_decay']}", content)
    if "dropout" in p:
        content = re.sub(r"DROPOUT_RATE\s*=\s*[\d\.]+", f"DROPOUT_RATE = {p['dropout']}", content)
        content = re.sub(r"DROPOUT\s*=\s*[\d\.]+", f"DROPOUT = {p['dropout']}", content)
    if "se_ratio" in p:
        content = re.sub(r"SE_RATIO\s*=\s*\d+", f"SE_RATIO = {p['se_ratio']}", content)
        content = re.sub(r"SE_REDUCTION\s*=\s*\d+", f"SE_REDUCTION = {p['se_ratio']}", content)
    if "physics_loss_weight" in p:
        content = re.sub(r"PHYSICS_LOSS_WEIGHT\s*=\s*[\d\.e\-]+", f"PHYSICS_LOSS_WEIGHT = {p['physics_loss_weight']}", content)
        content = re.sub(r"PHYSICS_WEIGHT\s*=\s*[\d\.e\-]+", f"PHYSICS_WEIGHT = {p['physics_loss_weight']}", content)
    elif "physics_weight" in p:
        content = re.sub(r"PHYSICS_LOSS_WEIGHT\s*=\s*[\d\.e\-]+", f"PHYSICS_LOSS_WEIGHT = {p['physics_weight']}", content)
        content = re.sub(r"PHYSICS_WEIGHT\s*=\s*[\d\.e\-]+", f"PHYSICS_WEIGHT = {p['physics_weight']}", content)
    if "cauchy_gamma" in p:
        content = re.sub(r"CAUCHY_GAMMA\s*=\s*[\d\.e\-]+", f"CAUCHY_GAMMA = {p['cauchy_gamma']}", content)
    if "bound_weight" in p:
        content = re.sub(r"BOUND_WEIGHT\s*=\s*[\d\.e\-]+", f"BOUND_WEIGHT = {p['bound_weight']}", content)

    with open(config_path, "w") as f:
        f.write(content)
    
    print("[Inject HPO] Successfully injected optimal hyperparameters into src/config.py!")

if __name__ == "__main__":
    inject_best_params()
