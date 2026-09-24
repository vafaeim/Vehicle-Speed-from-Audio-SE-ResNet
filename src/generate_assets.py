#!/usr/bin/env python3
"""
generate_assets.py — Publication-Quality Visual Asset Generator (v2)
====================================================================
Generates 9 high-impact assets (5 PDFs + 4 TikZ CSVs) from REAL data.

Required CLI flags:
  --audio_path       : A single .wav from VS13 dataset
  --data_dir         : Root of VS13 dataset (vehicle subfolders)
  --model_dir        : Directory with 10-fold .pt checkpoints
  --model_path       : Path to a single .pt checkpoint (for SE hook)
  --benchmark_json   : benchmark_results.json from src/benchmark.py
  --ablation_csv     : ablation_results_final.csv
  --robustness_csv   : noise_robustness.csv
  --hpo_db           : optuna_study.db

Outputs (in --output_dir, default: assets/):
  A1  fig_spectrogram_masking.pdf   Clean/20dB/10dB/0dB spectrograms
  A2  fig_waveform_to_spec.pdf      Waveform + spectrogram (2x1)
  A3  fig_se_activation.pdf         SE channel excitation heatmap
  A4  fig_scatter_pred.pdf          Predicted vs True + R² + 95% CI
  A5  fig_error_violin.pdf          Residual violins by vehicle class
  B1  tikz_ablation.csv
  B2  tikz_hardware.csv
  B3  tikz_robustness.csv
  B4  tikz_hpo_history.csv
"""

import argparse
import glob
import json
import os
import sqlite3
import sys

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.models_torch import build_se_resnet
from src.utils import get_official_train_test_split

# ── Publication Style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ── Utilities ──────────────────────────────────────────────────────

def inject_awgn(audio, snr_db):
    """Inject AWGN at a precise physical SNR."""
    if snr_db is None:
        return audio.copy()
    sig_pow = np.mean(audio ** 2)
    if sig_pow < 1e-10:
        return audio.copy()
    noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed=42)
    return audio + rng.normal(0, np.sqrt(noise_pow), len(audio)).astype(audio.dtype)


def compute_mel(audio, sr):
    """Amplitude-preserving log-Mel spectrogram (ref=1.0, no peak norm)."""
    D = np.abs(librosa.stft(y=audio, n_fft=Config.N_FFT,
                            hop_length=Config.HOP_LENGTH, window="hann")) ** 2
    S = librosa.feature.melspectrogram(
        S=D, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS, fmin=0, fmax=sr // 2)
    return librosa.power_to_db(S, ref=1.0)


def load_audio(path, sr=None):
    """Load audio at project sample rate."""
    if sr is None:
        sr = Config.SAMPLE_RATE
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def load_model_from_checkpoint(ckpt_path, device):
    """Load a single SE-ResNet from a .pt checkpoint."""
    model = build_se_resnet(
        input_shape=(1, Config.N_MELS, 313),
        se_ratio=Config.SE_RATIO,
        dropout=Config.DROPOUT_RATE,
        stages=3,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model



def load_dataset_stats(data_dir):
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
            return float(stats["mean"]), float(stats["std"])
    print(f"  [WARN] dataset_stats.json not found in {data_dir}. Using mean=0, std=1.")
    return 0.0, 1.0

def audio_to_tensor(audio, sr, device, data_dir):
    """Convert raw audio to model-ready tensor (1, 1, 128, T) with normalization."""
    if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
        audio = audio[:Config.AUDIO_LENGTH_SAMPLES]
    else:
        audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)))
    mel = compute_mel(audio, sr)
    mean, std = load_dataset_stats(data_dir)
    mel = (mel - mean) / std
    return torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


# ── Ensemble Inference ─────────────────────────────────────────────

def run_ensemble_inference(model_dir, test_paths, test_speeds, data_dir):
    """Load all .pt checkpoints and run real inference on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_paths = sorted(glob.glob(os.path.join(model_dir, "*.pt")))
    if not ckpt_paths:
        raise FileNotFoundError(f"No .pt checkpoints in {model_dir}")
    print(f"  Found {len(ckpt_paths)} checkpoints")

    # Pre-load test audio
    print("  Loading test audio...")
    tensors = []

    mean, std = load_dataset_stats(data_dir)
    for p in test_paths:
        y, sr = load_audio(p)
        if len(y) > Config.AUDIO_LENGTH_SAMPLES:
            y = y[:Config.AUDIO_LENGTH_SAMPLES]
        else:
            y = np.pad(y, (0, Config.AUDIO_LENGTH_SAMPLES - len(y)))
        mel = compute_mel(y, sr)
        mel = (mel - mean) / std
        tensors.append(torch.tensor(mel, dtype=torch.float32).unsqueeze(0))
    X = torch.stack(tensors, dim=0).to(device)

    all_fold_preds = []
    for i, cp in enumerate(ckpt_paths):
        model = load_model_from_checkpoint(cp, device)
        with torch.inference_mode():
            preds = model(X).squeeze(-1).cpu().numpy()
        all_fold_preds.append(preds)
        fold_rmse = np.sqrt(np.mean((test_speeds - preds) ** 2))
        print(f"    Fold {i+1}/{len(ckpt_paths)} RMSE: {fold_rmse:.2f}")

    ens = np.mean(np.array(all_fold_preds), axis=0)
    print(f"  Ensemble RMSE: {np.sqrt(np.mean((test_speeds - ens)**2)):.2f}")
    return ens, test_speeds, all_fold_preds


# ── A1: Spectrogram Masking (1×4) ─────────────────────────────────

def gen_spectrogram_masking(audio_path, out):
    print(f"[A1] {out}")
    y, sr = load_audio(audio_path)
    snrs = [None, 20, 10, 0]
    titles = ["(a) Clean", "(b) SNR = 20 dB",
              "(c) SNR = 10 dB\n(Onset of Masking)",
              "(d) SNR = 0 dB\n(Full Masking)"]

    S_clean = compute_mel(y, sr)
    vmin, vmax = S_clean.min(), S_clean.max()

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for i, (snr, title) in enumerate(zip(snrs, titles)):
        noisy = inject_awgn(y, snr)
        S = compute_mel(noisy, sr)
        ax = axes[i]
        img = librosa.display.specshow(
            S, x_axis="time", y_axis="mel", sr=sr,
            hop_length=Config.HOP_LENGTH, fmin=0, fmax=sr // 2,
            ax=ax, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel Frequency (Hz)" if i == 0 else "")
    fig.colorbar(img, ax=axes, format="%+2.0f dB", pad=0.015, aspect=30,
                 label="Power (dB, ref=1.0)")
    fig.savefig(out); plt.close(fig)
    print(f"     ✓ {out}")


# ── A2: Waveform + Spectrogram (2×1) ──────────────────────────────

def gen_waveform_to_spec(audio_path, out):
    print(f"[A2] {out}")
    y, sr = load_audio(audio_path)
    S = compute_mel(y, sr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5),
                                   gridspec_kw={"height_ratios": [1, 2]},
                                   sharex=False)
    fig.subplots_adjust(hspace=0.35)

    # Waveform
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color="#2196F3", linewidth=0.4, alpha=0.85)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (s)")
    ax1.set_title("(a) Raw Audio Waveform")
    ax1.set_xlim(0, t[-1])

    # Spectrogram
    img = librosa.display.specshow(
        S, x_axis="time", y_axis="mel", sr=sr,
        hop_length=Config.HOP_LENGTH, fmin=0, fmax=sr // 2,
        ax=ax2, cmap="magma")
    ax2.set_title("(b) Amplitude-Preserving Mel-Spectrogram (128 bins)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Mel Frequency (Hz)")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.02,
                 label="Power (dB, ref=1.0)")

    fig.savefig(out); plt.close(fig)
    print(f"     ✓ {out}")


# ── A3: SE Activation Heatmap ──────────────────────────────────────

def gen_se_activation(audio_path, model_path, data_dir, out):
    print(f"[A3] {out}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(model_path, device)
    y, sr = load_audio(audio_path)
    x_tensor = audio_to_tensor(y, sr, device, data_dir)

    # Collect SE excitation weights from ALL SE blocks via hooks
    se_activations = {}

    def make_hook(name):
        def hook_fn(module, inp, output):
            # SqueezeExcitation forward: returns x * e.view(b,c,1,1)
            # We need to re-compute 'e' from the input
            x_in = inp[0]
            b, c, _, _ = x_in.shape
            s = x_in.mean(dim=(2, 3))          # squeeze
            e = module.relu(module.fc1(s))      # excite bottleneck
            e = module.sigmoid(module.fc2(e))   # excite gate
            se_activations[name] = e.detach().cpu().numpy().flatten()
        return hook_fn

    hooks = []
    block_idx = 0
    for stage_i, stage in enumerate(model.stages):
        for blk_j, block in enumerate(stage):
            if hasattr(block, 'se') and hasattr(block.se, 'fc1'):
                name = f"S{stage_i+1}.B{blk_j+1}"
                hooks.append(block.se.register_forward_hook(make_hook(name)))
                block_idx += 1

    with torch.inference_mode():
        _ = model(x_tensor)

    for h in hooks:
        h.remove()

    if not se_activations:
        print("     ⚠ No SE blocks found, skipping.")
        return

    # Build heatmap matrix: rows = blocks, cols = channels
    names = list(se_activations.keys())
    max_ch = max(len(v) for v in se_activations.values())
    mat = np.zeros((len(names), max_ch))
    for i, n in enumerate(names):
        v = se_activations[n]
        mat[i, :len(v)] = v

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="inferno", interpolation="nearest",
                   vmin=0, vmax=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("SE Block")
    ax.set_title("Squeeze-and-Excitation Channel Activation Weights")
    fig.colorbar(im, ax=ax, label="Excitation Weight (σ)", pad=0.02, aspect=25)

    fig.savefig(out); plt.close(fig)
    print(f"     ✓ {out}")


# ── A4: Scatter Prediction Plot ────────────────────────────────────

def gen_scatter_pred(y_true, y_pred, out):
    print(f"[A4] {out}")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    fig, ax = plt.subplots(figsize=(7, 7))
    sns.regplot(x=y_true, y=y_pred, ax=ax,
                scatter_kws={"alpha": 0.65, "s": 45, 
                             "color": "#2196F3"},
                line_kws={"color": "#E53935", "linewidth": 1.8,
                           "label": f"OLS Fit (R² = {r2:.3f})"},
                ci=95)
    lo = min(y_true.min(), y_pred.min()) - 5
    hi = max(y_true.max(), y_pred.max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.7, label="Ideal (y = x)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Ground Truth Speed (km/h)")
    ax.set_ylabel("Predicted Speed (km/h)")
    ax.set_title("10-Fold Ensemble: Predicted vs. Ground Truth")

    txt = f"RMSE = {rmse:.2f} km/h\nMAE  = {mae:.2f} km/h\nR²   = {r2:.3f}\nN    = {len(y_true)}"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            va="top", bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                ec="gray", alpha=0.9), family="monospace")
    ax.legend(loc="lower right")
    fig.savefig(out); plt.close(fig)
    print(f"     ✓ {out}")


# ── A5: Violin Error Plot ─────────────────────────────────────────

def gen_error_violin(y_true, y_pred, classes, out):
    print(f"[A5] {out}")
    df = pd.DataFrame({"Vehicle Class": classes,
                        "Residual Error (km/h)": y_pred - y_true})
    order = (df.groupby("Vehicle Class")["Residual Error (km/h)"]
             .median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.violinplot(data=df, x="Vehicle Class", y="Residual Error (km/h)",
                   order=order, ax=ax, palette="Set2", inner="quartile",
                   density_norm="width", linewidth=1.0, saturation=0.85)
    ax.axhline(0, color="black", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("Vehicle Class (VS13)")
    ax.set_ylabel("Residual Error (km/h)")
    ax.set_title("Residual Error Distribution Across Vehicle Classes")
    ax.tick_params(axis="x", rotation=35)
    for i, cls in enumerate(order):
        n = len(df[df["Vehicle Class"] == cls])
        ax.text(i, ax.get_ylim()[1] * 0.92, f"n={n}", ha="center",
                fontsize=8, color="gray")
    fig.savefig(out); plt.close(fig)
    print(f"     ✓ {out}")


# ── B1: TikZ Ablation CSV ─────────────────────────────────────────

def gen_tikz_ablation(csv_path, out):
    print(f"[B1] {out}")
    df = pd.read_csv(csv_path)
    keep = [c for c in ["experiment_name", "variant_name", "group",
                         "parameter_count", "val_rmse", "val_mae",
                         "use_se", "se_ratio", "stages"] if c in df.columns]
    df_out = df[keep].copy() if keep else df.copy()
    for c in df_out.select_dtypes(include=[np.number]).columns:
        df_out[c] = df_out[c].round(4)
    df_out.to_csv(out, index=False)
    print(f"     ✓ {out} ({len(df_out)} rows)")


# ── B2: TikZ Hardware CSV ─────────────────────────────────────────

def gen_tikz_hardware(json_path, out):
    print(f"[B2] {out}")
    with open(json_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    rename = {"Batch Size": "BatchSize", "Mean Latency (ms)": "LatencyMs",
              "P95 Latency (ms)": "P95Ms", "P99 Latency (ms)": "P99Ms",
              "Throughput (samples/s)": "ThroughputSPS"}
    df.rename(columns=rename, inplace=True)
    df.to_csv(out, index=False)
    print(f"     ✓ {out} ({len(df)} rows)")


# ── B3: TikZ Robustness CSV ───────────────────────────────────────

def gen_tikz_robustness(csv_path, out):
    print(f"[B3] {out}")
    df = pd.read_csv(csv_path)
    # Standardize column names for pgfplots
    clean_cols = {}
    for c in df.columns:
        clean_cols[c] = c.replace(" ", "_").replace("(", "").replace(")", "")
    df.rename(columns=clean_cols, inplace=True)
    df.to_csv(out, index=False)
    print(f"     ✓ {out} ({len(df)} rows)")


# ── B4: TikZ HPO History CSV (from Optuna SQLite) ─────────────────

def gen_tikz_hpo(db_path, out):
    print(f"[B4] {out}")
    conn = sqlite3.connect(db_path)
    query = """
        SELECT t.number AS trial,
               tv.value AS objective_rmse,
               t.state
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY t.number
    """
    df = pd.read_sql_query(query, conn)

    # Add running-best column
    df["best_so_far"] = df["objective_rmse"].cummin()

    # Also extract hyperparameters for the best trial
    best_query = """
        SELECT tp.param_name, tp.param_value
        FROM trial_params tp
        JOIN trial_values tv ON tp.trial_id = tv.trial_id
        WHERE tv.value = (SELECT MIN(value) FROM trial_values)
    """
    best_params = pd.read_sql_query(best_query, conn)
    conn.close()

    print(f"     Total completed trials: {len(df)}")
    print(f"     Best RMSE: {df['objective_rmse'].min():.4f}")
    if not best_params.empty:
        for _, row in best_params.iterrows():
            print(f"       {row['param_name']}: {row['param_value']}")

    df[["trial", "objective_rmse", "best_so_far"]].to_csv(out, index=False)
    print(f"     ✓ {out} ({len(df)} rows)")


# ── Main CLI ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate 9 publication-quality assets from REAL data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (Kaggle):
  python src/generate_assets.py \\
      --audio_path  data/vs13/Audi_A4/a4_50.wav \\
      --data_dir    data/vs13 \\
      --model_dir   checkpoints/ \\
      --model_path  checkpoints/fold_0_best.pt \\
      --benchmark_json benchmark_results.json \\
      --ablation_csv   ablation_results_final.csv \\
      --robustness_csv noise_robustness.csv \\
      --hpo_db         optuna_study.db
""")
    p.add_argument("--audio_path", required=True,
                   help="Single .wav from VS13 for spectrograms & SE hook")
    p.add_argument("--data_dir", required=True,
                   help="VS13 root (vehicle subfolders with Train_valid_split.txt)")
    p.add_argument("--model_dir", required=True,
                   help="Dir with 10-fold .pt checkpoints")
    p.add_argument("--model_path", required=True,
                   help="Single .pt checkpoint for SE activation hook")
    p.add_argument("--benchmark_json", required=True,
                   help="benchmark_results.json from src/benchmark.py")
    p.add_argument("--ablation_csv", required=True,
                   help="Ablation results CSV from ablation_runner.py")
    p.add_argument("--robustness_csv", required=True,
                   help="noise_robustness.csv from noise_robustness.py")
    p.add_argument("--hpo_db", required=True,
                   help="optuna_study.db SQLite database")
    p.add_argument("--output_dir", default="assets",
                   help="Output directory (default: assets/)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    O = lambda name: os.path.join(args.output_dir, name)

    print("=" * 70)
    print(" PUBLICATION ASSET GENERATOR — SE-ResNet Manuscript (v2)")
    print("=" * 70)

    # ── A1: Spectrogram masking ──
    gen_spectrogram_masking(args.audio_path, O("fig_spectrogram_masking.pdf"))

    # ── A2: Waveform + spectrogram ──
    gen_waveform_to_spec(args.audio_path, O("fig_waveform_to_spec.pdf"))

    # ── A3: SE activation heatmap ──
    gen_se_activation(args.audio_path, args.model_path, args.data_dir,
                      O("fig_se_activation.pdf"))

    # ── Run real ensemble inference for A4 & A5 ──
    print("\n[INF] Running 10-fold ensemble on real test set...")
    split = get_official_train_test_split(args.data_dir)
    test_paths, test_speeds, test_classes = split[3], split[4], split[5]
    ens_preds, gt, _ = run_ensemble_inference(
        args.model_dir, test_paths, test_speeds, args.data_dir)

    # ── A4: Scatter plot ──
    gen_scatter_pred(gt, ens_preds, O("fig_scatter_pred.pdf"))

    # ── A5: Violin plot ──
    gen_error_violin(gt, ens_preds, test_classes,
                     O("fig_error_violin.pdf"))

    # ── B1–B4: TikZ CSVs ──
    gen_tikz_ablation(args.ablation_csv, O("tikz_ablation.csv"))
    gen_tikz_hardware(args.benchmark_json, O("tikz_hardware.csv"))
    gen_tikz_robustness(args.robustness_csv, O("tikz_robustness.csv"))
    gen_tikz_hpo(args.hpo_db, O("tikz_hpo_history.csv"))

    # ── Summary ──
    print("\n" + "=" * 70)
    print(" ALL 9 ASSETS GENERATED")
    print("=" * 70)
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith((".pdf", ".csv")):
            sz = os.path.getsize(O(f))
            print(f"  ✓ {f:42s} ({sz/1024:.1f} KB)")
    print(f"\n  Output: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
