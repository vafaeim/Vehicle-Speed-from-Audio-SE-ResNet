# Acoustic Vehicle Speed Estimation via Attention-Augmented Residual Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

PyTorch implementation for high-precision vehicle speed estimation from single-channel (monaural) roadside audio. 

*(Note: This repository accompanies a manuscript currently under double-blind peer review. Specific architectural names and repository identifiers have been generalized to maintain review integrity).*

## Overview

This framework addresses acoustic loudness bias in roadside monitoring through scale-invariant spectral normalization, coupled with an attention-augmented residual network designed to extract non-linear kinematic Doppler shifts from 2D Mel-spectrograms under adverse noise.

## Key Features

- **Robust Normalization:** Per-frequency bin temporal Z-score standardization ensuring invariance to volume scaling and background noise floors.
- **Attention Mechanism:** Squeeze-and-Excitation (SE) channel recalibration modules for spatio-temporal feature refinement.
- **Rigorous Validation:** Implements a leakage-free 10-fold nested cross-validation and asynchronous Bayesian hyperparameter optimization (Optuna/TPE).
- **High-Throughput Inference:** Optimized for real-time streaming via PyTorch 2.0 Ahead-Of-Time (AOT) kernel fusion.

## Repository Structure

```text
.
├── checkpoints/              # Pre-trained ensemble weights (.pt files)
├── src/
│   ├── benchmark.py          # Hardware latency profiling (CPU & CUDA AOT)
│   ├── config.py             # Global hyperparameters and path definitions
│   ├── data_loader.py        # Data preprocessing pipeline
│   ├── models_torch.py       # PyTorch network architecture
│   ├── noise_robustness.py   # Colored noise evaluation (White/Pink/Brown)
│   ├── optimize.py           # Asynchronous Bayesian HPO (Optuna)
│   ├── train_engine.py       # Training engine with early stopping
│   ├── ablation_runner.py    # Architectural component ablation
│   └── utils.py              # Dataset splitting and utilities
├── tests/                    # Unit tests
├── main.py                   # Training entry point
├── inference.py              # Evaluation entry point
├── requirements.txt
└── README.md

```

## Getting Started

### 1. Installation

```bash
git clone https://github.com/vafaeim/Vehicle-Speed-from-Audio-SE-ResNet.git
pip install -r requirements.txt

```

### 2. Dataset & Pre-trained Weights

* **Dataset:** Download the [VS13 benchmark](https://slobodan.ucg.ac.me/science/vs13/).
* **Weights:** Download the ensemble checkpoints from **[Google Drive Link](https://drive.google.com/drive/folders/1B5JILfoSLnWQbYVUBp8yXcQon7A8Dr5t?usp=sharing)** and place them in `checkpoints/`.

## Usage

### Run Inference (Ensemble Evaluation)

```bash
python inference.py --data_dir /path/to/vs13 --weights_dir checkpoints/

```

### Train from Scratch (Nested Cross-Validation)

```bash
python main.py --data_dir /path/to/vs13

```

### Run Hardware Benchmark (PyTorch 2.0 AOT)

```bash
python -m src.benchmark

```

### Evaluate Noise Robustness

```bash
python -m src.noise_robustness --model_dir checkpoints/ --data_dir /path/to/vs13

```

## Citation

If you use this code in your research, please cite our paper (currently under review). Full citation details will be updated upon formal acceptance.

## License

MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.
