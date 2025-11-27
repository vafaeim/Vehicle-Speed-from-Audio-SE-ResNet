# Acoustic Vehicle Speed Estimation (SE-ResNet)

## Project Overview

This repository hosts the official TensorFlow source code for our acoustic vehicle speed estimation framework. The project implements a deep learning pipeline that processes single-channel audio recordings to estimate vehicle speed.

**Key Features:**

* **Model:** Custom **Squeeze-and-Excitation Residual Network (SE-ResNet)**.

* **Validation:** Full **10-Fold Cross-Validation** engine with ensemble inference.

* **Performance:** RMSE of **7.29 km/h** on the VS13 benchmark.

## Repository Structure

.
├── checkpoints/
│   ├── fold_1_best_weights.weights.h5
│   ├── ...
│   └── fold_10_best_weights.weights.h5
├── src/
│   ├── config.py           # Hyperparameters
│   ├── data_loader.py      # Data pipeline
│   ├── models.py           # Model architecture
│   └── utils.py            # Utilities
├── tests/                  # Unit tests
├── main.py                 # Training entry point
├── inference.py            # Inference entry point
├── requirements.txt
└── README.md

## Getting Started

### 1. Installation

git clone https://github.com/vafaeim/Vehicle-Speed-from-Audio-SE-ResNet.git
cd Vehicle-Speed-from-Audio-SE-ResNet
pip install -r requirements.txt

### 2. Download Pre-trained Models

We provide the official ensemble weights (10 folds) achieving the reported results.

1. **Download** the weights from **[Google Drive Link](https://drive.google.com/drive/folders/1B5JILfoSLnWQbYVUBp8yXcQon7A8Dr5t?usp=sharing)**.

2. **Extract** (or move) the `.h5` files into the `checkpoints/` directory inside the project root.

### 3. Usage

#### Run Inference (Evaluation)

To evaluate the pre-trained ensemble model on the dataset:

python inference.py --data_dir /path/to/vs13 --weights_dir checkpoints/

#### Train from Scratch

To reproduce the training process:

python main.py --data_dir /path/to/vs13

## Performance Metrics

| Model Architecture | Input | RMSE (km/h) | 
| :----- | :----- | :----- | 
| **SE-ResNet (Ensemble)** | **Mel Spectrogram** | **7.29** | 

## License

MIT License. See `LICENSE` file for details.