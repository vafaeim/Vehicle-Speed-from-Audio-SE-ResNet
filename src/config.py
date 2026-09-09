# configuration parameters for acoustic vehicle speed estimation

import os
import torch

class Config:
    # audio settings
    SAMPLE_RATE = 16000
    DURATION_SECONDS = 10
    AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    N_BINS = 1025

    # model architecture settings
    IN_CHANNELS = 1
    BASE_FILTERS = 64
    KERNEL_SIZE_TIME = 7
    KERNEL_SIZE_FREQ = 5
    SE_REDUCTION = 8
    DROPOUT = 0.2
    SINC_KERNEL_SIZE = 251
    SINC_STRIDE = 16

    # training hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 3e-4
    INIT_LR = 3e-4
    WEIGHT_DECAY = 1e-4
    SEED = 42
    N_FOLDS = 10
    USE_AMP = True
    CLIP_GRAD_NORM = 5.0
    PATIENCE = 50

    # loss function hyperparameters
    LOSS_TYPE = 'cauchy'
    CAUCHY_GAMMA = 5.0
    HUBER_DELTA = 10.0
    PHYSICS_WEIGHT = 0.1
    MAX_ACCELERATION = 30.0
    SPEED_MIN = 10.0
    SPEED_MAX = 140.0
    BOUND_WEIGHT = 0.05

    # data augmentation settings
    AUGMENT_PROB = 0.8
    GAIN_DB = (-6.0, 6.0)
    NOISE_SNR_DB = (10.0, 25.0)

    # hardware and runtime paths
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2
    PIN_MEMORY = True
    CHECKPOINT_DIR = 'checkpoints'
