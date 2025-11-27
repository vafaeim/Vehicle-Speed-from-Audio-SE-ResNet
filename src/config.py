import os

class Config:
    # --- Audio Parameters ---
    SAMPLE_RATE = 16000
    DURATION_SECONDS = 10
    AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # --- Training Hyperparameters (HPO-optimized, Trial #22, RMSE=7.54 km/h) ---
    BATCH_SIZE = 32
    EPOCHS = 150
    BASE_FILTERS = 64
    SE_REDUCTION = 8
    DROPOUT = 0.2
    SE_RATIO = 8
    DROPOUT_RATE = 0.10
    WEIGHT_DECAY = 2.93e-5
    INIT_LR = 1.91e-3
    PATIENCE = 30

    # --- Physics-Informed Architecture Parameters ---
    SINC_KERNEL_SIZE = 251
    SINC_STRIDE = 16
    IN_CHANNELS = 1
    KERNEL_SIZE_TIME = 7

    # --- Physics-Informed Loss Hyperparameters ---
    SPEED_MIN = 10.0
    SPEED_MAX = 140.0
    PHYSICS_LOSS_WEIGHT = 0.10
    PHYSICS_WEIGHT = 0.10
    BOUND_WEIGHT = 0.05
    MAX_ACCELERATION = 30.0
    CAUCHY_GAMMA = 5.0
    HUBER_DELTA = 10.0
    LOSS_TYPE = "cauchy"
    
    # --- Augmentation ---
    AUGMENT_PROB = 0.8
    NOISE_SNR_DB = (10, 25)
    GAIN_DB = (-6, 6)
    
    # --- System ---
    SEED = 42
    N_FOLDS = 10
    CHECKPOINT_DIR = "checkpoints"