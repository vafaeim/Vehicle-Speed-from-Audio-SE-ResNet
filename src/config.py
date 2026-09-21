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
    BASE_FILTERS = 96  
    DROPOUT_RATE = 0.10
    WEIGHT_DECAY = 2.93e-5
    INIT_LR = 1.91e-3
    SE_RATIO = 8
    PATIENCE = 30
    
    # --- Augmentation ---
    AUGMENT_PROB = 0.8
    NOISE_SNR_DB = (20.0, 30.0)
    GAIN_DB = (0, 0)
    
    # --- System ---
    SEED = 42
    N_FOLDS = 10
    CHECKPOINT_DIR = "checkpoints"