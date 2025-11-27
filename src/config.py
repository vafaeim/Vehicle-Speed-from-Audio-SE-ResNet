import os

class Config:
    # --- Audio Parameters ---
    SAMPLE_RATE = 16000
    DURATION_SECONDS = 10
    AUDIO_LENGTH_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 150
    BASE_FILTERS = 96  
    DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 1e-4
    INIT_LR = 5e-4
    PATIENCE = 30
    
    # --- Augmentation ---
    AUGMENT_PROB = 0.8
    NOISE_SNR_DB = (10, 25)
    GAIN_DB = (-6, 6)
    
    # --- System ---
    SEED = 42
    N_FOLDS = 10
    CHECKPOINT_DIR = "checkpoints"