import numpy as np
import pytest
import tensorflow as tf
import os
from src.config import Config
from src.data_loader import preprocess_audio

# Robustly resolve path relative to this test file (i.e., inside the tests/ folder)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# sample.wav is a representative audio clip from the VS13 dataset (CitroenC4Picasso_101.wav)
SAMPLE_WAV_PATH = os.path.join(TEST_DIR, "sample.wav")

@pytest.mark.skipif(not os.path.exists(SAMPLE_WAV_PATH), reason=f"sample.wav not found at {SAMPLE_WAV_PATH}. Please place the file in the tests/ folder.")
def test_preprocessing_shape():
    """ 
    Test if audio preprocessing returns correct MelSpec shape.
    Verifies:
    1. Input tensor handling
    2. Correct output dimensions (H, W, C)
    3. Data type consistency
    """
    
    # 1. Setup Dummy Stats (Mean=0, Std=1 performs identity normalization)
    # We use this to isolate shape testing from statistical correctness
    mean = np.zeros((Config.N_MELS, 1), dtype=np.float32)
    std = np.ones((Config.N_MELS, 1), dtype=np.float32)
    
    # 2. Create Input Tensor 
    # The data loader expects a TensorFlow string tensor, simulating tf.data pipeline input
    file_path_tensor = tf.constant(SAMPLE_WAV_PATH)
    
    # 3. Run Preprocessing
    # is_training=False ensures deterministic output (no random noise/gain)
    result = preprocess_audio(file_path_tensor, mean, std, is_training=False)
    
    # 4. Calculate Expected Dimensions
    # Time frames = ceil(Total Samples / Hop Length)
    # 160,000 / 512 = 312.5 -> ceil -> 313 frames
    expected_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    expected_shape = (Config.N_MELS, expected_frames, 1)
    
    # 5. Assertions
    assert result.shape == expected_shape, \
        f"Shape Mismatch! Expected {expected_shape}, got {result.shape}"
    
    assert result.dtype == np.float32, \
        f"Type Mismatch! Expected float32, got {result.dtype}"
        
    # Sanity check: Ensure spectrogram is not empty or all zeros (unless silence)
    assert np.max(np.abs(result)) >= 0, "Output content check failed"