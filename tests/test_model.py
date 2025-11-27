import tensorflow as tf
import pytest
from src.models import build_se_resnet, squeeze_excite_block
from src.config import Config

def test_se_block_shape():
    """ Test if SE block maintains tensor shape """
    input_tensor = tf.random.normal((1, 32, 32, 64))
    output_tensor = squeeze_excite_block(input_tensor)
    assert input_tensor.shape == output_tensor.shape

def test_model_build():
    """ Test if the full model builds and outputs a single regression value """
    n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
    input_shape = (Config.N_MELS, n_frames, 1)
    
    model = build_se_resnet(input_shape)
    
    # Create dummy input batch
    dummy_input = tf.random.normal((2, *input_shape))
    output = model(dummy_input)
    
    # Check output shape (Batch_Size, 1)
    assert output.shape == (2, 1)
    # Check parameter count (sanity check for SE-ResNet complexity)
    assert model.count_params() > 100000