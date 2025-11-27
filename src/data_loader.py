import tensorflow as tf
import numpy as np
import librosa
import random
from .config import Config

def apply_augmentations(audio):
    """ Applies Gain and Additive White Noise with probability p """
    if random.random() > Config.AUGMENT_PROB:
        return audio
        
    # Gain Augmentation
    gain_db = random.uniform(*Config.GAIN_DB)
    audio *= 10.0 ** (gain_db / 20.0)
    
    # Additive White Noise Augmentation
    snr_db = random.uniform(*Config.NOISE_SNR_DB)
    power = np.sum(audio ** 2) / len(audio)
    if power > 1e-6:
        noise_power = power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        audio += noise
        
    return audio

def preprocess_audio(file_path_bytes, stats_mean, stats_std, is_training):
    """ 
    Core processing logic: Load -> Pad -> Augment -> MelSpec -> Normalize 
    Note: stats_mean and stats_std are passed as numpy arrays
    """
    file_path = file_path_bytes.numpy().decode('utf-8')
    
    try:
        audio, _ = librosa.load(file_path, sr=Config.SAMPLE_RATE, mono=True)
    except Exception:
        # Return zeros on failure to not crash pipeline
        return np.zeros((Config.N_MELS, int(np.ceil(Config.AUDIO_LENGTH_SAMPLES/Config.HOP_LENGTH)), 1), dtype=np.float32)

    # Pad or Truncate
    if len(audio) > Config.AUDIO_LENGTH_SAMPLES:
        audio = audio[:Config.AUDIO_LENGTH_SAMPLES]
    else:
        audio = np.pad(audio, (0, Config.AUDIO_LENGTH_SAMPLES - len(audio)), 'constant')
        
    if is_training:
        audio = apply_augmentations(audio)
        
    # Normalize amplitude
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
        
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=Config.SAMPLE_RATE, 
        n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Z-Score Normalization using pre-calculated stats
    mel_norm = (mel_db - stats_mean) / stats_std
    
    # Add channel dimension (H, W, C)
    return np.expand_dims(mel_norm, axis=-1).astype(np.float32)

def get_tf_dataset(file_paths, labels, stats, is_training=True):
    """ Creates a optimized tf.data pipeline """
    
    # Convert stats to tensor/numpy for the wrapper
    mean_val = np.array(stats['mean'], dtype=np.float32)
    std_val = np.array(stats['std'], dtype=np.float32)
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def _map_fn(path, label):
        # Wrap python function for TF graph
        spec = tf.py_function(
            func=lambda p: preprocess_audio(p, mean_val, std_val, is_training),
            inp=[path], 
            Tout=tf.float32
        )
        
        # Explicitly set shape so Keras knows input dims
        n_frames = int(np.ceil(Config.AUDIO_LENGTH_SAMPLES / Config.HOP_LENGTH))
        spec.set_shape([Config.N_MELS, n_frames, 1])
        return spec, label

    if is_training:
        ds = ds.shuffle(1024)
    
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(Config.BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds