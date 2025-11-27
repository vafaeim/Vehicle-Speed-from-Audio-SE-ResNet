import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import os
import tempfile
import numpy as np
import pytest
import torch

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def sample_rate():
    return 16000

@pytest.fixture
def duration_seconds():
    return 10

@pytest.fixture
def audio_length_samples(sample_rate, duration_seconds):
    return sample_rate * duration_seconds

@pytest.fixture
def synthetic_doppler_audio(sample_rate, duration_seconds):
    """
    Generates a synthetic 10-second pass-by acoustic signal.
    Vehicle speed v = 20.0 m/s (72.0 km/h), impact parameter d = 5.0 m,
    fundamental engine frequency f0 = 400.0 Hz, speed of sound c = 343.0 m/s.
    """
    t = np.linspace(-duration_seconds / 2.0, duration_seconds / 2.0, sample_rate * duration_seconds, dtype=np.float32)
    v = 20.0
    d = 5.0
    c = 343.0
    f0 = 400.0
    
    cos_theta = -(v * t) / np.sqrt(d**2 + (v * t)**2)
    f_obs = f0 / (1.0 - (v / c) * cos_theta)
    
    # Cumulative phase integral: phi(t) = 2 * pi * integral(f_obs) dt
    dt = 1.0 / sample_rate
    phase = 2.0 * np.pi * np.cumsum(f_obs) * dt
    waveform = np.sin(phase).astype(np.float32)
    
    # Atmospheric distance attenuation 1 / r(t)
    distance = np.sqrt(d**2 + (v * t)**2)
    attenuation = (d / distance).astype(np.float32)
    waveform = waveform * attenuation
    
    # Target speed in km/h
    target_speed_kmh = float(v * 3.6)
    
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "speed_kmh": target_speed_kmh,
        "v_mps": v,
        "d_m": d,
        "f0": f0,
        "c": c
    }

@pytest.fixture
def synthetic_audio_batch(synthetic_doppler_audio):
    """
    Returns a batch of synthetic waveforms and ground truth labels.
    Batch size = 4, audio length = 160,000 samples.
    """
    batch_size = 4
    base_wave = synthetic_doppler_audio["waveform"]
    base_speed = synthetic_doppler_audio["speed_kmh"]
    
    waves = []
    speeds = []
    for i in range(batch_size):
        scale = 1.0 + 0.1 * i
        w = base_wave * scale
        waves.append(w)
        speeds.append([base_speed + float(i * 5.0)])
        
    x = torch.tensor(np.stack(waves), dtype=torch.float32).unsqueeze(1) # (B, 1, 160000)
    y = torch.tensor(speeds, dtype=torch.float32) # (B, 1)
    return x, y

@pytest.fixture
def temp_dataset_dir(synthetic_doppler_audio):
    """
    Creates a temporary directory with mock audio files following VS13 layout.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        vehicle_dir = os.path.join(tmp_dir, "CitroenC4Picasso")
        os.makedirs(vehicle_dir, exist_ok=True)
        
        # Create 20 mock audio files with speed labels in filename
        for idx in range(20):
            speed = 30.0 + idx * 4.0
            filename = f"CitroenC4Picasso_{int(speed)}_{idx}.wav"
            filepath = os.path.join(vehicle_dir, filename)
            # Write a minimal dummy wave file or byte data
            with open(filepath, "wb") as f:
                # Write 16-bit PCM wave header + synthetic samples
                samples = (synthetic_doppler_audio["waveform"][:16000] * 32767).astype(np.int16)
                import wave
                with wave.open(filepath, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(samples.tobytes())
                    
        yield tmp_dir
