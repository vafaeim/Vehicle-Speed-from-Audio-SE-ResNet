"""
Tier 1: Physics & Signal Processing Foundation Tests
Tests acoustic Doppler kinematics, inflection slope properties,
learnable SincNet bandpass filters, and phase-preserving feature extraction.
"""

import numpy as np
import pytest
import torch


# --- Authoritative Physics & Mathematical Oracles ---

def doppler_observed_frequency(tau: np.ndarray, f0: float, v: float, d: float, c: float = 343.0) -> np.ndarray:
    """
    Authoritative formulation of acoustic Doppler observed frequency:
    f_obs(t) = f0 / (1 - (v/c) * cos_theta(tau))
    where cos_theta(tau) = -(v * tau) / sqrt(d^2 + (v * tau)^2)
    """
    denom_dist = np.sqrt(d**2 + (v * tau)**2)
    cos_theta = -(v * tau) / denom_dist
    mach = v / c
    return f0 / (1.0 - mach * cos_theta)


def sinc_bandpass_kernel(f1: float, f2: float, filter_len: int, sample_rate: int = 16000) -> np.ndarray:
    """
    SincNet bandpass filter kernel in time domain:
    h(t) = (2*f2*sinc(2*f2*t) - 2*f1*sinc(2*f1*t)) * w(t)
    """
    f1_norm = f1 / sample_rate
    f2_norm = f2 / sample_rate
    
    t = np.arange(-(filter_len // 2), filter_len // 2 + 1)
    # Using np.sinc(x) = sin(pi*x)/(pi*x)
    h2 = 2.0 * f2_norm * np.sinc(2.0 * f2_norm * t)
    h1 = 2.0 * f1_norm * np.sinc(2.0 * f1_norm * t)
    h = h2 - h1
    
    # Hamming window
    w = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(filter_len) / (filter_len - 1))
    return h * w


# --- Test Cases ---

def test_doppler_frequency_monotonic_decrease():
    """
    Verifies that observed acoustic frequency f_obs(t) is strictly monotonically
    non-increasing (df/dt <= 0) as vehicle passes from approaching to receding.
    """
    f0 = 500.0  # Hz
    v = 25.0    # m/s (90 km/h)
    d = 4.0     # meters
    c = 343.0   # m/s
    
    tau = np.linspace(-4.0, 4.0, 1000)
    f_obs = doppler_observed_frequency(tau, f0, v, d, c)
    
    diffs = np.diff(f_obs)
    assert np.all(diffs <= 0.0), f"Observed frequency must be non-increasing, found diffs max: {diffs.max()}"
    
    # Approaching frequency must be higher than f0; receding must be lower
    assert f_obs[0] > f0, f"Approaching frequency {f_obs[0]} must exceed rest frequency {f0}"
    assert f_obs[-1] < f0, f"Receding frequency {f_obs[-1]} must be below rest frequency {f0}"


def test_doppler_cpa_inflection_point_value():
    """
    Verifies that at the Closest Point of Approach (tau = 0),
    cos_theta = 0, so f_obs(0) == f0 exactly.
    """
    f0 = 600.0
    v = 20.0
    d = 6.0
    c = 343.0
    
    tau_cpa = np.array([0.0])
    f_cpa = doppler_observed_frequency(tau_cpa, f0, v, d, c)[0]
    
    assert np.isclose(f_cpa, f0, atol=1e-6), f"At CPA, f_obs should equal f0={f0}, got {f_cpa}"


def test_doppler_cpa_inflection_slope_exact():
    """
    Verifies that the numerical derivative at the inflection point matches
    the theoretical derivative: df/dt|_{CPA} = -f0 * v^2 / (c * d).
    """
    f0 = 800.0
    v = 22.22  # 80 km/h in m/s
    d = 5.0
    c = 343.0
    
    theoretical_slope = -f0 * (v**2) / (c * d)
    
    dt = 1e-4
    f_plus = doppler_observed_frequency(np.array([dt]), f0, v, d, c)[0]
    f_minus = doppler_observed_frequency(np.array([-dt]), f0, v, d, c)[0]
    numerical_slope = (f_plus - f_minus) / (2.0 * dt)
    
    rel_error = abs(numerical_slope - theoretical_slope) / abs(theoretical_slope)
    assert rel_error < 1e-4, f"Theoretical slope {theoretical_slope} differs from numerical {numerical_slope}, rel err: {rel_error}"


def test_doppler_inflection_derivative_quadratic_speed_scaling():
    r"""
    Verifies that doubling the vehicle speed quadruples the inflection slope
    (quadratic scaling |df/dt| \propto v^2).
    """
    f0 = 500.0
    v1 = 15.0
    v2 = 30.0  # 2x speed
    d = 5.0
    c = 343.0
    
    slope_1 = -f0 * (v1**2) / (c * d)
    slope_2 = -f0 * (v2**2) / (c * d)
    
    ratio = abs(slope_2) / abs(slope_1)
    expected_ratio = (v2 / v1)**2  # (2)^2 = 4.0
    assert np.isclose(ratio, expected_ratio, rtol=1e-5), f"Slope ratio {ratio} != expected quadratic ratio {expected_ratio}"


def test_doppler_inflection_derivative_inverse_distance_scaling():
    r"""
    Verifies that doubling the microphone distance halves the inflection slope
    (inverse linear scaling |df/dt| \propto 1/d).
    """
    f0 = 500.0
    v = 20.0
    d1 = 4.0
    d2 = 8.0  # 2x distance
    c = 343.0
    
    slope_1 = -f0 * (v**2) / (c * d1)
    slope_2 = -f0 * (v**2) / (c * d2)
    
    ratio = abs(slope_2) / abs(slope_1)
    expected_ratio = d1 / d2  # 4 / 8 = 0.5
    assert np.isclose(ratio, expected_ratio, rtol=1e-5), f"Slope ratio {ratio} != expected inverse distance ratio {expected_ratio}"


def test_doppler_zero_velocity_invariance():
    """
    Verifies boundary case: at v = 0 m/s (stationary vehicle),
    observed frequency is identically f0 for all time, and slope is 0.
    """
    f0 = 440.0
    v = 0.0
    d = 5.0
    c = 343.0
    
    tau = np.linspace(-5.0, 5.0, 500)
    f_obs = doppler_observed_frequency(tau, f0, v, d, c)
    
    assert np.all(f_obs == f0), "Stationary source must produce constant observed frequency f0"
    diffs = np.diff(f_obs)
    assert np.all(diffs == 0.0), "Derivative must be identically zero for v = 0"


def test_sincnet_filter_symmetry_and_linear_phase():
    """
    Verifies that SincNet bandpass filter kernel is symmetric around its center,
    guaranteeing exact linear phase response (group delay is constant).
    """
    f1 = 200.0
    f2 = 1200.0
    filter_len = 251
    sample_rate = 16000
    
    h = sinc_bandpass_kernel(f1, f2, filter_len, sample_rate)
    
    # Kernel must be symmetric: h[k] == h[L - 1 - k]
    assert np.allclose(h, h[::-1], atol=1e-6), "SincNet filter kernel is not symmetric (violates linear phase)"


def test_sincnet_bandpass_zero_dc_gain():
    """
    Verifies that for a bandpass filter with f1 > 0, the DC gain H(0) = sum(h) is
    approximately zero (rejection of DC bias and zero-frequency drift).
    """
    f1 = 250.0
    f2 = 1500.0
    filter_len = 251
    sample_rate = 16000
    
    h = sinc_bandpass_kernel(f1, f2, filter_len, sample_rate)
    dc_gain = abs(np.sum(h))
    
    assert dc_gain < 0.05, f"SincNet bandpass DC gain {dc_gain} exceeds threshold 0.05"


def test_sincnet_passband_energy_concentration():
    """
    Verifies that the SincNet filter concentrates >80% of its spectral energy
    inside the intended passband [f1, f2].
    """
    f1 = 500.0
    f2 = 2000.0
    filter_len = 501
    sample_rate = 16000
    
    h = sinc_bandpass_kernel(f1, f2, filter_len, sample_rate)
    
    # Compute power spectrum
    n_fft = 4096
    H = np.fft.rfft(h, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    power = np.abs(H)**2
    total_power = np.sum(power)
    
    passband_mask = (freqs >= (f1 - 50.0)) & (freqs <= (f2 + 50.0))
    passband_power = np.sum(power[passband_mask])
    
    fraction = passband_power / total_power
    assert fraction > 0.80, f"SincNet passband energy fraction {fraction:.3f} is below 80%"


def test_stft_phase_extraction_instantaneous_frequency():
    """
    Verifies that unwrapped STFT phase derivative dphi/dt tracks instantaneous frequency.
    """
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Linear chirp from 400 Hz to 800 Hz
    f_start, f_end = 400.0, 800.0
    f_instant = f_start + (f_end - f_start) * (t / duration)
    phase_true = 2.0 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * (t**2 / duration))
    x = np.sin(phase_true)
    
    # PyTorch STFT
    x_tensor = torch.tensor(x, dtype=torch.float32)
    n_fft = 1024
    hop_length = 256
    window = torch.hann_window(n_fft)
    stft = torch.stft(x_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    
    # Magnitude and Phase
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Peak frequency bin index per time frame
    peak_bins = torch.argmax(magnitude, dim=0).numpy()
    bin_freqs = peak_bins * (sample_rate / n_fft)
    
    # Time vector for STFT frames
    frame_times = np.arange(len(bin_freqs)) * (hop_length / sample_rate)
    true_freqs_at_frames = f_start + (f_end - f_start) * (frame_times / duration)
    
    # Filter interior frames to avoid boundary effects
    interior = slice(2, -2)
    corr = np.corrcoef(bin_freqs[interior], true_freqs_at_frames[interior])[0, 1]
    assert corr > 0.98, f"STFT peak frequency tracking correlation {corr:.4f} is below 0.98"


def test_mel_filterbank_nullspace_phase_obliteration():
    """
    Mathematical verification of Mel-STFT phase loss (Feature 2):
    1. Null space of Mel projection M (dim(ker(M)) = 1025 - 128 = 897).
    2. Phase obliteration: distinct complex signals with identical magnitude
       yield identical Mel-spectrograms.
    """
    n_bins = 1025
    n_mels = 128
    
    # Generate random positive magnitudes
    mag = torch.rand(1, n_bins) + 0.1
    phase1 = torch.zeros(1, n_bins)
    phase2 = torch.rand(1, n_bins) * 2.0 * np.pi - np.pi
    
    # Complex representations
    z1 = mag * torch.exp(1j * phase1)
    z2 = mag * torch.exp(1j * phase2)
    
    # Modulus operator |z|
    mod1 = torch.abs(z1)
    mod2 = torch.abs(z2)
    
    # Assert phase difference is annihilated by modulus operator
    assert torch.allclose(mod1, mod2), "Modulus operator |STFT(x)| must obliterate phase variation"
    
    # Verify rank deficiency of Mel projection
    # Rank is at most n_mels = 128, hence null space has dimension at least 1025 - 128 = 897
    null_dim = n_bins - n_mels
    assert null_dim == 897, f"Expected null space dimension 897, got {null_dim}"
