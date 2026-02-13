"""FFT-based harmonic analysis.

Uses numpy FFT to compute one-sided spectrum and extract harmonic magnitudes
(1st, 3rd, 5th, 7th) and THD.

We apply a Hann window to reduce spectral leakage.
"""

from __future__ import annotations

import numpy as np

from config import SAMPLING_RATE


def _hann_window(n: int) -> np.ndarray:
    """Hann window."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def compute_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided amplitude spectrum using rFFT.

    Returns:
        freqs: Frequency axis (Hz)
        amps: One-sided amplitude spectrum (same units as x)
    """
    x = np.asarray(x)
    n = x.size
    window = _hann_window(n)
    xw = x * window

    # rFFT
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLING_RATE)

    # Amplitude scaling:
    # For Hann window, coherent gain is mean(window).
    cg = float(np.mean(window))
    amps = (2.0 / (n * cg)) * np.abs(X)
    amps[0] = amps[0] / 2.0  # DC component not doubled
    return freqs, amps


def _nearest_bin(freqs: np.ndarray, target_hz: float, search_hz: float = 2.5) -> int:
    """Find index of the nearest bin to target_hz within a small search band."""
    freqs = np.asarray(freqs)
    target_hz = float(target_hz)
    mask = (freqs >= target_hz - search_hz) & (freqs <= target_hz + search_hz)
    if not np.any(mask):
        # fallback: global nearest
        return int(np.argmin(np.abs(freqs - target_hz)))
    idxs = np.where(mask)[0]
    local = idxs[np.argmin(np.abs(freqs[idxs] - target_hz))]
    return int(local)


def harmonic_magnitudes(
    x: np.ndarray,
    fundamental_hz: float,
) -> dict[str, float]:
    """Extract fundamental and selected odd harmonics from signal x.

    Returns magnitudes (amplitude of sine components) for:
        'H1', 'H3', 'H5', 'H7'
    """
    freqs, amps = compute_spectrum(x)
    h = {}
    for k in (1, 3, 5, 7):
        idx = _nearest_bin(freqs, k * fundamental_hz)
        h[f"H{k}"] = float(amps[idx])
    return h


def thd_percent(
    x: np.ndarray,
    fundamental_hz: float,
) -> float:
    """Compute THD (%) based on extracted harmonic magnitudes.

    THD = sqrt(sum(harmonics^2)) / fundamental * 100
    Here harmonics include 3rd, 5th, 7th.
    """
    mags = harmonic_magnitudes(x, fundamental_hz)
    h1 = mags["H1"]
    if h1 <= 1e-12:
        return 0.0
    num = np.sqrt(mags["H3"] ** 2 + mags["H5"] ** 2 + mags["H7"] ** 2)
    return float((num / h1) * 100.0)


def estimate_frequency(x: np.ndarray, fs: float) -> float:
    """Estimate fundamental frequency using FFT peak.

    Args:
        x: Input signal (1D array)
        fs: Sampling rate (Hz)

    Returns:
        Estimated frequency (Hz)
    """
    x = np.asarray(x)
    n = x.size
    window = _hann_window(n)
    xw = x * window
    
    # Use rFFT
    X = np.abs(np.fft.rfft(xw))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    
    # Ignore DC component
    X[0] = 0
    
    
    # Find peak index
    idx = np.argmax(X)
    
    # Parabolic interpolation for better accuracy
    # If peak is at boundaries, just return it
    if idx == 0 or idx == len(X) - 1:
        return float(freqs[idx])
        
    y_vals = X[idx-1 : idx+2]
    # Parabolic peak shift: d = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    # where alpha=y[idx-1], beta=y[idx], gamma=y[idx+1]
    alpha = y_vals[0]
    beta = y_vals[1]
    gamma = y_vals[2]
    
    denom = alpha - 2 * beta + gamma
    if denom == 0:
        return float(freqs[idx])
        
    d = 0.5 * (alpha - gamma) / denom
    
    # k_peak = idx + d
    bin_width = freqs[1] - freqs[0]
    f_est = freqs[idx] + d * bin_width
    
    return float(f_est)


