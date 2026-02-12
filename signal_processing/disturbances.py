"""Disturbance simulation for power quality events.

Disturbances are applied mathematically to the voltage waveform (and optionally current if needed).
Required disturbances:
- Voltage Sag (20–40% reduction)
- Voltage Swell (20–40% increase)
- Harmonic Injection (3rd, 5th, 7th)
- Frequency deviation (±1 Hz)
"""

from __future__ import annotations

import numpy as np

from config import SYSTEM_FREQUENCY


def apply_voltage_sag(v: np.ndarray, sag_fraction: float) -> np.ndarray:
    """Apply a voltage sag by reducing amplitude.

    sag_fraction: fraction reduction (0.2 to 0.4 means 20% to 40% reduction)
    """
    sag_fraction = float(sag_fraction)
    if not (0.0 < sag_fraction < 1.0):
        raise ValueError("sag_fraction must be between 0 and 1.")
    return v * (1.0 - sag_fraction)


def apply_voltage_swell(v: np.ndarray, swell_fraction: float) -> np.ndarray:
    """Apply a voltage swell by increasing amplitude.

    swell_fraction: fraction increase (0.2 to 0.4 means 20% to 40% increase)
    """
    swell_fraction = float(swell_fraction)
    if swell_fraction <= 0:
        raise ValueError("swell_fraction must be > 0.")
    return v * (1.0 + swell_fraction)


def apply_harmonic_injection(
    t: np.ndarray,
    v: np.ndarray,
    h3: float = 0.0,
    h5: float = 0.0,
    h7: float = 0.0,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Inject voltage harmonics (3rd, 5th, 7th) as a fraction of fundamental peak.

    Args:
        t: Time axis.
        v: Base voltage waveform.
        h3, h5, h7: Harmonic magnitudes as a fraction of fundamental peak (e.g., 0.05 -> 5%).
        phase_rad: Phase offset for harmonics (rad).

    Returns:
        Distorted voltage waveform.
    """
    # Estimate fundamental peak from base waveform amplitude
    v_peak = float(np.max(np.abs(v)))
    w1 = 2.0 * np.pi * SYSTEM_FREQUENCY
    vh = (
        (h3 * v_peak) * np.sin(3.0 * w1 * t + phase_rad)
        + (h5 * v_peak) * np.sin(5.0 * w1 * t + phase_rad)
        + (h7 * v_peak) * np.sin(7.0 * w1 * t + phase_rad)
    )
    return v + vh


def apply_frequency_deviation(
    t: np.ndarray,
    vrms: float,
    frequency_hz: float,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Regenerate a clean sine voltage waveform with deviated frequency."""
    v_peak = vrms * np.sqrt(2.0)
    return v_peak * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad)
