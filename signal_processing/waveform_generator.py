"""Waveform generation utilities.

This module generates simulated voltage and current waveforms for different load types.
The voltage is a pure sine base (before disturbances).
The current waveform depends on the load class (linear, induction motor, SMPS, LED driver, nonlinear).
"""

from __future__ import annotations

import numpy as np

from config import SAMPLING_RATE, SYSTEM_FREQUENCY, DURATION, NOMINAL_VOLTAGE_RMS, NOMINAL_CURRENT_RMS


def _time_axis() -> np.ndarray:
    """Create a time axis for one record."""
    n = int(SAMPLING_RATE * DURATION)
    return np.arange(n) / SAMPLING_RATE


def generate_voltage_wave(
    vrms: float = NOMINAL_VOLTAGE_RMS,
    frequency_hz: float = SYSTEM_FREQUENCY,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean sinusoidal voltage waveform.

    Args:
        vrms: Desired RMS voltage.
        frequency_hz: System frequency (Hz).
        phase_rad: Phase shift (rad).

    Returns:
        t: Time axis (s).
        v: Voltage samples (V).
    """
    t = _time_axis()
    v_peak = vrms * np.sqrt(2.0)
    v = v_peak * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad)
    return t, v


def generate_current_wave(
    load_type: str,
    irms: float = NOMINAL_CURRENT_RMS,
    frequency_hz: float = SYSTEM_FREQUENCY,
    base_phase_rad: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a current waveform according to load type.

    Load types supported:
        - 'Linear'          : pure sine, near-unity PF
        - 'InductionMotor'  : sine with phase lag + slight 5th harmonic
        - 'SMPS'            : strongly distorted, rich odd harmonics
        - 'LEDDriver'       : 3rd harmonic dominant
        - 'Nonlinear'       : multiple harmonics with moderate distortion

    Args:
        load_type: Load label.
        irms: Target RMS current.
        frequency_hz: Fundamental frequency (Hz).
        base_phase_rad: Optional phase relative to voltage. If None, chosen by load type.

    Returns:
        t: Time axis (s).
        i: Current samples (A), scaled to target RMS.
    """
    t = _time_axis()

    # Choose default phase shifts (typical behaviors)
    if base_phase_rad is None:
        if load_type == "Linear":
            base_phase_rad = np.deg2rad(3.0)   # ~unity PF
        elif load_type == "InductionMotor":
            base_phase_rad = np.deg2rad(35.0)  # lagging
        elif load_type == "SMPS":
            base_phase_rad = np.deg2rad(10.0)  # slight lag/lead depends, keep small
        elif load_type == "LEDDriver":
            base_phase_rad = np.deg2rad(15.0)
        elif load_type == "Nonlinear":
            base_phase_rad = np.deg2rad(20.0)
        else:
            raise ValueError(f"Unknown load_type: {load_type}")

    w1 = 2.0 * np.pi * frequency_hz

    # Build waveform shapes (unscaled)
    if load_type == "Linear":
        i = np.sin(w1 * t - base_phase_rad)

    elif load_type == "InductionMotor":
        # Slight distortion + lagging PF
        i = (
            1.00 * np.sin(w1 * t - base_phase_rad)
            + 0.07 * np.sin(5.0 * w1 * t - 5.0 * base_phase_rad)
        )

    elif load_type == "SMPS":
        # Odd harmonics rich: 3rd, 5th, 7th, 9th, 11th
        i = (
            1.00 * np.sin(w1 * t - base_phase_rad)
            + 0.35 * np.sin(3.0 * w1 * t - 3.0 * base_phase_rad)
            + 0.22 * np.sin(5.0 * w1 * t - 5.0 * base_phase_rad)
            + 0.16 * np.sin(7.0 * w1 * t - 7.0 * base_phase_rad)
            + 0.10 * np.sin(9.0 * w1 * t - 9.0 * base_phase_rad)
            + 0.07 * np.sin(11.0 * w1 * t - 11.0 * base_phase_rad)
        )

    elif load_type == "LEDDriver":
        # 3rd harmonic dominant + some 5th
        i = (
            1.00 * np.sin(w1 * t - base_phase_rad)
            + 0.45 * np.sin(3.0 * w1 * t - 3.0 * base_phase_rad)
            + 0.10 * np.sin(5.0 * w1 * t - 5.0 * base_phase_rad)
        )

    elif load_type == "Nonlinear":
        # Mixed odd harmonics moderate distortion
        i = (
            1.00 * np.sin(w1 * t - base_phase_rad)
            + 0.20 * np.sin(3.0 * w1 * t - 3.0 * base_phase_rad)
            + 0.15 * np.sin(5.0 * w1 * t - 5.0 * base_phase_rad)
            + 0.12 * np.sin(7.0 * w1 * t - 7.0 * base_phase_rad)
        )

    # Scale to target RMS
    current_rms = np.sqrt(np.mean(i ** 2))
    if current_rms <= 1e-12:
        raise RuntimeError("Generated current waveform has near-zero RMS.")
    i = i * (irms / current_rms)
    return t, i
