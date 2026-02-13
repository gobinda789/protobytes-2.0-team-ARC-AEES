"""Waveform generation utilities.

This module generates simulated voltage and current waveforms for different load types.
The voltage is a pure sine base (before disturbances).
The current waveform depends on the load class.

Load types supported:
    Legacy:  Linear, InductionMotor, SMPS, LEDDriver, Nonlinear
    New:     Heater, Fan, Cooler, Computer, Elevator, AC
"""

from __future__ import annotations

import numpy as np

from config import SAMPLING_RATE, SYSTEM_FREQUENCY, DURATION, NOMINAL_VOLTAGE_RMS, NOMINAL_CURRENT_RMS


# ---------------------------------------------------------------------------
# All supported load types
# ---------------------------------------------------------------------------
LOAD_TYPES = [
    "Linear",
    "InductionMotor",
    "SMPS",
    "LEDDriver",
    "Nonlinear",
    "Heater",
    "Fan",
    "Cooler",
    "Computer",
    "Elevator",
    "AC",
]


def _time_axis() -> np.ndarray:
    """Create a time axis for one record."""
    n = int(SAMPLING_RATE * DURATION)
    return np.arange(n) / SAMPLING_RATE


def generate_voltage_wave(
    vrms: float = NOMINAL_VOLTAGE_RMS,
    frequency_hz: float = SYSTEM_FREQUENCY,
    phase_rad: float = 0.0,
    time_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean sinusoidal voltage waveform."""
    if time_array is not None:
        t = time_array
    else:
        t = _time_axis()

    v_peak = vrms * np.sqrt(2.0)
    v = v_peak * np.sin(2.0 * np.pi * frequency_hz * t + phase_rad)
    return t, v


# ---------------------------------------------------------------------------
# Default phase shifts (radians) per load type – represent typical behaviour
# ---------------------------------------------------------------------------
_DEFAULT_PHASE_DEG: dict[str, float] = {
    "Linear":         3.0,   # ~unity PF
    "InductionMotor": 35.0,  # lagging
    "SMPS":           10.0,
    "LEDDriver":      15.0,
    "Nonlinear":      20.0,
    "Heater":         2.0,   # resistive, near-unity PF
    "Fan":            25.0,  # lagging (inductive motor)
    "Cooler":         28.0,  # lagging (compressor motor)
    "Computer":       12.0,  # SMPS-like
    "Elevator":       30.0,  # VFD driven motor
    "AC":             32.0,  # compressor motor with periodic startup
}


def _build_raw_current(
    load_type: str,
    t: np.ndarray,
    w1: float,
    phi: float,
    inrush: bool = False,
) -> np.ndarray:
    """Build an un-scaled current waveform for the given load type.

    Args:
        load_type: Device category.
        t: Time axis.
        w1: Angular frequency = 2*pi*f.
        phi: Phase shift in radians.
        inrush: If True, add a startup inrush spike at the beginning.

    Returns:
        Raw (un-scaled) current waveform.
    """
    # --- Legacy types ---
    if load_type == "Linear":
        i = np.sin(w1 * t - phi)

    elif load_type == "InductionMotor":
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.07 * np.sin(5.0 * w1 * t - 5.0 * phi)
        )

    elif load_type == "SMPS":
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.35 * np.sin(3.0 * w1 * t - 3.0 * phi)
            + 0.22 * np.sin(5.0 * w1 * t - 5.0 * phi)
            + 0.16 * np.sin(7.0 * w1 * t - 7.0 * phi)
            + 0.10 * np.sin(9.0 * w1 * t - 9.0 * phi)
            + 0.07 * np.sin(11.0 * w1 * t - 11.0 * phi)
        )

    elif load_type == "LEDDriver":
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.45 * np.sin(3.0 * w1 * t - 3.0 * phi)
            + 0.10 * np.sin(5.0 * w1 * t - 5.0 * phi)
        )

    elif load_type == "Nonlinear":
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.20 * np.sin(3.0 * w1 * t - 3.0 * phi)
            + 0.15 * np.sin(5.0 * w1 * t - 5.0 * phi)
            + 0.12 * np.sin(7.0 * w1 * t - 7.0 * phi)
        )

    # --- New device-level types ---
    elif load_type == "Heater":
        # Nearly pure resistive – very low harmonics, PF ≈ 1
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.02 * np.sin(3.0 * w1 * t - 3.0 * phi)
        )

    elif load_type == "Fan":
        # Small inductive motor, lagging PF, low THD, slight 3rd/5th
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.06 * np.sin(3.0 * w1 * t - 3.0 * phi)
            + 0.04 * np.sin(5.0 * w1 * t - 5.0 * phi)
        )

    elif load_type == "Cooler":
        # Compressor motor – lagging PF, moderate harmonics
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.08 * np.sin(3.0 * w1 * t - 3.0 * phi)
            + 0.05 * np.sin(5.0 * w1 * t - 5.0 * phi)
            + 0.03 * np.sin(7.0 * w1 * t - 7.0 * phi)
        )

    elif load_type == "Computer":
        # SMPS-based – high THD, strong odd harmonics (similar to SMPS but different PF/phase)
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.38 * np.sin(3.0 * w1 * t - 3.0 * phi)
            + 0.20 * np.sin(5.0 * w1 * t - 5.0 * phi)
            + 0.14 * np.sin(7.0 * w1 * t - 7.0 * phi)
            + 0.08 * np.sin(11.0 * w1 * t - 11.0 * phi)
            + 0.05 * np.sin(13.0 * w1 * t - 13.0 * phi)
        )

    elif load_type == "Elevator":
        # VFD driven motor: high harmonics, variable power profile
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.25 * np.sin(5.0 * w1 * t - 5.0 * phi)
            + 0.18 * np.sin(7.0 * w1 * t - 7.0 * phi)
            + 0.12 * np.sin(11.0 * w1 * t - 11.0 * phi)
            + 0.09 * np.sin(13.0 * w1 * t - 13.0 * phi)
        )

    elif load_type == "AC":
        # Air conditioner compressor – lagging PF, periodic startup behaviour
        i = (
            1.00 * np.sin(w1 * t - phi)
            + 0.10 * np.sin(3.0 * w1 * t - 3.0 * phi)
            + 0.06 * np.sin(5.0 * w1 * t - 5.0 * phi)
            + 0.04 * np.sin(7.0 * w1 * t - 7.0 * phi)
        )
    else:
        raise ValueError(f"Unknown load_type: {load_type}")

    # ---- Inrush spike simulation ----
    if inrush:
        # Add a decaying exponential spike in the first ~2 cycles
        cycle_duration = 1.0 / (w1 / (2.0 * np.pi))
        inrush_duration = 2.0 * cycle_duration
        tau = inrush_duration / 4.0  # time constant
        spike = np.where(t < inrush_duration, 4.0 * np.exp(-t / tau), 0.0)
        i = i + spike

    return i


def generate_current_wave(
    load_type: str,
    irms: float = NOMINAL_CURRENT_RMS,
    frequency_hz: float = SYSTEM_FREQUENCY,
    base_phase_rad: float | None = None,
    time_array: np.ndarray | None = None,
    inrush: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a current waveform according to load type.

    Args:
        load_type: Load label (see LOAD_TYPES).
        irms: Target RMS current.
        frequency_hz: Fundamental frequency (Hz).
        base_phase_rad: Optional phase relative to voltage. If None, chosen by load type.
        time_array: Optional custom time array.
        inrush: If True, simulate startup inrush spike.

    Returns:
        t: Time axis (s).
        i: Current samples (A), scaled to target RMS.
    """
    if time_array is not None:
        t = time_array
    else:
        t = _time_axis()

    if base_phase_rad is None:
        if load_type not in _DEFAULT_PHASE_DEG:
            raise ValueError(f"Unknown load_type: {load_type}")
        base_phase_rad = np.deg2rad(_DEFAULT_PHASE_DEG[load_type])

    w1 = 2.0 * np.pi * frequency_hz
    i = _build_raw_current(load_type, t, w1, base_phase_rad, inrush=inrush)

    # Scale to target RMS
    current_rms = np.sqrt(np.mean(i ** 2))
    if current_rms <= 1e-12:
        raise RuntimeError("Generated current waveform has near-zero RMS.")
    i = i * (irms / current_rms)
    return t, i
