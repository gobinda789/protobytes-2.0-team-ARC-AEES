"""Dataset generation.

Generates labeled samples by simulating:
- a load type (drives current waveform shape)
- a disturbance on voltage (sag/swell/harmonics/frequency deviation/none)

Then extracts power-quality + harmonic features and saves to CSV.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import (
    SYSTEM_FREQUENCY,
    NOMINAL_VOLTAGE_RMS,
    NOMINAL_CURRENT_RMS,
)
from signal_processing.waveform_generator import generate_voltage_wave, generate_current_wave
from signal_processing.disturbances import (
    apply_voltage_sag,
    apply_voltage_swell,
    apply_harmonic_injection,
    apply_frequency_deviation,
)
from data.feature_extraction import FEATURE_COLUMNS, extract_features


LOAD_TYPES = ["Linear", "InductionMotor", "SMPS", "LEDDriver", "Nonlinear"]
DISTURBANCES = ["None", "Sag", "Swell", "Harmonics", "FreqDev"]


@dataclass(frozen=True)
class SampleConfig:
    load_type: str
    disturbance: str
    sag_fraction: float | None = None
    swell_fraction: float | None = None
    h3: float | None = None
    h5: float | None = None
    h7: float | None = None
    freq_hz: float | None = None


def _random_sample_config(rng: np.random.Generator) -> SampleConfig:
    """Create a random sample configuration within required ranges."""
    load = rng.choice(LOAD_TYPES)
    dist = rng.choice(DISTURBANCES, p=[0.35, 0.20, 0.15, 0.20, 0.10])  # bias to none/sag/harmonics

    if dist == "Sag":
        return SampleConfig(load, dist, sag_fraction=float(rng.uniform(0.20, 0.40)))
    if dist == "Swell":
        return SampleConfig(load, dist, swell_fraction=float(rng.uniform(0.20, 0.40)))
    if dist == "Harmonics":
        # Inject modest voltage harmonics (not too large)
        return SampleConfig(
            load,
            dist,
            h3=float(rng.uniform(0.02, 0.08)),
            h5=float(rng.uniform(0.01, 0.06)),
            h7=float(rng.uniform(0.01, 0.05)),
        )
    if dist == "FreqDev":
        return SampleConfig(load, dist, freq_hz=float(rng.uniform(SYSTEM_FREQUENCY - 1.0, SYSTEM_FREQUENCY + 1.0)))
    return SampleConfig(load, "None")


def generate_one_sample(rng: np.random.Generator) -> tuple[dict[str, float], SampleConfig]:
    """Generate one labeled training sample."""
    cfg = _random_sample_config(rng)

    # Small realistic operating variation
    vrms = float(rng.uniform(0.92, 1.08) * NOMINAL_VOLTAGE_RMS)
    irms = float(rng.uniform(0.70, 1.25) * NOMINAL_CURRENT_RMS)

    # Voltage: base sine at nominal frequency (or deviated)
    t, v = generate_voltage_wave(vrms=vrms, frequency_hz=SYSTEM_FREQUENCY)

    # Apply disturbance to voltage
    if cfg.disturbance == "Sag":
        v = apply_voltage_sag(v, cfg.sag_fraction or 0.3)
    elif cfg.disturbance == "Swell":
        v = apply_voltage_swell(v, cfg.swell_fraction or 0.3)
    elif cfg.disturbance == "Harmonics":
        v = apply_harmonic_injection(t, v, h3=cfg.h3 or 0.05, h5=cfg.h5 or 0.03, h7=cfg.h7 or 0.02)
    elif cfg.disturbance == "FreqDev":
        # Frequency deviation regenerates the base sine at different f
        v = apply_frequency_deviation(t, vrms=vrms, frequency_hz=cfg.freq_hz or SYSTEM_FREQUENCY)

    # Current waveform depends on load type, and uses same frequency as voltage in case of freq deviation
    f_i = cfg.freq_hz if cfg.disturbance == "FreqDev" and cfg.freq_hz is not None else SYSTEM_FREQUENCY
    _, i = generate_current_wave(load_type=cfg.load_type, irms=irms, frequency_hz=float(f_i))

    feats = extract_features(v, i, fundamental_hz=float(f_i))
    return feats, cfg


def generate_dataset(
    out_csv_path: str,
    n_samples: int = 900,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate and save dataset to CSV.

    Args:
        out_csv_path: Output CSV path.
        n_samples: Number of samples (>= 800 required).
        seed: Random seed.

    Returns:
        DataFrame with features + label columns.
    """
    if n_samples < 800:
        raise ValueError("n_samples must be at least 800.")

    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_samples):
        feats, cfg = generate_one_sample(rng)
        row = {**feats, "LoadType": cfg.load_type, "Disturbance": cfg.disturbance}
        rows.append(row)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["LoadType", "Disturbance"])
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return df
