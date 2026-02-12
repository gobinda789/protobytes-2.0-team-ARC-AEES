"""Feature extraction for ML.

Features required:
- Vrms
- Irms
- THD
- PF
- H3 magnitude
- H5 magnitude
- H7 magnitude
- Crest factor
"""

from __future__ import annotations

import numpy as np

from config import SYSTEM_FREQUENCY
from signal_processing.pq_parameters import rms, active_power, apparent_power, power_factor, crest_factor
from signal_processing.fft_analysis import harmonic_magnitudes, thd_percent


FEATURE_COLUMNS = [
    "Vrms",
    "Irms",
    "THD_percent",
    "PF",
    "H3_mag",
    "H5_mag",
    "H7_mag",
    "CrestFactor_I",
]


def extract_features(v: np.ndarray, i: np.ndarray, fundamental_hz: float = SYSTEM_FREQUENCY) -> dict[str, float]:
    """Compute all required features from voltage and current waveforms."""
    vr = rms(v)
    ir = rms(i)

    p = active_power(v, i)
    s = apparent_power(vr, ir)
    pf = power_factor(p, s)

    hm = harmonic_magnitudes(i, fundamental_hz)
    thd = thd_percent(i, fundamental_hz)
    cf = crest_factor(i)

    return {
        "Vrms": float(vr),
        "Irms": float(ir),
        "THD_percent": float(thd),
        "PF": float(pf),
        "H3_mag": float(hm["H3"]),
        "H5_mag": float(hm["H5"]),
        "H7_mag": float(hm["H7"]),
        "CrestFactor_I": float(cf),
    }
