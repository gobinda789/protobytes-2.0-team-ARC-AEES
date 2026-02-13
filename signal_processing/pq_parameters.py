"""Power quality parameter calculations."""

from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    """RMS value: sqrt(mean(x^2))."""
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x ** 2)))


def active_power(v: np.ndarray, i: np.ndarray) -> float:
    """Active power P = mean(v * i)."""
    v = np.asarray(v)
    i = np.asarray(i)
    return float(np.mean(v * i))


def apparent_power(vrms: float, irms: float) -> float:
    """Apparent power S = Vrms * Irms."""
    return float(vrms * irms)


def power_factor(p: float, s: float) -> float:
    """Power factor PF = P / S (clipped to [-1, 1])."""
    if s <= 1e-12:
        return 0.0
    pf = p / s
    return float(np.clip(pf, -1.0, 1.0))


def crest_factor(x: np.ndarray) -> float:
    """Crest factor = peak / RMS."""
    x = np.asarray(x)
    r = rms(x)
    if r <= 1e-12:
        return 0.0
    return float(np.max(np.abs(x)) / r)


def calculate_sag_swell_fraction(v_rms: float, nominal_rms: float = 230.0) -> float:
    """Calculate sag or swell fraction.
    
    If Vrms < Nominal: Returns Sag fraction (positive value, e.g. 0.2 means 20% sag).
    If Vrms > Nominal: Returns Swell fraction (positive value, e.g. 0.1 means 10% swell).
    If Vrms == Nominal: Returns 0.0.
    
    Formula: |1 - (Vrms / Nominal)|
    """
    if nominal_rms == 0:
        return 0.0
    ratio = v_rms / nominal_rms
    # The user wants "Sag fraction 0.20" etc.
    # Usually Sag 0.2 means remaining voltage is 0.8pu OR voltage dropped by 0.2pu.
    # Standard definition: Sag depth = 1 - (V / Vnom).
    # Swell magnitude = (V / Vnom) - 1.
    
    return float(abs(1.0 - ratio))

