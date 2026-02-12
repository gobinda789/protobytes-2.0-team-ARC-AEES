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
