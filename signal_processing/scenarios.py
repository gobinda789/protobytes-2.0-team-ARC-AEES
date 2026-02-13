"""Real-world Power Quality Scenarios.

This module defines presets for generating realistic waveforms to test the analyzer.
It replaces manual slider guessing with engineered data.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from config import SYSTEM_FREQUENCY, NOMINAL_VOLTAGE_RMS, NOMINAL_CURRENT_RMS
from signal_processing.waveform_generator import generate_voltage_wave, generate_current_wave
from signal_processing.disturbances import (
    apply_voltage_sag,
    apply_voltage_swell,
    apply_harmonic_injection,
)


@dataclass
class Scenario:
    name: str
    description: str
    generate_func: callable


def _gen_residential_linear() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Healthy Residential Load (Heater/Bulb)."""
    t, v = generate_voltage_wave(vrms=230.0, frequency_hz=50.0)
    # Linear load, PF=1.0
    _, i = generate_current_wave(load_type="Linear", irms=5.0, frequency_hz=50.0, base_phase_rad=0.0)
    return t, v, i, 50.0


def _gen_industrial_motor_healthy() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Healthy Induction Motor (Lagging PF)."""
    t, v = generate_voltage_wave(vrms=230.0, frequency_hz=50.0)
    # Induction Motor, typically 0.8-0.85 PF lag (~30-36 degrees)
    # 35 degrees = 0.61 rad
    _, i = generate_current_wave(load_type="InductionMotor", irms=12.0, frequency_hz=50.0, base_phase_rad=np.deg2rad(35.0))
    return t, v, i, 50.0


def _gen_industrial_motor_overloaded() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Overloaded Motor (High Current)."""
    t, v = generate_voltage_wave(vrms=228.0, frequency_hz=50.0) # Slight V drop
    # High current, same lag
    _, i = generate_current_wave(load_type="InductionMotor", irms=25.0, frequency_hz=50.0, base_phase_rad=np.deg2rad(35.0))
    return t, v, i, 50.0


def _gen_office_smps_high_thd() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Office Building (Server Room / PCs)."""
    t, v = generate_voltage_wave(vrms=230.0, frequency_hz=50.0)
    # SMPS, high harmonic distortion
    _, i = generate_current_wave(load_type="SMPS", irms=8.0, frequency_hz=50.0)
    return t, v, i, 50.0


def _gen_grid_fault_sag() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Grid Fault: Voltage Sag."""
    t, v = generate_voltage_wave(vrms=230.0, frequency_hz=50.0)
    v = apply_voltage_sag(v, 0.40) # 40% sag
    # Load is linear (resistive) during sag
    _, i = generate_current_wave(load_type="Linear", irms=5.0 * 0.6, frequency_hz=50.0, base_phase_rad=0.0)
    return t, v, i, 50.0


def _gen_grid_harmonic_pollution() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Grid Fault: Background Voltage Harmonics."""
    t, v = generate_voltage_wave(vrms=230.0, frequency_hz=50.0)
    v = apply_harmonic_injection(t, v, h3=0.08, h5=0.05, h7=0.03) # Dirty grid
    # Linear load reflects voltage distortion
    _, i = generate_current_wave(load_type="Linear", irms=5.0, frequency_hz=50.0, base_phase_rad=0.0)
    # Add some current distortion matching voltage
    i = i + (0.08 * i) # Simplified reflection
    return t, v, i, 50.0


SCENARIOS = {
    "Residential Heater (Healthy)": Scenario(
        "Residential Heater",
        "Linear resistive load. Unity PF. No harmonics. Ideal case.",
        _gen_residential_linear
    ),
    "Industrial Motor (Healthy)": Scenario(
        "Industrial Motor",
        "Induction motor running normally. Lagging PF (~0.8). Low harmonics.",
        _gen_industrial_motor_healthy
    ),
    "Industrial Motor (Overloaded)": Scenario(
        "Overloaded Motor",
        "Motor drawing excessive current (>20A). Risk of overheating.",
        _gen_industrial_motor_overloaded
    ),
    "Office IT Equipment (Non-Linear)": Scenario(
        "Office IT Equipment",
        "SMPS loads (PCs, Servers). High odd harmonics (3rd, 5th). Neutral heating risk.",
        _gen_office_smps_high_thd
    ),
    "Grid Fault: Voltage Sag": Scenario(
        "Grid Voltage Sag",
        "Supply voltage drops by 40%. Common during grid switching or faults.",
        _gen_grid_fault_sag
    ),
    "Grid Fault: Harmonic Pollution": Scenario(
        "Grid Harmonic Pollution",
        "Supply voltage is distorted (Dirty Grid). Source-side issue.",
        _gen_grid_harmonic_pollution
    ),
}

def get_scenario_names() -> list[str]:
    return list(SCENARIOS.keys())

def generate_scenario_data(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate t, v, i, freq for the selected scenario."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    return SCENARIOS[name].generate_func()
