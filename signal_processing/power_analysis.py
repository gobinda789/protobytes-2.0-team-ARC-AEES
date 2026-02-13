"""Power Quality Analysis Logic – Smart Doctor.

This module implements:
1. Detect Fault Type (Sag, Swell, Harmonics, Low PF, Overcurrent, Inrush).
2. Localize Fault Section (Supply Side / Feeder / Load Side).
3. Predict probable device category causing the issue.
4. Recommend Solutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from config import NOMINAL_VOLTAGE_RMS, NOMINAL_CURRENT_RMS


@dataclass
class Diagnosis:
    status: str          # "Healthy", "Warning", "Critical"
    fault_type: str      # e.g., "Voltage Sag", "High Harmonics"
    fault_section: str   # "Supply Side", "Feeder", "Load Side", "Unknown"
    probable_device: str # e.g., "VFD / Elevator", "Heater", "Unknown"
    recommendation: str  # Actionable advice


def detect_inrush(inrush_ratio: float, threshold: float = 3.0) -> bool:
    """Return True if inrush ratio exceeds threshold (indicates startup event)."""
    return inrush_ratio > threshold


def estimate_fault_section(
    vrms: float,
    irms: float,
    thd_i: float,
    pf: float,
    nominal_v: float = NOMINAL_VOLTAGE_RMS,
) -> str:
    """Estimate fault section based on electrical signatures.

    Logic:
    - Voltage problems (sag/swell) with normal current → Supply Side
    - Voltage AND current problems together → Feeder (losses/impedance)
    - Normal voltage, abnormal current/THD/PF → Load Side
    """
    v_dev = abs(vrms - nominal_v) / nominal_v
    i_dev = abs(irms - NOMINAL_CURRENT_RMS) / NOMINAL_CURRENT_RMS

    voltage_abnormal = v_dev > 0.08   # >8% deviation
    current_abnormal = i_dev > 0.20 or thd_i > 8.0 or pf < 0.85

    if voltage_abnormal and current_abnormal:
        return "Feeder"
    elif voltage_abnormal:
        return "Supply Side"
    elif current_abnormal:
        return "Load Side"
    return "None"


def _guess_device_from_signature(
    thd_i: float,
    pf: float,
    phase_shift: float,
    h3_frac: float,
    h5_frac: float,
    h11_frac: float,
    inrush_ratio: float,
) -> str:
    """Rule-based guess of the probable device category from electrical signatures."""

    # High inrush → motor-type startup
    if inrush_ratio > 3.5:
        if h11_frac > 0.05 or h5_frac > 0.15:
            return "Elevator / VFD Motor"
        if phase_shift > 28:
            return "AC Compressor (Startup)"
        return "Large Induction Motor (Startup)"

    # Very low THD, near-unity PF → resistive
    if thd_i < 3.0 and pf > 0.97:
        return "Heater / Resistive Load"

    # High THD with strong odd harmonics → SMPS / Computer
    if thd_i > 20.0 and h3_frac > 0.25:
        return "Computer / SMPS Load"

    # High THD with dominant H3 → LED
    if thd_i > 15.0 and h3_frac > 0.30 and h5_frac < 0.12:
        return "LED Lighting"

    # VFD signature: strong H5, H7, H11, H13
    if h5_frac > 0.15 and h11_frac > 0.06:
        return "Elevator / VFD Drive"

    # Moderate THD, lagging PF → motor loads
    if pf < 0.90 and phase_shift > 25:
        if thd_i > 8.0:
            return "Cooler / Compressor"
        return "AC / Fan Motor"

    # Low THD, lagging PF → simple motor
    if pf < 0.95 and phase_shift > 15:
        return "Fan / Small Motor"

    return "Mixed / Unidentified Load"


def diagnose_power_quality(
    vrms: float,
    irms: float,
    thd_i: float,
    pf: float,
    phase_shift: float,
    nominal_v: float = NOMINAL_VOLTAGE_RMS,
    inrush_ratio: float = 1.4,
    h3_frac: float = 0.0,
    h5_frac: float = 0.0,
    h11_frac: float = 0.0,
) -> Diagnosis:
    """Analyze electrical parameters and return a diagnosis."""

    SAG_THRESHOLD = 0.9 * nominal_v
    SWELL_THRESHOLD = 1.1 * nominal_v
    THD_CRITICAL = 10.0
    THD_WARNING = 5.0
    PF_POOR = 0.90
    PF_BAD = 0.85

    fault_section = estimate_fault_section(vrms, irms, thd_i, pf, nominal_v)
    device = _guess_device_from_signature(thd_i, pf, phase_shift, h3_frac, h5_frac, h11_frac, inrush_ratio)

    # --- Inrush Detection ---
    if detect_inrush(inrush_ratio):
        return Diagnosis(
            status="Warning",
            fault_type="Inrush Current Detected (Startup Event)",
            fault_section="Load Side",
            probable_device=device,
            recommendation=(
                f"Inrush ratio {inrush_ratio:.1f}x detected. "
                "This is typical during motor/compressor startup. "
                "If frequent, consider Soft Starters or VFDs to limit inrush. "
                "Check thermal protection relay settings."
            ),
        )

    # --- Voltage Analysis ---
    if vrms < SAG_THRESHOLD:
        return Diagnosis(
            status="Critical",
            fault_type="Voltage Sag (Undervoltage)",
            fault_section=fault_section,
            probable_device=device,
            recommendation=(
                "Check upstream transformer tap settings. "
                "Inspect for loose connections. "
                "Consider installing a Voltage Stabilizer or UPS."
            ),
        )

    if vrms > SWELL_THRESHOLD:
        return Diagnosis(
            status="Critical",
            fault_type="Voltage Swell (Overvoltage)",
            fault_section=fault_section,
            probable_device=device,
            recommendation=(
                "Check for sudden load drops nearby. "
                "Verify capacitor bank switching logic. "
                "Contact utility provider."
            ),
        )

    # --- Harmonic Analysis ---
    if thd_i > THD_CRITICAL:
        return Diagnosis(
            status="Critical",
            fault_type="Severe Harmonic Distortion",
            fault_section="Load Side",
            probable_device=device,
            recommendation=(
                f"THD exceeds {THD_CRITICAL}%! "
                "Isolate sensitive equipment. "
                "Install Active Harmonic Filter (AHF) or Passive Tuned Filters. "
                "Verify IEEE 519 compliance at PCC."
            ),
        )

    if thd_i > THD_WARNING:
        return Diagnosis(
            status="Warning",
            fault_type="Harmonic Distortion",
            fault_section="Load Side",
            probable_device=device,
            recommendation=(
                "THD is between 5–10%. Monitor load temperature. "
                "Install Line Reactors (chokes) on VFDs/Drives. "
                "Check cable derating for harmonic heating."
            ),
        )

    # --- Power Factor ---
    if pf < PF_BAD:
        return Diagnosis(
            status="Critical",
            fault_type="Very Low Power Factor",
            fault_section="Load Side",
            probable_device=device,
            recommendation=(
                "High reactive power draw! "
                "Install Capacitor Bank or Automatic Power Factor Correction (APFC) unit. "
                "Check for oversized/underloaded motors."
            ),
        )

    if pf < PF_POOR:
        return Diagnosis(
            status="Warning",
            fault_type="Low Power Factor",
            fault_section="Load Side",
            probable_device=device,
            recommendation=(
                "PF < 0.90. Consider PFC capacitors to avoid utility penalties "
                "and reduce I²R losses."
            ),
        )

    # --- Overcurrent ---
    if irms > 1.2 * NOMINAL_CURRENT_RMS:
        return Diagnosis(
            status="Warning",
            fault_type="Overcurrent / Overload",
            fault_section="Load Side",
            probable_device=device,
            recommendation=(
                "Reduce load. Check for mechanical jams (if motor) or short circuits. "
                "Verify circuit breaker ratings."
            ),
        )

    # --- Healthy ---
    if abs(phase_shift) > 5.0 and pf >= PF_POOR:
        lag_lead = "Lagging" if phase_shift > 0 else "Leading"
        return Diagnosis(
            status="Healthy",
            fault_type=f"Normal Inductive Load ({lag_lead})",
            fault_section="None",
            probable_device=device,
            recommendation=(
                "System is operating efficiently.\n"
                "**Preventive Tip:** Check motor cooling fans and bearing lubrication periodically."
            ),
        )

    return Diagnosis(
        status="Healthy",
        fault_type="Optimal Resistive/Linear Load",
        fault_section="None",
        probable_device=device,
        recommendation=(
            "System is running perfectly.\n"
            "**Preventive Tip:** Regularly torque terminal connections to prevent hotspots."
        ),
    )
