"""Power Quality Analysis Logic.

This module implements the "Smart Doctor" logic to:
1. Detect Fault Type (Sag, Swell, Harmonics, Low PF).
2. Localize Fault (Source Side vs. Load Side).
3. Recommend Solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from config import NOMINAL_VOLTAGE_RMS, NOMINAL_CURRENT_RMS


@dataclass
class Diagnosis:
    status: str          # "Healthy", "Warning", "Critical"
    fault_type: str      # e.g., "Voltage Sag", "High Harmonics"
    fault_source: str    # "Grid Side", "Load Side", "Unknown"
    recommendation: str  # Actionable advice


def diagnose_power_quality(
    vrms: float,
    irms: float,
    thd_i: float,
    pf: float,
    phase_shift: float,
    nominal_v: float = NOMINAL_VOLTAGE_RMS,
) -> Diagnosis:
    """Analyze electrical parameters and return a diagnosis."""
    
    # Thresholds (IEEE 519 & User Specs)
    SAG_THRESHOLD = 0.9 * nominal_v
    SWELL_THRESHOLD = 1.1 * nominal_v
    
    # THD Limits (IEEE 519: <5% Good, 5-8% Warning, >8-10% Critical)
    THD_CRITICAL = 8.0   # Strict industrial limit (User asked for >10, but 8 is safer. Let's start with 8 as critical, or maybe 10 for "Severe"?)
    # User said: "5-8% warning, >10% critical"
    THD_CRITICAL = 10.0
    THD_WARNING = 5.0

    # Power Factor
    # User said 0.949 is "acceptable". Standard is >0.9.
    PF_POOR = 0.90 
    PF_BAD = 0.85

    # 1. Voltage Analysis (Grid Side Issues)
    if vrms < SAG_THRESHOLD:
        return Diagnosis(
            status="Critical",
            fault_type="Voltage Sag (Undervoltage)",
            fault_source="Grid Side / Source",
            recommendation="Check upstream transformer tap settings. Inspect for loose connections. Consider installing a Voltage Stabilizer or UPS."
        )
    
    if vrms > SWELL_THRESHOLD:
        return Diagnosis(
            status="Critical",
            fault_type="Voltage Swell (Overvoltage)",
            fault_source="Grid Side / Source",
            recommendation="Check for sudden load drops nearby. Verify capacitor bank switching logic. Contact utility provider."
        )

    # 2. Harmonic Analysis (Load vs Grid)
    if thd_i > THD_CRITICAL:
        return Diagnosis(
            status="Critical",
            fault_type="Severe Harmonic Distortion",
            fault_source="Load Side",
            recommendation="THD exceeds 10%! Isolate sensitive equipment. Install Active Harmonic Filter (AHF) or Passive Tuned Filters immediately."
        )
    
    if thd_i > THD_WARNING:
        return Diagnosis(
            status="Warning",
            fault_type="Harmonic Distortion",
            fault_source="Load Side",
            recommendation="THD is between 5-10%. Monitor load temperature. Install Line Reactors (chokes) on VFDs/Drives. Verify IEEE 519 compliance."
        )

    # 3. Power Factor Analysis (Efficiency)
    if pf < PF_BAD:
        return Diagnosis(
            status="Critical",
            fault_type="Very Low Power Factor",
            fault_source="Load Side (Inductive)",
            recommendation="High reactive power draw! Install Capacitor Bank or Automatic Power Factor Correction (APFC) unit."
        )

    if pf < PF_POOR:
        return Diagnosis(
            status="Warning",
            fault_type="Low Power Factor",
            fault_source="Load Side",
            recommendation="PF < 0.90. Consider PFC capacitors to avoid utility penalties and reduce losses."
        )

    # 4. Overcurrent
    if irms > 1.2 * NOMINAL_CURRENT_RMS:
         return Diagnosis(
            status="Warning",
            fault_type="Overcurrent / Overload",
            fault_source="Load Side",
            recommendation="Reduce load. Check for mechanical jams (if motor) or short circuits."
        )

    # 5. Healthy State Analysis (Industrial Realism)
    # Even if healthy, give insight into Load Type behavior.
    
    if abs(phase_shift) > 5.0 and pf >= PF_POOR:
        # It's an inductive/capacitive load that is EFFICIENT, but definitely not "Resistive/Linear"
        lag_lead = "Lagging" if phase_shift > 0 else "Leading"
        return Diagnosis(
            status="Healthy",
            fault_type=f"Normal Inductive Load ({lag_lead})",
            fault_source="Load Side",
            recommendation="System is operating efficiently. \n**Preventive Tip:** Check motor cooling fans and bearing lubrication periodically."
        )
        
    # Pure Linear / Resistive
    return Diagnosis(
        status="Healthy",
        fault_type="Optimal Resistive/Linear Load",
        fault_source="None",
        recommendation="System is running perfectly. \n**Preventive Tip:** Regularly torque terminal connections to prevent hotspots."
    )
