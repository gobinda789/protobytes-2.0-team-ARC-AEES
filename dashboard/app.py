"""Streamlit dashboard for Smart PQ AI."""

from __future__ import annotations

import os
import sys

# Add parent directory to path to allow importing config and signal_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import SYSTEM_FREQUENCY, NOMINAL_VOLTAGE_RMS, NOMINAL_CURRENT_RMS
from signal_processing.waveform_generator import generate_voltage_wave, generate_current_wave
from signal_processing.disturbances import (
    apply_voltage_sag,
    apply_voltage_swell,
    apply_harmonic_injection,
    apply_frequency_deviation,
)
from signal_processing.fft_analysis import compute_spectrum, thd_percent, harmonic_magnitudes, estimate_frequency
from signal_processing.pq_parameters import rms, active_power, apparent_power, power_factor, crest_factor, calculate_sag_swell_fraction
from data.feature_extraction import extract_features, FEATURE_COLUMNS
from ml_model.load_classifier import LoadClassifier
from sensors.mock_sensor import MockSensor, FileSensor

MODEL_PATH = os.path.join("ml_model", "model.pkl")


def _plot_waveforms(t: np.ndarray, v: np.ndarray, i: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, v, label="Voltage (V)")
    ax.plot(t, i, label="Current (A)")
    ax.set_title("Voltage & Current Waveforms")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_spectrum(freqs: np.ndarray, amps: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(freqs, amps)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 800)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Smart PQ AI Analyzer", layout="wide")

    st.title("AI-Based Smart Power Quality Analyzer")
    st.caption("Analyzes power quality from Simulation or Real Sensors/Files.")

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run: python main.py (it will generate dataset and train model).")
        st.stop()

    clf = LoadClassifier(MODEL_PATH)

    # Sidebar controls
    st.sidebar.header("Configuration")
    data_source = st.sidebar.radio("Data Source", ["Simulation", "Real Sensor / File"])

    v = np.array([])
    i = np.array([])
    t = np.array([])
    f_used = SYSTEM_FREQUENCY # Default
    fs = 5000.0 # Default

    if data_source == "Simulation":
        st.sidebar.header("Simulation Controls")

        load_type = st.sidebar.selectbox(
            "Choose TRUE load type (for simulation)",
            ["Linear", "InductionMotor", "SMPS", "LEDDriver", "Nonlinear"],
            index=0,
        )

        disturbance = st.sidebar.selectbox(
            "Disturbance",
            ["None", "Sag", "Swell", "Harmonics", "FreqDev"],
            index=0,
        )

        vrms = st.sidebar.slider("Voltage RMS (V)", 180, 260, int(NOMINAL_VOLTAGE_RMS), 1)
        irms = st.sidebar.slider("Current RMS (A)", 3, 20, int(NOMINAL_CURRENT_RMS), 1)

        sag = st.sidebar.slider("Sag fraction", 0.20, 0.40, 0.30, 0.01)
        swell = st.sidebar.slider("Swell fraction", 0.20, 0.40, 0.30, 0.01)

        h3 = st.sidebar.slider("Voltage 3rd harmonic fraction", 0.00, 0.10, 0.05, 0.005)
        h5 = st.sidebar.slider("Voltage 5th harmonic fraction", 0.00, 0.10, 0.03, 0.005)
        h7 = st.sidebar.slider("Voltage 7th harmonic fraction", 0.00, 0.10, 0.02, 0.005)

        f_dev = st.sidebar.slider("Frequency (Hz)", 49.0, 51.0, float(SYSTEM_FREQUENCY), 0.1)

        st.sidebar.markdown("---")
        st.sidebar.write("IEEE 519 indicator based on **current THD**:")
        st.sidebar.write("🟢 THD < 5% (Compliant), 🔴 THD ≥ 5% (Non-compliant)")

        # Generate waveforms
        t, v = generate_voltage_wave(vrms=float(vrms), frequency_hz=SYSTEM_FREQUENCY)

        if disturbance == "Sag":
            v = apply_voltage_sag(v, sag)
            f_used = SYSTEM_FREQUENCY
        elif disturbance == "Swell":
            v = apply_voltage_swell(v, swell)
            f_used = SYSTEM_FREQUENCY
        elif disturbance == "Harmonics":
            v = apply_harmonic_injection(t, v, h3=h3, h5=h5, h7=h7)
            f_used = SYSTEM_FREQUENCY
        elif disturbance == "FreqDev":
            v = apply_frequency_deviation(t, vrms=float(vrms), frequency_hz=float(f_dev))
            f_used = float(f_dev)
        else:
            f_used = SYSTEM_FREQUENCY

        _, i = generate_current_wave(load_type=load_type, irms=float(irms), frequency_hz=float(f_used))
        
        # In simulation, we know fs is fixed at config 5000 from wf generator
        fs = 5000.0

    else:
        # Real Sensor / File
        st.sidebar.header("Sensor Configuration")
        sensor_type = st.sidebar.selectbox("Input Type", ["Mock Sensor (Demo)", "Upload CSV"])
        
        if sensor_type == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader("Upload CSV (Time, Voltage, Current)", type=["csv"])
            if uploaded_file is not None:
                # Save to temp file or read directly
                # We can use our FileSensor but it takes a path. 
                # Let's just use pandas directly here for simplicity, or save temp.
                try:
                    df = pd.read_csv(uploaded_file)
                    st.sidebar.success(f"Loaded {len(df)} samples")
                    # Assume columns 0,1,2 are T, V, I
                    t = df.iloc[:, 0].values
                    v = df.iloc[:, 1].values
                    i = df.iloc[:, 2].values
                    
                    # Estimate fs
                    if len(t) > 1:
                        fs = 1.0 / np.mean(np.diff(t))
                    else:
                        fs = 5000.0
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.stop()
            else:
                st.info("Please upload a CSV file to proceed.")
                st.stop()
        else:
            # Mock Sensor
            noise = st.sidebar.slider("Noise Level", 0.0, 0.1, 0.02)
            sensor = MockSensor(sampling_rate=5000.0, noise_level=noise)
            t, v, i = sensor.read_batch(n_samples=1000)
            fs = sensor.sampling_rate

        # Estimate frequency from data
        f_est = estimate_frequency(v, fs)
        f_used = f_est
        st.sidebar.metric("Estimated Frequency", f"{f_est:.2f} Hz")


    # --- Analysis & Display (Common) ---

    # Compute metrics
    vr = rms(v)
    ir = rms(i)
    p = active_power(v, i)
    s_pwr = apparent_power(vr, ir) # renamed to avoid conflict with 's' var
    pf = power_factor(p, s_pwr)
    cf = crest_factor(i)

    # Harmonics & THD
    thd_i = thd_percent(i, fundamental_hz=float(f_used))
    hm_i = harmonic_magnitudes(i, fundamental_hz=float(f_used))
    
    # Voltage Harmonics (Requested by user)
    hm_v = harmonic_magnitudes(v, fundamental_hz=float(f_used))
    # Normalize to fraction of fundamental if fundamental > 0
    v_fund = hm_v['H1']
    if v_fund > 1.0:
        h3_v = hm_v['H3'] / v_fund
        h5_v = hm_v['H5'] / v_fund
        h7_v = hm_v['H7'] / v_fund
    else:
        h3_v, h5_v, h7_v = 0.0, 0.0, 0.0

    # Sag/Swell
    sag_swell_frac = calculate_sag_swell_fraction(vr, NOMINAL_VOLTAGE_RMS)
    is_sag = vr < NOMINAL_VOLTAGE_RMS and sag_swell_frac > 0.05
    is_swell = vr > NOMINAL_VOLTAGE_RMS and sag_swell_frac > 0.05

    # Feature Extraction & Prediction
    feats = extract_features(v, i, fundamental_hz=float(f_used))
    pred = clf.predict(feats)

    compliant = thd_i < 5.0

    # Diagnosis Logic
    # Estimate phase shift from PF. 
    # PF = cos(phi), so |phi| = arccos(PF).
    # We don't have sign from just PF, but we can assume lagging for now or try to extract from data if needed.
    # For now, let's just use the scalar values.
    # In simulation, we know if we injected a phase shift (we didn't explicitly, but load type implicitly does).
    # Let's just pass a dummy phase shift or estimate it.
    try:
        phi_rad = np.arccos(np.clip(pf, -1.0, 1.0))
        phi_deg = np.degrees(phi_rad)
        # Rudimentary check for lead/lag if we had instantaneous data, but here we just take magnitude
        # or assume lagging for inductive loads (most common).
        # The diagnosis logic uses phase_shift > 5.
    except:
        phi_deg = 0.0

    from signal_processing.power_analysis import diagnose_power_quality
    diagnosis = diagnose_power_quality(
        vrms=vr,
        irms=ir,
        thd_i=thd_i,
        pf=pf,
        phase_shift=phi_deg,
        nominal_v=NOMINAL_VOLTAGE_RMS
    )

    # Layout
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("Waveforms")
        fig_w = _plot_waveforms(t, v, i)
        st.pyplot(fig_w, use_container_width=True)

        st.subheader("FFT Spectrum (Current)")
        freqs, amps = compute_spectrum(i)
        fig_s = _plot_spectrum(freqs, amps, "Current Spectrum (One-Sided Amplitude)")
        st.pyplot(fig_s, use_container_width=True)

    with right:
        st.subheader("Power Quality Metrics")

        c1, c2 = st.columns(2)
        c1.metric("Vrms (V)", f"{vr:.2f}")
        c2.metric("Irms (A)", f"{ir:.2f}")

        c3, c4 = st.columns(2)
        c3.metric("Freq (Hz)", f"{f_used:.2f}")
        c4.metric("PF", f"{pf:.3f}")

        st.markdown("---")
        st.write("**Sag / Swell Analysis**")
        c5, c6 = st.columns(2)
        
        if is_sag:
             c5.metric("Sag Fraction", f"{sag_swell_frac:.2f}", delta="-SAG", delta_color="inverse")
        elif is_swell:
             c5.metric("Swell Fraction", f"{sag_swell_frac:.2f}", delta="+SWELL", delta_color="inverse")
        else:
             c5.metric("Sag/Swell", "Normal")
             
        c6.metric("Crest Factor (I)", f"{cf:.3f}")

        st.markdown("---")
        st.subheader("Harmonics")
        
        tab1, tab2 = st.tabs(["Current (Abs)", "Voltage (Fraction)"])
        
        with tab1:
            st.write(f"**Current THD:** {thd_i:.2f}%")
            st.write(f"H3: **{hm_i['H3']:.3f} A**")
            st.write(f"H5: **{hm_i['H5']:.3f} A**")
            st.write(f"H7: **{hm_i['H7']:.3f} A**")
            
        with tab2:
            st.write(f"**Voltage Harmonics (Fraction of Fundamental)**")
            st.write(f"H3 Fraction: **{h3_v:.3f}**")
            st.write(f"H5 Fraction: **{h5_v:.3f}**")
            st.write(f"H7 Fraction: **{h7_v:.3f}**")

        st.markdown("---")
        st.subheader("AI Load Classification")
        st.write(f"**Predicted Load Type:** `{pred}`")
        if data_source == "Simulation":
            st.caption(f"True Load: {load_type} | Disturbance: {disturbance}")

        st.markdown("---")
        st.subheader("IEEE 519 Compliance")
        if compliant:
            st.success("🟢 Compliant (THD < 5%)")
        else:
            st.error("🔴 Non-compliant (THD ≥ 5%)")

        with st.expander("View Feature Vector"):
            st.dataframe({k: [float(feats[k])] for k in FEATURE_COLUMNS}, use_container_width=True)

    # --- New Section: Smart Doctor Diagnosis ---
    st.markdown("---")
    st.header("🩺 Smart Doctor Diagnosis")
    
    # Color-code status
    status_color = "green"
    if diagnosis.status == "Warning":
        status_color = "orange"
    elif diagnosis.status == "Critical":
        status_color = "red"

    st.markdown(f"**Status:** :{status_color}[{diagnosis.status}]")
    st.markdown(f"**Fault Type:** {diagnosis.fault_type}")
    st.markdown(f"**Source:** {diagnosis.fault_source}")
    
    st.info(f"**💡 Engineering Recommendation:**\n\n{diagnosis.recommendation}")


if __name__ == "__main__":
    main()
