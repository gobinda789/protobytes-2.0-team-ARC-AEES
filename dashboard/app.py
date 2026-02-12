"""Streamlit dashboard for Smart PQ AI (Simulation Only)."""

from __future__ import annotations

import os
import numpy as np
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
from signal_processing.fft_analysis import compute_spectrum, thd_percent, harmonic_magnitudes
from signal_processing.pq_parameters import rms, active_power, apparent_power, power_factor, crest_factor
from data.feature_extraction import extract_features, FEATURE_COLUMNS
from ml_model.load_classifier import LoadClassifier


MODEL_PATH = os.path.join("ml_model", "model.pkl")


def _plot_waveforms(t: np.ndarray, v: np.ndarray, i: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, v, label="Voltage (V)")
    ax.plot(t, i, label="Current (A)")
    ax.set_title("Simulated Voltage & Current Waveforms")
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
    st.set_page_config(page_title="Smart PQ AI Analyzer (Simulation)", layout="wide")

    st.title("AI-Based Smart Power Quality Analyzer with Load Classification (Simulation Only)")
    st.caption("Simulates voltage/current, injects disturbances, performs FFT-based harmonic analysis, computes PQ metrics, and classifies load type.")

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run: python main.py (it will generate dataset and train model).")
        st.stop()

    clf = LoadClassifier(MODEL_PATH)

    # Sidebar controls
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

    # Compute metrics & features
    vr = rms(v)
    ir = rms(i)
    p = active_power(v, i)
    s = apparent_power(vr, ir)
    pf = power_factor(p, s)
    cf = crest_factor(i)

    thd_i = thd_percent(i, fundamental_hz=float(f_used))
    hm = harmonic_magnitudes(i, fundamental_hz=float(f_used))

    feats = extract_features(v, i, fundamental_hz=float(f_used))
    pred = clf.predict(feats)

    compliant = thd_i < 5.0

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
        st.subheader("Power Quality Summary")

        c1, c2 = st.columns(2)
        c1.metric("Vrms (V)", f"{vr:.2f}")
        c2.metric("Irms (A)", f"{ir:.2f}")

        c3, c4 = st.columns(2)
        c3.metric("Power Factor", f"{pf:.3f}")
        c4.metric("Crest Factor (I)", f"{cf:.3f}")

        st.metric("Current THD (%)", f"{thd_i:.2f}")

        st.markdown("---")
        st.subheader("Harmonics (Current)")
        st.write(f"H3 magnitude: **{hm['H3']:.3f}**")
        st.write(f"H5 magnitude: **{hm['H5']:.3f}**")
        st.write(f"H7 magnitude: **{hm['H7']:.3f}**")

        st.markdown("---")
        st.subheader("AI Load Classification")
        st.write(f"**Predicted Load Type:** `{pred}`")
        st.caption(f"True simulated load type: {load_type} | Disturbance: {disturbance}")

        st.markdown("---")
        st.subheader("IEEE 519 Indicator (THD)")
        if compliant:
            st.success("🟢 Compliant: THD < 5%")
        else:
            st.error("🔴 Non-compliant: THD ≥ 5%")

        st.markdown("---")
        st.subheader("Feature Vector Used")
        st.dataframe({k: [float(feats[k])] for k in FEATURE_COLUMNS}, use_container_width=True)


if __name__ == "__main__":
    main()
