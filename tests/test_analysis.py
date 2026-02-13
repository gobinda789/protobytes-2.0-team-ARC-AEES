"""Test signal processing functions."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from signal_processing.fft_analysis import estimate_frequency, harmonic_magnitudes
from signal_processing.pq_parameters import calculate_sag_swell_fraction

def test_frequency_estimation():
    fs = 5000.0
    t = np.linspace(0, 1.0, int(fs), endpoint=False)
    
    # Test 50Hz
    v = 230 * np.sin(2 * np.pi * 50 * t)
    f_est = estimate_frequency(v, fs)
    print(f"50Hz Test: Estimated {f_est:.2f} Hz")
    assert abs(f_est - 50.0) < 0.5, f"Expected 50Hz, got {f_est}"

    # Test 49.5Hz
    v = 230 * np.sin(2 * np.pi * 49.5 * t)
    f_est = estimate_frequency(v, fs)
    print(f"49.5Hz Test: Estimated {f_est:.2f} Hz")
    assert abs(f_est - 49.5) < 0.5, f"Expected 49.5Hz, got {f_est}"

def test_sag_swell():
    # Nominal
    s = calculate_sag_swell_fraction(230, 230)
    print(f"Nominal 230V: {s}")
    assert s == 0.0

    # Sag 20% -> 0.8 * 230 = 184V
    s = calculate_sag_swell_fraction(184, 230)
    print(f"Sag 184V (0.8pu): {s}")
    assert abs(s - 0.2) < 0.01

    # Swell 10% -> 1.1 * 230 = 253V
    s = calculate_sag_swell_fraction(253, 230)
    print(f"Swell 253V (1.1pu): {s}")
    assert abs(s - 0.1) < 0.01

def test_harmonics():
    fs = 5000.0
    t = np.linspace(0, 0.2, int(0.2*fs), endpoint=False)
    # Fundamental 50Hz, 3rd 10%
    v = 100 * np.sin(2 * np.pi * 50 * t) + 10 * np.sin(2 * np.pi * 150 * t)
    
    mags = harmonic_magnitudes(v, 50.0)
    h1 = mags['H1']
    h3 = mags['H3']
    ratio = h3 / h1
    print(f"Harmonics: H1={h1:.2f}, H3={h3:.2f}, Ratio={ratio:.3f}")
    assert abs(ratio - 0.1) < 0.01

if __name__ == "__main__":
    try:
        test_frequency_estimation()
        test_sag_swell()
        test_harmonics()
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
