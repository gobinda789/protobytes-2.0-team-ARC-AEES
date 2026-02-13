"""Test MockSensor."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sensors.mock_sensor import MockSensor

def test_mock_sensor():
    ms = MockSensor(sampling_rate=5000.0)
    t, v, i = ms.read_batch(n_samples=100)
    
    print(f"Read {len(t)} samples.")
    print(f"Time range: {t[0]:.4f} to {t[-1]:.4f}")
    print(f"V range: {np.min(v):.2f} to {np.max(v):.2f}")
    
    assert len(t) == 100
    assert len(v) == 100
    assert len(i) == 100
    
    # Check continuity
    t2, v2, i2 = ms.read_batch(n_samples=100)
    print(f"Read next {len(t2)} samples.")
    print(f"Time range: {t2[0]:.4f} to {t2[-1]:.4f}")
    
    # Expected: t2[0] should be t[-1] + dt
    dt = 1.0/5000.0
    assert abs(t2[0] - (t[-1] + dt)) < 1e-9

if __name__ == "__main__":
    try:
        test_mock_sensor()
        print("MOCK SENSOR TEST PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
