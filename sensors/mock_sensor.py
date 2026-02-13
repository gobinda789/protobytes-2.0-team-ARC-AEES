"""Mock sensor implementation for testing."""

import numpy as np
import pandas as pd
from .base import SensorInterface
from signal_processing.waveform_generator import generate_voltage_wave, generate_current_wave

class MockSensor(SensorInterface):
    """Generates synthetic data with noise to simulate a real sensor."""

    def __init__(self, sampling_rate: float = 5000.0, noise_level: float = 0.05):
        self._fs = sampling_rate
        self.noise_level = noise_level
        self.time_counter = 0.0

    @property
    def sampling_rate(self) -> float:
        return self._fs

    def read_batch(self, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a batch of noisy data."""
        duration = n_samples / self._fs
        t = np.linspace(self.time_counter, self.time_counter + duration, n_samples, endpoint=False)
        self.time_counter += duration

        # Generate base nominal waves
        # Pass the time array we just created
        _, v = generate_voltage_wave(230, 50, time_array=t)
        _, i = generate_current_wave("Linear", 10, 50, time_array=t)

        # Add noise

        v += np.random.normal(0, self.noise_level * 230, size=v.shape)
        i += np.random.normal(0, self.noise_level * 10, size=i.shape)

        return t, v, i

class FileSensor(SensorInterface):
    """Reads data from a CSV file."""

    def __init__(self, file_path: str, sampling_rate: float = 5000.0):
        self._fs = sampling_rate
        try:
            df = pd.read_csv(file_path)
            # Expect generic names or standardized
            # We assume columns: [Time, Voltage, Current] or similar
            self.t = df.iloc[:, 0].values
            self.v = df.iloc[:, 1].values
            self.i = df.iloc[:, 2].values
            self.n_total = len(self.t)
            self.pointer = 0
            
            # Estimate sampling rate if possible from time
            if len(self.t) > 1:
                dt = np.mean(np.diff(self.t))
                if dt > 0:
                    self._fs = 1.0 / dt
                    
        except Exception as e:
            print(f"Error loading file: {e}")
            self.t, self.v, self.i = np.array([]), np.array([]), np.array([])
            self.n_total = 0
            self.pointer = 0

    @property
    def sampling_rate(self) -> float:
        return self._fs

    def read_batch(self, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read simpler implementation: just return the whole file or loop it."""
        # For this demo, let's just return the whole dataset if it fits, 
        # or implement looping if we want a "stream" feel.
        # But user likely wants to analyze "a capture".
        
        # If n_samples is -1, return everything
        if n_samples == -1 or n_samples >= self.n_total:
            return self.t, self.v, self.i
            
        # Basic looping implementation
        start = self.pointer
        end = start + n_samples
        
        if end > self.n_total:
            # Wrap around
            t_chunk = np.concatenate([self.t[start:], self.t[:end-self.n_total]])
            v_chunk = np.concatenate([self.v[start:], self.v[:end-self.n_total]])
            i_chunk = np.concatenate([self.i[start:], self.i[:end-self.n_total]])
            self.pointer = end - self.n_total
        else:
            t_chunk = self.t[start:end]
            v_chunk = self.v[start:end]
            i_chunk = self.i[start:end]
            self.pointer = end
            
        return t_chunk, v_chunk, i_chunk
