"""Abstract base class for sensor data ingestion."""

from abc import ABC, abstractmethod
import numpy as np

class SensorInterface(ABC):
    """Interface for reading voltage and current data from a sensor."""

    @abstractmethod
    def read_batch(self, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read a batch of samples.

        Args:
            n_samples: Number of samples to read.

        Returns:
            tuple: (time_array, voltage_array, current_array)
        """
        pass

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """Return the sampling rate in Hz."""
        pass
