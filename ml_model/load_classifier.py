"""Load classifier inference utility."""

from __future__ import annotations

import numpy as np
import joblib

from data.feature_extraction import FEATURE_COLUMNS


class LoadClassifier:
    """Wrapper around the trained scikit-learn pipeline."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, feature_vector: dict[str, float] | np.ndarray) -> str:
        """Predict load class from a feature vector.

        Args:
            feature_vector: Either a dict with FEATURE_COLUMNS keys,
                            or a numpy array shape (n_features,).

        Returns:
            Predicted class label.
        """
        if isinstance(feature_vector, dict):
            x = np.array([[feature_vector[c] for c in FEATURE_COLUMNS]], dtype=float)
        else:
            arr = np.asarray(feature_vector, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            x = arr
        return str(self.model.predict(x)[0])
