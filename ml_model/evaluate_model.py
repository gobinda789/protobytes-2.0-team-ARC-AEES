"""Evaluate trained model and print detailed metrics."""

from __future__ import annotations

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data.feature_extraction import FEATURE_COLUMNS


def evaluate(
    dataset_csv: str,
    model_path: str,
    random_state: int = 7,
) -> float:
    """Evaluate model on held-out test set and print metrics.

    Returns:
        Test accuracy.
    """
    df = pd.read_csv(dataset_csv)
    X = df[FEATURE_COLUMNS].values
    y = df["LoadType"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[EVAL] Test accuracy: {acc*100:.2f}%")
    print("[EVAL] Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("[EVAL] Classification report:")
    print(classification_report(y_test, y_pred))

    return float(acc)
