"""Train RandomForest model for load classification."""

from __future__ import annotations

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data.feature_extraction import FEATURE_COLUMNS


def train_and_save(
    dataset_csv: str,
    model_path: str,
    random_state: int = 7,
) -> float:
    """Train the model and save it to disk.

    Returns:
        Training accuracy on the training split.
    """
    df = pd.read_csv(dataset_csv)

    X = df[FEATURE_COLUMNS].values
    y = df["LoadType"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Pipeline with scaling + RF (scaling not mandatory for RF, but helps keep features comparable for future models)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
                class_weight=None
            )),
        ]
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"[TRAIN] Training accuracy: {train_acc*100:.2f}%")
    return float(train_acc)
