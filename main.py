"""Main execution flow.

- Generate dataset if missing
- Train model if missing
- Evaluate model and enforce minimum accuracy
- Launch Streamlit dashboard

Run:
    python main.py
or:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import subprocess

from data.dataset_generator import generate_dataset
from ml_model.train_model import train_and_save
from ml_model.evaluate_model import evaluate


DATASET_CSV = os.path.join("data", "dataset.csv")
MODEL_PATH = os.path.join("ml_model", "model.pkl")


def ensure_dataset() -> None:
    """Generate dataset CSV if it doesn't exist."""
    if os.path.exists(DATASET_CSV):
        print(f"[MAIN] Dataset exists: {DATASET_CSV}")
        return
    print("[MAIN] Generating dataset...")
    generate_dataset(DATASET_CSV, n_samples=900, seed=42)
    print(f"[MAIN] Dataset saved: {DATASET_CSV}")


def ensure_model() -> None:
    """Train model if it doesn't exist."""
    if os.path.exists(MODEL_PATH):
        print(f"[MAIN] Model exists: {MODEL_PATH}")
        return
    print("[MAIN] Training model...")
    train_and_save(DATASET_CSV, MODEL_PATH, random_state=7)
    print(f"[MAIN] Model saved: {MODEL_PATH}")


def evaluate_model(min_accuracy: float = 0.85) -> None:
    """Evaluate model and enforce accuracy threshold."""
    acc = evaluate(DATASET_CSV, MODEL_PATH, random_state=7)
    if acc < min_accuracy:
        raise RuntimeError(
            f"Model accuracy {acc*100:.2f}% is below required {min_accuracy*100:.2f}%. "
            "Adjust simulation parameters/features or model settings."
        )
    print(f"[MAIN] Accuracy requirement met: {acc*100:.2f}%")


def launch_dashboard() -> None:
    """Launch Streamlit dashboard."""
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.join("dashboard", "app.py")]
    print("[MAIN] Launching dashboard...")
    subprocess.run(cmd, check=False)


def main() -> None:
    ensure_dataset()
    ensure_model()
    evaluate_model(min_accuracy=0.85)
    launch_dashboard()


if __name__ == "__main__":
    main()
