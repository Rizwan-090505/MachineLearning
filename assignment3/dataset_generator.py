"""
dataset_generator.py
Generate datasets for ensemble methods assignment with numpy vectorization
Roll Number: BSAI23004
"""

import numpy as np
import pandas as pd
import os

SEED = 4
DATASETS_DIR = "datasets"


def make_low_noise_dataset(n=1500, random_state=SEED):
    """Low-noise dataset for clean learning scenarios."""
    rng = np.random.RandomState(random_state)
    informative = np.hstack([
        rng.randn(n, 5) * np.array([3, 2, 4, 1.5, 2.5]),
        rng.randn(n, 10)
    ])
    labels_raw = (
        2 * informative[:, 0]
        - 1.5 * informative[:, 1]
        + informative[:, 2]
        + 0.5 * informative[:, 3]
        - informative[:, 4]
    )
    y = (labels_raw > labels_raw.mean()).astype(int)
    cols = [f"informative_{i}" for i in range(5)] + [f"noise_{i}" for i in range(10)]
    return pd.DataFrame(informative, columns=cols), pd.Series(y, name="label")


def make_high_noise_dataset(n=600, random_state=SEED+1):
    """High-noise dataset with 25% label flip."""
    rng = np.random.RandomState(random_state)
    informative = rng.randn(n, 5)
    noisy_features = rng.randn(n, 15) * 5
    X = np.hstack([informative, noisy_features])
    signal = informative[:, 0] + informative[:, 1] - informative[:, 2]
    y_clean = (signal > signal.mean()).astype(int)
    flip = rng.rand(n) < 0.25
    y = np.where(flip, 1 - y_clean, y_clean)
    cols = [f"informative_{i}" for i in range(5)] + [f"noise_{i}" for i in range(15)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def make_bagging_helps_dataset(n=600, random_state=SEED+2):
    """Dataset where bagging significantly helps (high variance, low bias)."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, 8)
    y_clean = ((X[:, 0] > 0) & (X[:, 1] > 0)) | ((X[:, 2] < 0) & (X[:, 3] < 0))
    y_clean = y_clean.astype(int)
    flip = rng.rand(n) < 0.20
    y = np.where(flip, 1 - y_clean, y_clean)
    cols = [f"feature_{i}" for i in range(8)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def make_bagging_fails_dataset(n=600, random_state=SEED+3):
    """Dataset where bagging doesn't help (low variance, already stable)."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n, 8)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    cols = [f"feature_{i}" for i in range(8)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def make_feature_dominant_dataset(n=600, random_state=SEED+4):
    """Dataset with one dominant feature and many noise features."""
    rng = np.random.RandomState(random_state)
    x0_informative = rng.randn(n) * 3
    y = (x0_informative > 0).astype(int)
    noise = rng.randn(n, 14)
    X = np.hstack([x0_informative.reshape(-1, 1), noise])
    cols = ["dominant_feature"] + [f"noise_{i}" for i in range(14)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def make_high_dimensional_dataset(n=500, n_informative=5, n_noise=50, random_state=SEED+5):
    """High-dimensional dataset with few informative features."""
    rng = np.random.RandomState(random_state)
    informative = rng.randn(n, n_informative)
    noise = rng.randn(n, n_noise)
    X = np.hstack([informative, noise])
    signal = (
        informative[:, 0] * informative[:, 1]
        + np.sin(informative[:, 2])
        - informative[:, 3] ** 2
        + informative[:, 4] if n_informative >= 5 else informative[:, 0]
    )
    y = (signal > np.median(signal)).astype(int)
    cols = [f"informative_{i}" for i in range(n_informative)] + [f"noise_{i}" for i in range(n_noise)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def make_boosting_noise_dataset(n=1500, flip_prob=0.15, random_state=SEED+6):
    """Dataset with label noise for boosting analysis."""
    rng = np.random.RandomState(random_state)
    informative = rng.randn(n, 5)
    X = np.hstack([informative, rng.randn(n, 10)])
    signal = informative[:, 0] - informative[:, 1] + informative[:, 2]
    y_clean = (signal > 0).astype(int)
    flip = rng.rand(n) < flip_prob
    y = np.where(flip, 1 - y_clean, y_clean)
    cols = [f"informative_{i}" for i in range(5)] + [f"noise_{i}" for i in range(10)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def save_all(random_state=SEED):
    """Generate and save all datasets to CSV."""
    os.makedirs(DATASETS_DIR, exist_ok=True)
    datasets = {
        "low_noise": make_low_noise_dataset(random_state=random_state),
        "high_noise": make_high_noise_dataset(random_state=random_state),
        "bagging_helps": make_bagging_helps_dataset(random_state=random_state),
        "bagging_fails": make_bagging_fails_dataset(random_state=random_state),
        "feature_dominant": make_feature_dominant_dataset(random_state=random_state),
        "high_dimensional": make_high_dimensional_dataset(random_state=random_state),
        "boosting_noise": make_boosting_noise_dataset(random_state=random_state),
    }
    for name, (X, y) in datasets.items():
        X.to_csv(f"{DATASETS_DIR}/{name}_X.csv", index=False)
        y.to_csv(f"{DATASETS_DIR}/{name}_y.csv", index=False)
    print(f"  [✓] Generated {len(datasets)} datasets in {DATASETS_DIR}/")


if __name__ == "__main__":
    save_all()
