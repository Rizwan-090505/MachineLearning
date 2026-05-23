import numpy as np
import pandas as pd

SEED = 4

def make_low_noise_dataset():
    rng = np.random.RandomState(SEED)
    n = 1500
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


def make_high_noise_dataset():
    rng = np.random.RandomState(SEED + 1)
    n = 2000
    informative = rng.randn(n, 5)
    noisy_features = rng.randn(n, 15) * 5
    X = np.hstack([informative, noisy_features])
    signal = informative[:, 0] + informative[:, 1] - informative[:, 2]
    y_clean = (signal > signal.mean()).astype(int)
    flip = rng.rand(n) < 0.25
    y = np.where(flip, 1 - y_clean, y_clean)
    cols = [f"informative_{i}" for i in range(5)] + [f"noise_{i}" for i in range(15)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def make_high_dimensional_dataset():
    rng = np.random.RandomState(SEED + 2)
    n = 5000
    informative = rng.randn(n, 10)
    noise = rng.randn(n, 50)
    X = np.hstack([informative, noise])
    signal = (
        informative[:, 0] * informative[:, 1]
        + np.sin(informative[:, 2])
        - informative[:, 3] ** 2
        + informative[:, 4]
    )
    y = (signal > np.median(signal)).astype(int)
    cols = [f"informative_{i}" for i in range(10)] + [f"noise_{i}" for i in range(50)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def make_kmeans_friendly_dataset():
    rng = np.random.RandomState(SEED)
    centers = np.array([[0, 0], [10, 0], [5, 8], [-5, 6], [10, 10]])
    k = len(centers)
    n_per = 300
    chunks = []
    labels = []
    for i, c in enumerate(centers):
        base = rng.randn(n_per, 2) * 0.8 + c
        extra = rng.randn(n_per, 13)
        chunks.append(np.hstack([base, extra]))
        labels.extend([i] * n_per)
    X = np.vstack(chunks)
    cols = ["x0", "x1"] + [f"noise_{i}" for i in range(13)]
    return pd.DataFrame(X, columns=cols), pd.Series(labels, name="cluster")


def make_kmeans_adversarial_dataset():
    rng = np.random.RandomState(SEED + 3)
    n = 500
    theta = rng.rand(n) * 2 * np.pi
    r_inner = 2.0
    r_outer = 5.0
    inner = np.column_stack([r_inner * np.cos(theta[:n//2]), r_inner * np.sin(theta[:n//2])])
    outer = np.column_stack([r_outer * np.cos(theta[n//2:]), r_outer * np.sin(theta[n//2:])])
    X2d = np.vstack([inner, outer])
    extra = rng.randn(n, 13)
    X = np.hstack([X2d, extra])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    cols = ["x0", "x1"] + [f"noise_{i}" for i in range(13)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="cluster")


def make_correlated_features_dataset():
    rng = np.random.RandomState(SEED + 4)
    n = 2000
    x1 = rng.randn(n)
    x2 = x1 + rng.randn(n) * 0.05
    x3 = rng.randn(n)
    x4 = rng.randn(n)
    x5 = rng.randn(n)
    noise_feats = rng.randn(n, 10)
    X = np.column_stack([x1, x2, x3, x4, x5, noise_feats])
    signal = 2 * x1 - x3 + 0.5 * x5
    y = (signal > 0).astype(int)
    cols = ["x1_informative", "x2_corr_x1", "x3_informative", "x4_redundant", "x5_informative"] + [f"noise_{i}" for i in range(10)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def make_nb_works_despite_violation():
    rng = np.random.RandomState(SEED + 5)
    n = 1500
    y = rng.randint(0, 2, n)
    x1 = y * 3 + rng.randn(n) * 0.3
    x2 = x1 + rng.randn(n) * 0.1
    x3 = (1 - y) * 3 + rng.randn(n) * 0.3
    noise = rng.randn(n, 12)
    X = np.column_stack([x1, x2, x3, noise])
    cols = ["x1", "x2_corr", "x3"] + [f"noise_{i}" for i in range(12)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


def make_nb_fails_dataset():
    rng = np.random.RandomState(SEED + 6)
    n = 2000
    X_ring = []
    y_ring = []
    for label, r in enumerate([1.5, 4.0]):
        theta = rng.rand(n // 2) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)]) + rng.randn(n // 2, 2) * 0.2
        X_ring.append(pts)
        y_ring.extend([label] * (n // 2))
    X2d = np.vstack(X_ring)
    noise = rng.randn(n, 13)
    X = np.hstack([X2d, noise])
    cols = ["x0", "x1"] + [f"noise_{i}" for i in range(13)]
    return pd.DataFrame(X, columns=cols), pd.Series(y_ring, name="label")


def make_greedy_counterexample_dataset():
    rng = np.random.RandomState(SEED + 7)
    n = 1200
    f1 = rng.choice([0, 1, 2, 3], size=n)
    f2 = np.where(f1 % 2 == 0, rng.randint(0, 2, n), rng.randint(2, 4, n))
    noise = rng.randn(n, 13)
    y = ((f1 + f2) % 2).astype(int)
    X = np.column_stack([f1, f2, noise])
    cols = ["f1_many_values", "f2_structured"] + [f"noise_{i}" for i in range(13)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="label")


if __name__ == "__main__":
    datasets = {
        "low_noise": make_low_noise_dataset(),
        "high_noise": make_high_noise_dataset(),
        "high_dimensional": make_high_dimensional_dataset(),
        "kmeans_friendly": make_kmeans_friendly_dataset(),
        "kmeans_adversarial": make_kmeans_adversarial_dataset(),
        "correlated_features": make_correlated_features_dataset(),
        "nb_works": make_nb_works_despite_violation(),
        "nb_fails": make_nb_fails_dataset(),
        "greedy_counterexample": make_greedy_counterexample_dataset(),
    }
    for name, (X, y) in datasets.items():
        print(f"{name}: X={X.shape}, y={y.shape}, classes={y.nunique()}")
    print("All datasets generated successfully.")
