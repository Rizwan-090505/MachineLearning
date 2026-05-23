import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from dataset_generator import (
    make_kmeans_friendly_dataset,
    make_kmeans_adversarial_dataset,
)

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


class KMeans:
    def __init__(self, k, max_iter=300, tol=1e-4, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

    def _init_centroids(self, X):
        idx = self.rng.choice(len(X), self.k, replace=False)
        return X[idx].astype(float)

    def _assign(self, X, centroids):
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        return np.argmin(dists, axis=1)

    def _update_centroids(self, X, labels):
        return np.array([
            X[labels == j].mean(axis=0) if (labels == j).any() else self.centroids_[j]
            for j in range(self.k)
        ])

    def _inertia(self, X, labels, centroids):
        diffs = X - centroids[labels]
        return (diffs ** 2).sum()

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.centroids_ = self._init_centroids(X)
        self.inertia_history_ = []
        for _ in range(self.max_iter):
            labels = self._assign(X, self.centroids_)
            new_centroids = self._update_centroids(X, labels)
            cost = self._inertia(X, labels, new_centroids)
            self.inertia_history_.append(cost)
            shift = np.sqrt(((new_centroids - self.centroids_) ** 2).sum())
            self.centroids_ = new_centroids
            if shift < self.tol:
                break
        self.labels_ = self._assign(X, self.centroids_)
        self.inertia_ = self._inertia(X, self.labels_, self.centroids_)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return self._assign(X, self.centroids_)


def run_kmeans_friendly():
    X_df, y = make_kmeans_friendly_dataset()
    X = X_df[["x0", "x1"]].values

    km = KMeans(k=5, random_state=42)
    km.fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=10, alpha=0.6)
    axes[0].set_title("Ground Truth Clusters")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("x1")

    axes[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="tab10", s=10, alpha=0.6)
    axes[1].scatter(km.centroids_[:, 0], km.centroids_[:, 1], c="black", marker="X", s=200, zorder=5, label="Centroids")
    axes[1].set_title(f"k-Means Result (inertia={km.inertia_:.1f})")
    axes[1].set_xlabel("x0")
    axes[1].set_ylabel("x1")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/kmeans_friendly.png", dpi=120)
    plt.close()
    print(f"k-Means friendly dataset inertia: {km.inertia_:.2f}")
    return km.inertia_


def run_kmeans_adversarial():
    X_df, y = make_kmeans_adversarial_dataset()
    X = X_df[["x0", "x1"]].values

    km = KMeans(k=2, random_state=42)
    km.fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=10, alpha=0.7)
    axes[0].set_title("Ground Truth: Ring Structure")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("x1")

    axes[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="bwr", s=10, alpha=0.7)
    axes[1].scatter(km.centroids_[:, 0], km.centroids_[:, 1], c="black", marker="X", s=200, zorder=5)
    axes[1].set_title("k-Means Result (Fails on Rings)")
    axes[1].set_xlabel("x0")
    axes[1].set_ylabel("x1")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/kmeans_adversarial.png", dpi=120)
    plt.close()

    X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    km_scaled = KMeans(k=2, random_state=42)
    km_scaled.fit(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=km_scaled.labels_, cmap="bwr", s=10, alpha=0.7)
    ax.set_title("k-Means on Scaled Adversarial Data (Still Fails)")
    ax.set_xlabel("x0 (scaled)")
    ax.set_ylabel("x1 (scaled)")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/kmeans_adversarial_scaled.png", dpi=120)
    plt.close()
    print("Adversarial dataset experiment done.")


def run_initialization_sensitivity():
    X_df, _ = make_kmeans_friendly_dataset()
    X = X_df[["x0", "x1"]].values

    inertias = []
    histories = []
    n_runs = 20

    for i in range(n_runs):
        km = KMeans(k=5, random_state=i * 7)
        km.fit(X)
        inertias.append(km.inertia_)
        histories.append(km.inertia_history_)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, h in enumerate(histories):
        axes[0].plot(h, alpha=0.5, linewidth=0.9)
    axes[0].set_title("Convergence per Initialization (20 Runs)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Inertia")

    axes[1].bar(range(n_runs), inertias, color="steelblue", edgecolor="black", linewidth=0.5)
    axes[1].axhline(np.min(inertias), color="red", linestyle="--", label=f"Min: {np.min(inertias):.1f}")
    axes[1].axhline(np.max(inertias), color="orange", linestyle="--", label=f"Max: {np.max(inertias):.1f}")
    axes[1].set_title("Final Inertia per Initialization")
    axes[1].set_xlabel("Run Index")
    axes[1].set_ylabel("Final Inertia")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/kmeans_init_sensitivity.png", dpi=120)
    plt.close()

    print(f"Initialization sensitivity: min={np.min(inertias):.2f}, max={np.max(inertias):.2f}, std={np.std(inertias):.2f}")
    return inertias


if __name__ == "__main__":
    print("Running k-Means experiments...")
    run_kmeans_friendly()
    run_kmeans_adversarial()
    run_initialization_sensitivity()
    print("k-Means experiments complete. Figures saved to figures/")
