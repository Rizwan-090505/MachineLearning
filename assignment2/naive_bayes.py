import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from dataset_generator import (
    make_low_noise_dataset,
    make_high_noise_dataset,
    make_correlated_features_dataset,
    make_nb_works_despite_violation,
    make_nb_fails_dataset,
)

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
SEED = 4


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_total = len(y)
        self.log_priors_ = {}
        self.means_ = {}
        self.vars_ = {}
        for c in self.classes_:
            X_c = X[y == c]
            self.log_priors_[c] = np.log(len(X_c) / n_total)
            self.means_[c] = X_c.mean(axis=0)
            self.vars_[c] = X_c.var(axis=0) + self.var_smoothing
        return self

    def _log_likelihood(self, X, c):
        mu = self.means_[c]
        var = self.vars_[c]
        log_norm = -0.5 * np.log(2 * np.pi * var)
        log_exp = -0.5 * ((X - mu) ** 2) / var
        return (log_norm + log_exp).sum(axis=1)

    def predict_log_proba(self, X):
        X = np.array(X, dtype=float)
        log_probs = np.column_stack([
            self.log_priors_[c] + self._log_likelihood(X, c)
            for c in self.classes_
        ])
        return log_probs

    def predict_proba(self, X):
        log_probs = self.predict_log_proba(X)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]


def run_basic_nb():
    X_df, y = make_low_noise_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y.values, test_size=0.2, random_state=SEED)
    gnb = GaussianNaiveBayes()
    gnb.fit(X_tr, y_tr)
    preds = gnb.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"[NB] Low-noise dataset accuracy: {acc:.4f}")
    return acc


def run_correlated_features_experiment():
    X_df, y = make_correlated_features_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y.values, test_size=0.2, random_state=SEED)
    gnb = GaussianNaiveBayes()
    gnb.fit(X_tr, y_tr)
    preds = gnb.predict(X_te)
    proba = gnb.predict_proba(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"[NB] Correlated features accuracy: {acc:.4f}")

    max_conf = proba.max(axis=1)
    errors = (preds != y_te)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(max_conf[~errors], bins=30, alpha=0.7, label="Correct", color="steelblue")
    axes[0].hist(max_conf[errors], bins=30, alpha=0.7, label="Wrong", color="tomato")
    axes[0].set_title("Confidence Distribution: Correlated Features")
    axes[0].set_xlabel("Max Predicted Probability")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    bins = np.linspace(0.5, 1.0, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (max_conf >= lo) & (max_conf < hi)
        if mask.sum() > 0:
            bin_accs.append(accuracy_score(y_te[mask], preds[mask]))
        else:
            bin_accs.append(np.nan)
    axes[1].plot(bin_centers, bin_centers, "k--", label="Perfect calibration")
    axes[1].plot(bin_centers, bin_accs, "o-", color="tomato", label="Model calibration")
    axes[1].set_title("Calibration Curve: Correlated Features NB")
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].set_ylabel("Fraction Correct")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/nb_correlated_calibration.png", dpi=120)
    plt.close()
    return acc


def run_nb_counterexamples():
    X_w, y_w = make_nb_works_despite_violation()
    X_wtr, X_wte, y_wtr, y_wte = train_test_split(X_w.values, y_w.values, test_size=0.2, random_state=SEED)
    gnb_w = GaussianNaiveBayes()
    gnb_w.fit(X_wtr, y_wtr)
    acc_works = accuracy_score(y_wte, gnb_w.predict(X_wte))
    print(f"[NB] Works despite violation accuracy: {acc_works:.4f}")

    X_f, y_f = make_nb_fails_dataset()
    X_ftr, X_fte, y_ftr, y_fte = train_test_split(X_f.values, y_f.values, test_size=0.2, random_state=SEED)
    gnb_f = GaussianNaiveBayes()
    gnb_f.fit(X_ftr, y_ftr)
    acc_fails = accuracy_score(y_fte, gnb_f.predict(X_fte))
    print(f"[NB] Fails dataset accuracy: {acc_fails:.4f}")

    Xf_2d = X_f[["x0", "x1"]].values
    preds_f = gnb_f.predict(X_f.values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(Xf_2d[:, 0], Xf_2d[:, 1], c=y_f.values, cmap="bwr", s=8, alpha=0.6)
    axes[0].set_title("Ring Dataset: Ground Truth")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("x1")

    axes[1].scatter(Xf_2d[:, 0], Xf_2d[:, 1], c=preds_f, cmap="bwr", s=8, alpha=0.6)
    axes[1].set_title(f"Ring Dataset: NB Predictions (acc={acc_fails:.2f})")
    axes[1].set_xlabel("x0")
    axes[1].set_ylabel("x1")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/nb_counterexamples.png", dpi=120)
    plt.close()
    return acc_works, acc_fails


def run_nb_dataset_comparison():
    results = {}
    for name, loader in [
        ("Low Noise", make_low_noise_dataset),
        ("High Noise", make_high_noise_dataset),
        ("Correlated", make_correlated_features_dataset),
    ]:
        X_df, y = loader()
        X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y.values, test_size=0.2, random_state=SEED)
        gnb = GaussianNaiveBayes()
        gnb.fit(X_tr, y_tr)
        results[name] = accuracy_score(y_te, gnb.predict(X_te))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results.keys(), results.values(), color=["steelblue", "tomato", "goldenrod"], edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Naive Bayes Accuracy Across Datasets")
    for i, (k, v) in enumerate(results.items()):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/nb_comparison.png", dpi=120)
    plt.close()
    return results


if __name__ == "__main__":
    print("Running Naive Bayes experiments...")
    run_basic_nb()
    run_correlated_features_experiment()
    run_nb_counterexamples()
    run_nb_dataset_comparison()
    print("Naive Bayes experiments complete. Figures saved to figures/")
