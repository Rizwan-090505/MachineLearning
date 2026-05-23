import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataset_generator import (
    make_low_noise_dataset,
    make_high_noise_dataset,
    make_high_dimensional_dataset,
    make_greedy_counterexample_dataset,
)

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
SEED = 4


def entropy(y):
    n = len(y)
    if n == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    probs = counts[counts > 0] / n
    return -np.sum(probs * np.log2(probs + 1e-12))


def information_gain(y, mask):
    n = len(y)
    left, right = y[mask], y[~mask]
    ig = entropy(y) - (len(left) / n) * entropy(left) - (len(right) / n) * entropy(right)
    return ig


def split_information(mask):
    n = len(mask)
    n_l, n_r = mask.sum(), (~mask).sum()
    if n_l == 0 or n_r == 0:
        return 1e-10
    p_l, p_r = n_l / n, n_r / n
    return -(p_l * np.log2(p_l) + p_r * np.log2(p_r))


def gain_ratio(y, mask):
    ig = information_gain(y, mask)
    si = split_information(mask)
    return ig / (si + 1e-10)


def best_split(X, y, feature_indices):
    best_gr = -np.inf
    best_ig = -np.inf
    best_feat = None
    best_thresh = None

    for fi in feature_indices:
        col = X[:, fi]
        thresholds = np.unique(col)
        if len(thresholds) < 2:
            continue
        thresholds = (thresholds[:-1] + thresholds[1:]) / 2
        if len(thresholds) > 40:
            idx = np.linspace(0, len(thresholds) - 1, 40, dtype=int)
            thresholds = thresholds[idx]
        for t in thresholds:
            mask = col <= t
            if mask.sum() < 1 or (~mask).sum() < 1:
                continue
            gr = gain_ratio(y, mask)
            ig = information_gain(y, mask)
            if gr > best_gr:
                best_gr = gr
                best_ig = ig
                best_feat = fi
                best_thresh = t

    return best_feat, best_thresh, best_ig, best_gr


class TreeNode:
    def __init__(self, is_leaf=False, prediction=None, feature=None,
                 threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class C45DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self.n_features_ = X.shape[1]
        self.root_ = self._build(X, y, depth=0)
        return self

    def _leaf(self, y):
        counts = np.bincount(y)
        return TreeNode(is_leaf=True, prediction=np.argmax(counts))

    def _build(self, X, y, depth):
        if len(y) < self.min_samples_split:
            return self._leaf(y)
        if len(np.unique(y)) == 1:
            return self._leaf(y)
        if self.max_depth is not None and depth >= self.max_depth:
            return self._leaf(y)

        feat, thresh, ig, gr = best_split(X, y, range(self.n_features_))

        if feat is None or ig <= 0:
            return self._leaf(y)

        mask = X[:, feat] <= thresh
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return TreeNode(is_leaf=False, feature=feat, threshold=thresh, left=left, right=right)

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._predict_one(x, self.root_) for x in X])


def run_gain_ratio_analysis():
    X_df, y = make_low_noise_dataset()
    X, y_arr = X_df.values, y.values

    feature_indices = range(X.shape[1])
    igs, grs = [], []
    for fi in feature_indices:
        col = X[:, fi]
        thresholds = np.unique(col)
        thresholds = (thresholds[:-1] + thresholds[1:]) / 2
        best_ig_f, best_gr_f = -np.inf, -np.inf
        for t in thresholds:
            mask = col <= t
            if mask.sum() < 1 or (~mask).sum() < 1:
                continue
            ig = information_gain(y_arr, mask)
            gr = gain_ratio(y_arr, mask)
            if ig > best_ig_f:
                best_ig_f = ig
            if gr > best_gr_f:
                best_gr_f = gr
        igs.append(max(best_ig_f, 0))
        grs.append(max(best_gr_f, 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_pos = np.arange(len(igs))
    axes[0].bar(x_pos, igs, color="steelblue", edgecolor="black", linewidth=0.4)
    axes[0].set_title("Information Gain per Feature")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Information Gain")

    axes[1].bar(x_pos, grs, color="coral", edgecolor="black", linewidth=0.4)
    axes[1].set_title("Gain Ratio per Feature")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Gain Ratio")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/dt_gain_ratio_analysis.png", dpi=120)
    plt.close()
    print("[DT] Gain ratio analysis done.")
    return igs, grs


def run_overfitting_investigation():
    X_df, y = make_low_noise_dataset()
    X_tr, X_val, y_tr, y_val = train_test_split(X_df.values, y.values, test_size=0.25, random_state=SEED)
    depths = [1, 2, 3, 5, 8, 12, 20, 30]
    train_accs, val_accs = [], []

    for d in depths:
        tree = C45DecisionTree(max_depth=d, min_samples_split=2)
        tree.fit(X_tr, y_tr)
        train_accs.append(accuracy_score(y_tr, tree.predict(X_tr)))
        val_accs.append(accuracy_score(y_val, tree.predict(X_val)))

    labels = [str(d) for d in depths]
    x_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_pos, train_accs, "o-", label="Train Accuracy", color="steelblue")
    ax.plot(x_pos, val_accs, "s--", label="Validation Accuracy", color="tomato")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("C4.5 Decision Tree: Overfitting Investigation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/dt_overfitting.png", dpi=120)
    plt.close()
    print("[DT] Overfitting investigation done.")
    return train_accs, val_accs, labels


def run_noise_sensitivity():
    rng = np.random.RandomState(SEED)
    X_df, y = make_low_noise_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y.values, test_size=0.2, random_state=SEED)

    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    accs = []

    for nl in noise_levels:
        y_noisy = y_tr.copy()
        n_flip = int(nl * len(y_noisy))
        flip_idx = rng.choice(len(y_noisy), n_flip, replace=False)
        y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
        tree = C45DecisionTree(max_depth=8, min_samples_split=5)
        tree.fit(X_tr, y_noisy)
        accs.append(accuracy_score(y_te, tree.predict(X_te)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([nl * 100 for nl in noise_levels], accs, "o-", color="purple")
    ax.set_xlabel("Label Noise (%)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("C4.5 Sensitivity to Label Noise")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/dt_noise_sensitivity.png", dpi=120)
    plt.close()
    print("[DT] Noise sensitivity done.")
    return noise_levels, accs


def run_greedy_counterexample():
    X_df, y = make_greedy_counterexample_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y.values, test_size=0.2, random_state=SEED)

    tree_shallow = C45DecisionTree(max_depth=2, min_samples_split=5)
    tree_shallow.fit(X_tr, y_tr)
    acc_shallow = accuracy_score(y_te, tree_shallow.predict(X_te))

    tree_deep = C45DecisionTree(max_depth=8, min_samples_split=5)
    tree_deep.fit(X_tr, y_tr)
    acc_deep = accuracy_score(y_te, tree_deep.predict(X_te))

    print(f"[DT] Greedy counterexample: shallow acc={acc_shallow:.4f}, deep acc={acc_deep:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Depth=2 (Greedy)", "Depth=8"], [acc_shallow, acc_deep],
           color=["tomato", "steelblue"], edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Greedy vs Deep Tree on Counterexample Dataset")
    for i, v in enumerate([acc_shallow, acc_deep]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/dt_greedy_counterexample.png", dpi=120)
    plt.close()
    return acc_shallow, acc_deep


def run_dataset_comparison():
    datasets = {
        "Low Noise": make_low_noise_dataset,
        "High Noise": make_high_noise_dataset,
        "High Dimensional": make_high_dimensional_dataset,
    }
    results = {}
    for name, loader in datasets.items():
        X_df, y = loader()
        X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y.values, test_size=0.2, random_state=SEED)
        tree = C45DecisionTree(max_depth=8, min_samples_split=5)
        tree.fit(X_tr, y_tr)
        results[name] = accuracy_score(y_te, tree.predict(X_te))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results.keys(), results.values(),
           color=["steelblue", "tomato", "goldenrod"], edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("C4.5 Accuracy Across Datasets")
    for i, (k, v) in enumerate(results.items()):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/dt_comparison.png", dpi=120)
    plt.close()
    print("[DT] Dataset comparison done.")
    return results


if __name__ == "__main__":
    print("Running Decision Tree experiments...")
    run_gain_ratio_analysis()
    run_overfitting_investigation()
    run_noise_sensitivity()
    run_greedy_counterexample()
    run_dataset_comparison()
    print("Decision Tree experiments complete. Figures saved to figures/")
