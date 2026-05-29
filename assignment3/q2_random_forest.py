"""
q2_random_forest.py — Question 2: Random Forests
Roll Number: BSAI23004
"""

import numpy as np
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset_generator import (
    make_low_noise_dataset,
    make_high_noise_dataset,
    make_feature_dominant_dataset,
    make_high_dimensional_dataset,
)
from random_forest import RandomForestClassifier
from bagging import BaggingClassifier

SEED = 4
PLOTS = "plots"
os.makedirs(PLOTS, exist_ok=True)


def part_a():
    print("\n[Q2-A] RF vs Bagging")
    X, y = make_high_noise_dataset(n=600)
    X, y = X.values, y.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)

    n_range = [5, 10, 15, 20]
    bag_accs, rf_accs = [], []
    for n in n_range:
        bag = BaggingClassifier(n_estimators=n, max_depth=4, random_state=SEED)
        bag.fit(X_tr, y_tr)
        bag_accs.append(accuracy_score(y_te, bag.predict(X_te)))
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=4, max_features="sqrt", random_state=SEED
        )
        rf.fit(X_tr, y_tr)
        rf_accs.append(accuracy_score(y_te, rf.predict(X_te)))
        print(f"  n={n}: Bag={bag_accs[-1]:.4f} RF={rf_accs[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_range, bag_accs, "b-o", label="Bagging", linewidth=2)
    ax.plot(n_range, rf_accs, "g-s", label="Random Forest", linewidth=2)
    ax.set_xlabel("# Trees")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Bagging vs Random Forest (High-Noise, n=600)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q2a_rf_vs_bagging.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  [✓] q2a done")


def part_b():
    print("\n[Q2-B] Feature Dominance")
    X_df, y_s = make_feature_dominant_dataset(n=600)
    X, y = X_df.values, y_s.values
    n_feat = X.shape[1]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)

    configs = [("sqrt(d)", "sqrt"), ("log2(d)", "log2"), ("All", None), ("2", 2)]
    feat_counts_all, accs = {}, {}

    for label, mf in configs:
        rf = RandomForestClassifier(
            n_estimators=10, max_depth=4, max_features=mf, random_state=SEED
        )
        rf.fit(X_tr, y_tr)
        accs[label] = accuracy_score(y_te, rf.predict(X_te))
        feat_counts_all[label] = rf.feature_usage_counts()
        print(f"  max_features={label}: acc={accs[label]:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Feature Dominance: Feature Usage per max_features Setting",
        fontsize=12,
        fontweight="bold",
    )
    for ax, (label, counts) in zip(axes.flatten(), feat_counts_all.items()):
        colors = ["#F44336" if i == 0 else "#42A5F5" for i in range(n_feat)]
        ax.bar(range(n_feat), counts, color=colors, alpha=0.85)
        ax.set_title(f"max_features={label}  acc={accs[label]:.4f}", fontsize=10)
        ax.set_xlabel("Feature (Red=Dominant 0)")
        ax.set_ylabel("Usage Count")
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q2b_feature_dominance.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Tree diversity correlation
    rf_sqrt = RandomForestClassifier(
        n_estimators=6, max_depth=4, max_features="sqrt", random_state=SEED
    )
    rf_sqrt.fit(X_tr, y_tr)
    preds = rf_sqrt.get_tree_predictions(X_te)
    corr = np.corrcoef(preds)
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
    im = ax2.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax2)
    ax2.set_title("Tree Prediction Correlation\n(Feature Dominant, sqrt)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q2b_tree_diversity.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  [✓] q2b done")


def part_c():
    print("\n[Q2-C] OOB Error")
    X, y = make_low_noise_dataset(n=600)
    X, y = X.values, y.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)

    n_range = [5, 10, 15, 20]
    oob_scores, val_scores = [], []
    for n in n_range:
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=4, max_features="sqrt", random_state=SEED
        )
        rf.fit(X_tr, y_tr)
        oob = rf.oob_score(X_tr, y_tr)
        val = accuracy_score(y_te, rf.predict(X_te))
        oob_scores.append(oob)
        val_scores.append(val)
        print(f"  n={n}: OOB={oob:.4f} Val={val:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_range, oob_scores, "orange", marker="s", linewidth=2, label="OOB")
    ax.plot(n_range, val_scores, "steelblue", marker="o", linewidth=2, label="Val")
    ax.set_xlabel("# Trees")
    ax.set_ylabel("Accuracy")
    ax.set_title("OOB vs Validation Accuracy (Low-Noise)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q2c_oob_vs_val.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  [✓] q2c done")


def part_d():
    print("\n[Q2-D] High-Dimensional (5 info + 50 noise)")
    X_df, y_s = make_high_dimensional_dataset(n=400, n_informative=5, n_noise=50)
    X, y = X_df.values, y_s.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)

    n_range = [5, 10, 15, 20]
    times, tr_accs, va_accs, oob_sc = [], [], [], []
    for n in n_range:
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=4, max_features="sqrt", random_state=SEED
        )
        rf.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        ta = accuracy_score(y_tr, rf.predict(X_tr))
        va = accuracy_score(y_te, rf.predict(X_te))
        oob = rf.oob_score(X_tr, y_tr)
        times.append(elapsed)
        tr_accs.append(ta)
        va_accs.append(va)
        oob_sc.append(oob)
        print(
            f"  n={n}: Train={ta:.3f} Val={va:.3f} OOB={oob:.3f} t={elapsed:.2f}s"
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "RF High-Dimensional (5 info + 50 noise)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].plot(n_range, tr_accs, "b-o", label="Train", linewidth=2)
    axes[0, 0].plot(n_range, va_accs, "r-o", label="Val", linewidth=2)
    axes[0, 0].plot(n_range, oob_sc, "g--s", label="OOB", linewidth=2)
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(n_range, times, "m-o", linewidth=2)
    axes[0, 1].set_title("Training Time (s)")
    axes[0, 1].grid(True, alpha=0.3)
    gap = np.array(tr_accs) - np.array(va_accs)
    axes[1, 0].plot(n_range, gap, "r-o", linewidth=2)
    axes[1, 0].axhline(0, color="gray", linestyle="--")
    axes[1, 0].set_title("Overfitting Gap (Train-Val)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(n_range, tr_accs, "b-o", alpha=0.5, linewidth=2)
    axes[1, 1].set_title("Training Accuracy")
    axes[1, 1].grid(True, alpha=0.3)
    for ax in axes.flatten():
        ax.set_xlabel("# Trees")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q2d_high_dimensional.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  [✓] q2d done")


def run_q2():
    print("=" * 60 + "\n  QUESTION 2: RANDOM FORESTS\n" + "=" * 60)
    part_a()
    part_b()
    part_c()
    part_d()
    print("[Q2] All done.")


if __name__ == "__main__":
    run_q2()
