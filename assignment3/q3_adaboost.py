"""
q3_adaboost.py
Question 3: Boosting and Sequential Error Correction
Roll Number: BSAI23004
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset_generator import (
    make_low_noise_dataset,
    make_high_noise_dataset,
    make_boosting_noise_dataset,
)
from adaboost import AdaBoostClassifier

SEED = 4
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  Part A: AdaBoost Implementation Verification
# ─────────────────────────────────────────────


def part_a_adaboost_basic():
    print("\n[Q3-A] AdaBoost Basic Verification")
    X, y = make_low_noise_dataset(n=1200)
    X, y = X.values, y.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)

    boost = AdaBoostClassifier(
        n_estimators=50, learning_rate=1.0, base_depth=1, random_state=SEED
    )
    boost.fit(X_tr, y_tr)

    train_curve = boost.staged_score(X_tr, y_tr)
    val_curve = boost.staged_score(X_te, y_te)
    rounds = range(1, len(train_curve) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "AdaBoost (Decision Stumps) – Low Noise Dataset", fontsize=12, fontweight="bold"
    )

    axes[0].plot(rounds, train_curve, "b-", linewidth=2, label="Train")
    axes[0].plot(rounds, val_curve, "r-", linewidth=2, label="Val")
    axes[0].set_xlabel("Boosting Rounds")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Learning Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, boost.estimator_errors_, "g-o", markersize=3, linewidth=1.5)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Weighted Error")
    axes[1].set_title("Weak Learner Error per Round")
    axes[1].axhline(0.5, color="gray", linestyle="--", label="Random Baseline")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rounds, boost.alphas_, "purple", linewidth=2)
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Alpha (Learner Weight)")
    axes[2].set_title("Alpha per Boosting Round")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/q3a_adaboost_basic.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"  Final Train Acc: {train_curve[-1]:.4f}, Val Acc: {val_curve[-1]:.4f}")
    print("  [✓] Saved q3a_adaboost_basic.png")


# ─────────────────────────────────────────────
#  Part B: Noise Sensitivity
# ─────────────────────────────────────────────


def part_b_noise_sensitivity():
    print("\n[Q3-B] Noise Sensitivity Analysis")
    noise_levels = [0.05, 0.15, 0.25]
    labels = ["5% Noise", "15% Noise", "25% Noise"]
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("AdaBoost Noise Sensitivity", fontsize=13, fontweight="bold")

    results = {}
    for col, (flip, label, color) in enumerate(zip(noise_levels, labels, colors)):
        X, y = make_boosting_noise_dataset(n=1200, flip_prob=flip)
        X, y = X.values, y.values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=SEED
        )

        boost = AdaBoostClassifier(n_estimators=50, base_depth=1, random_state=SEED)
        boost.fit(X_tr, y_tr)

        tr_curve = boost.staged_score(X_tr, y_tr)
        va_curve = boost.staged_score(X_te, y_te)
        rounds = range(1, 51)

        axes[col].plot(rounds, tr_curve, "b-", linewidth=2, label="Train")
        axes[col].plot(rounds, va_curve, "r-", linewidth=2, label="Val")
        axes[col].set_title(f"{label} – Learning Curve", fontsize=10)
        axes[col].set_xlabel("Rounds")
        axes[col].set_ylabel("Accuracy")
        axes[col].legend(fontsize=8)
        axes[col].grid(True, alpha=0.3)

        results[label] = {
            "final_train": tr_curve[-1],
            "final_val": va_curve[-1],
            "max_val": max(va_curve),
            "peak_round": int(np.argmax(va_curve)) + 1,
        }
        print(
            f"  {label}: Train={tr_curve[-1]:.4f}, Val={va_curve[-1]:.4f}, "
            f"Best Val={max(va_curve):.4f} @ round {int(np.argmax(va_curve)) + 1}"
        )

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/q3b_noise_sensitivity.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Summary bar chart
    fig2, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    w = 0.3
    ax.bar(
        x - w / 2,
        [results[l]["final_train"] for l in labels],
        width=w,
        label="Final Train",
        color="#42A5F5",
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        [results[l]["final_val"] for l in labels],
        width=w,
        label="Final Val",
        color="#EF5350",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("AdaBoost Final Accuracy vs Noise Level")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/q3b_noise_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  [✓] Saved q3b_noise_sensitivity.png, q3b_noise_summary.png")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/q3b_noise_sensitivity.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Summary bar chart
    fig2, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    w = 0.3
    ax.bar(
        x - w / 2,
        [results[l]["final_train"] for l in labels],
        width=w,
        label="Final Train",
        color="#42A5F5",
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        [results[l]["final_val"] for l in labels],
        width=w,
        label="Final Val",
        color="#EF5350",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("AdaBoost Final Accuracy vs Noise Level")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/q3b_noise_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  [✓] Saved q3b_noise_sensitivity.png, q3b_noise_summary.png")


# ─────────────────────────────────────────────
#  Part C: Weak vs Strong Learner
# ─────────────────────────────────────────────


def part_c_weak_vs_strong():
    print("\n[Q3-C] Weak Learner (Stump) vs Strong Learner (Deep Tree)")
    X, y = make_low_noise_dataset(n=1200)
    X, y = X.values, y.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)

    configs = [
        ("Stump (depth=1)", 1),
        ("Shallow (depth=2)", 2),
        ("Deep (depth=4)", 4),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "AdaBoost: Effect of Base Learner Depth", fontsize=12, fontweight="bold"
    )

    summary = {}
    for ax, (label, depth) in zip(axes, configs):
        boost = AdaBoostClassifier(n_estimators=40, base_depth=depth, random_state=SEED)
        boost.fit(X_tr, y_tr)
        tr_curve = boost.staged_score(X_tr, y_tr)
        va_curve = boost.staged_score(X_te, y_te)
        rounds = range(1, len(tr_curve) + 1)

        ax.plot(rounds, tr_curve, "b-", linewidth=2, label="Train")
        ax.plot(rounds, va_curve, "r-", linewidth=2, label="Val")
        ax.set_title(f"{label}", fontsize=10)
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        summary[label] = {
            "final_train": tr_curve[-1],
            "final_val": va_curve[-1],
            "best_val": max(va_curve),
            "best_round": int(np.argmax(va_curve)) + 1,
        }
        print(
            f"  {label}: Train={tr_curve[-1]:.4f}, Val={va_curve[-1]:.4f}, "
            f"Best Val={max(va_curve):.4f} @ round {int(np.argmax(va_curve)) + 1}"
        )

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/q3c_weak_vs_strong.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  [✓] Saved q3c_weak_vs_strong.png")


# ─────────────────────────────────────────────
#  Part D: Comparative Reflection
# ─────────────────────────────────────────────


def part_d_comparative():
    print("\n[Q3-D] Comparative Reflection (Bagging vs Boosting)")
    from random_forest import RandomForestClassifier
    from bagging import BaggingClassifier

    configs_ds = [
        ("Low Noise", make_low_noise_dataset(n=1200)),
        ("High Noise", make_high_noise_dataset(n=600)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Bagging vs Boosting: Accuracy Comparison", fontsize=12, fontweight="bold"
    )

    for ax, (ds_name, (X_df, y_s)) in zip(axes, configs_ds):
        X, y = X_df.values, y_s.values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=SEED
        )

        bag = BaggingClassifier(n_estimators=20, max_depth=4, random_state=SEED)
        bag.fit(X_tr, y_tr)

        rf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=SEED)
        rf.fit(X_tr, y_tr)

        boost = AdaBoostClassifier(n_estimators=40, base_depth=1, random_state=SEED)
        boost.fit(X_tr, y_tr)

        names = ["Bagging", "Random Forest", "AdaBoost"]
        accs = [
            accuracy_score(y_te, bag.predict(X_te)),
            accuracy_score(y_te, rf.predict(X_te)),
            accuracy_score(y_te, boost.predict(X_te)),
        ]
        colors = ["#42A5F5", "#66BB6A", "#EF5350"]
        bars = ax.bar(names, accs, color=colors, width=0.5, edgecolor="white")
        ax.set_title(f"{ds_name}", fontsize=11)
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(0.5, 1.0)
        for bar, val in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.grid(True, alpha=0.3, axis="y")
        print(f"  {ds_name}: {dict(zip(names, [f'{a:.4f}' for a in accs]))}")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/q3d_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  [✓] Saved q3d_comparison.png")
    print("  [✓] Saved q3d_comparison.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────


def run_q3():
    print("=" * 60)
    print("  QUESTION 3: BOOSTING AND SEQUENTIAL ERROR CORRECTION")
    print("=" * 60)
    part_a_adaboost_basic()
    part_b_noise_sensitivity()
    part_c_weak_vs_strong()
    part_d_comparative()
    print("\n[Q3] All done.")


if __name__ == "__main__":
    run_q3()
