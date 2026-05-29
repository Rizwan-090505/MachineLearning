"""
q1_bagging.py
Question 1: Bagging and Instability Analysis
Roll Number: BSAI23004
"""

import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset_generator import (
    make_high_noise_dataset,
    make_bagging_helps_dataset,
    make_bagging_fails_dataset,
)
from decision_tree import DecisionTreeClassifier
from bagging import BaggingClassifier, bootstrap_sample

SEED = 4
PLOTS = "plots"
os.makedirs(PLOTS, exist_ok=True)

# ─── Part A ───────────────────────────────────────────────────────────────────


def part_a():
    print("\n[Q1-A] Bootstrap Sampling Visualization")
    n_total = 30
    X_demo = np.arange(n_total).reshape(-1, 1).astype(float)
    y_demo = (X_demo.flatten() > 15).astype(int)

    n_rounds = 5
    fig, axes = plt.subplots(n_rounds, 1, figsize=(12, 9))
    fig.suptitle(
        "Bootstrap Sampling: Selected (Blue) vs OOB (Red)",
        fontsize=13,
        fontweight="bold",
    )

    for r in range(n_rounds):
        _, _, oob_mask, indices = bootstrap_sample(
            X_demo, y_demo, random_state=SEED + r
        )
        counts = np.bincount(indices, minlength=n_total)
        colors = ["#2196F3" if counts[i] > 0 else "#F44336" for i in range(n_total)]
        axes[r].bar(
            np.arange(n_total), counts, color=colors, alpha=0.8, edgecolor="white"
        )
        axes[r].set_title(
            f"Round {r + 1}: OOB fraction={oob_mask.mean():.3f}", fontsize=9
        )
        axes[r].set_ylabel("Count")
    axes[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q1a_bootstrap_rounds.png", dpi=110, bbox_inches="tight")
    plt.close()

    oob_fracs = [
        bootstrap_sample(X_demo, y_demo, random_state=r)[2].mean() for r in range(20)
    ]
    fig2, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(oob_fracs, "o-", color="#E91E63", linewidth=2)
    ax.axhline(1 / np.e, color="gray", linestyle="--", label=f"Theory ≈ {1 / np.e:.3f}")
    ax.set_title("OOB Fraction per Bootstrap Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("OOB Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q1a_oob_fraction.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Mean OOB fraction = {np.mean(oob_fracs):.3f} (theory={1 / np.e:.3f})")
    print("  [✓] q1a done")


# ─── Part B ───────────────────────────────────────────────────────────────────


def part_b():
    print("\n[Q1-B] Bagging Experiments")
    X, y = make_high_noise_dataset(n=600)
    X, y = X.values, y.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)

    # Exp 1: n_trees (reduced range for speed)
    n_tree_range = [1, 5, 10, 15, 20]
    tr_accs, va_accs, variances = [], [], []
    for nt in n_tree_range:
        bag = BaggingClassifier(n_estimators=nt, max_depth=4, random_state=SEED)
        bag.fit(X_tr, y_tr)
        tr_accs.append(accuracy_score(y_tr, bag.predict(X_tr)))
        va_accs.append(accuracy_score(y_te, bag.predict(X_te)))
        run_accs = bag.variance_across_runs(
            X_tr, y_tr, n_runs=2, test_X=X_te, test_y=y_te
        )
        variances.append(run_accs.var())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Bagging Experiments (High-Noise, n=600)", fontsize=12, fontweight="bold"
    )

    axes[0].plot(n_tree_range, tr_accs, "b-o", label="Train", linewidth=2)
    axes[0].plot(n_tree_range, va_accs, "r-o", label="Val", linewidth=2)
    axes[0].set_xlabel("# Trees")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Train vs Val Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n_tree_range, variances, "g-o", linewidth=2)
    axes[1].set_xlabel("# Trees")
    axes[1].set_ylabel("Variance")
    axes[1].set_title("Variance Across Runs")
    axes[1].grid(True, alpha=0.3)

    # Exp 2: depth (reduced range)
    depth_range = [1, 2, 3, 4, 5]
    dtr, dva = [], []
    for d in depth_range:
        bag = BaggingClassifier(n_estimators=10, max_depth=d, random_state=SEED)
        bag.fit(X_tr, y_tr)
        dtr.append(accuracy_score(y_tr, bag.predict(X_tr)))
        dva.append(accuracy_score(y_te, bag.predict(X_te)))
    axes[2].plot(depth_range, dtr, "b-o", label="Train", linewidth=2)
    axes[2].plot(depth_range, dva, "r-o", label="Val", linewidth=2)
    axes[2].set_xlabel("Max Depth")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Effect of Tree Depth (10 trees)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q1b_bagging_experiments.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Exp 3: dataset size (reduced range)
    sizes = [100, 150, 200, 250, 300]
    str_accs, sva_accs = [], []
    for sz in sizes:
        bag = BaggingClassifier(n_estimators=10, max_depth=4, random_state=SEED)
        bag.fit(X_tr[:sz], y_tr[:sz])
        str_accs.append(accuracy_score(y_tr[:sz], bag.predict(X_tr[:sz])))
        sva_accs.append(accuracy_score(y_te, bag.predict(X_te)))

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes, str_accs, "b-o", label="Train", linewidth=2)
    ax.plot(sizes, sva_accs, "r-o", label="Val", linewidth=2)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Effect of Dataset Size (10 trees, depth=4)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q1b_dataset_size.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Best val acc: {max(va_accs):.4f}")
    print("  [✓] q1b done")


# ─── Part C ───────────────────────────────────────────────────────────────────


def part_c():
    print("\n[Q1-C] Failure Analysis")
    results = {}
    for name, (Xdf, ys) in [
        ("Helps (High Noise)", make_bagging_helps_dataset(n=600)),
        ("Fails (Low Variance)", make_bagging_fails_dataset(n=600)),
    ]:
        X, y = Xdf.values, ys.values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=SEED
        )
        single = DecisionTreeClassifier(max_depth=4, random_state=SEED)
        single.fit(X_tr, y_tr)
        s_acc = accuracy_score(y_te, single.predict(X_te))
        bag = BaggingClassifier(n_estimators=10, max_depth=4, random_state=SEED)
        bag.fit(X_tr, y_tr)
        b_acc = accuracy_score(y_te, bag.predict(X_te))
        results[name] = {"single": s_acc, "bag": b_acc, "gain": b_acc - s_acc}
        print(
            f"  {name}: Single={s_acc:.4f}, Bagging={b_acc:.4f}, Gain={b_acc - s_acc:+.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bagging: Helps vs Fails", fontsize=13, fontweight="bold")
    for ax, (name, res) in zip(axes, results.items()):
        bars = ax.bar(
            ["Single Tree", "Bagging(10)"],
            [res["single"], res["bag"]],
            color=["#FF7043", "#42A5F5"],
            width=0.5,
        )
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Test Accuracy")
        ax.set_title(name)
        for bar, v in zip(bars, [res["single"], res["bag"]]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.005,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        color = "#4CAF50" if res["gain"] > 0 else "#F44336"
        ax.text(
            0.5,
            0.05,
            f"Gain: {res['gain']:+.4f}",
            transform=ax.transAxes,
            ha="center",
            color=color,
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/q1c_failure_analysis.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Correlation plots
    for ds_name, (Xdf, ys) in [
        ("High_Noise_Bagging_Helps", make_bagging_helps_dataset(n=300)),
        ("Low_Variance_Bagging_Fails", make_bagging_fails_dataset(n=300)),
    ]:
        X, y = Xdf.values, ys.values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=SEED
        )
        bag = BaggingClassifier(n_estimators=6, max_depth=4, random_state=SEED)
        bag.fit(X_tr, y_tr)
        preds = np.array([t.predict(X_te) for t in bag.estimators_])
        corr = np.corrcoef(preds)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Tree Correlation\n{ds_name.replace('_', ' ')}", fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"{PLOTS}/q1c_correlation_{ds_name}.png", dpi=110, bbox_inches="tight"
        )
        plt.close()

    print("  [✓] q1c done")


def run_q1():
    print("=" * 60 + "\n  QUESTION 1: BAGGING\n" + "=" * 60)
    part_a()
    part_b()
    part_c()
    print("[Q1] All done.")


if __name__ == "__main__":
    run_q1()
