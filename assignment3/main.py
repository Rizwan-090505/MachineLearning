"""
main.py
Master script: runs dataset generation, all experiments, and report generation.
Roll Number: BSAI23004
"""

import time


def main():
    t_start = time.time()
    print("=" * 70)
    print("  ENSEMBLE METHODS ASSIGNMENT — BSAI23004")
    print("  Seed = 4  (last 3 digits of BSAI23004)")
    print("=" * 70)

    # Step 1: Generate datasets
    print("\n[STEP 1] Generating datasets...")
    from dataset_generator import save_all

    save_all()

    # Step 2: Q1 — Bagging
    print("\n[STEP 2] Running Question 1: Bagging...")
    from q1_bagging import run_q1

    run_q1()

    # Step 3: Q2 — Random Forest
    print("\n[STEP 3] Running Question 2: Random Forest...")
    from q2_random_forest import run_q2

    run_q2()

    # Step 4: Q3 — AdaBoost
    print("\n[STEP 4] Running Question 3: AdaBoost...")
    from q3_adaboost import run_q3

    run_q3()

    # Step 5: Generate PDF report
    print("\n[STEP 5] Generating PDF report...")
    from generate_report import generate_report

    generate_report()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  ALL DONE in {elapsed:.1f}s")
    print("  Outputs:")
    print("    plots/             — all experiment figures")
    print("    datasets/          — generated CSV datasets")
    print("    BSAI23004_Ensemble_Report.pdf  — full report")
    print("=" * 70)


if __name__ == "__main__":
    main()
