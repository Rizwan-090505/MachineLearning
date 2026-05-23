#!/usr/bin/env bash
set -e

echo "ML Assignment 2 — UV Run"

uv sync

uv run python dataset_generator.py
uv run python kmeans.py
uv run python naive_bayes.py
uv run python decision_tree.py
uv run python generate_report.py
