#!/usr/bin/env bash
# run_uv.sh — Ensemble Methods Assignment (BSAI23004)
# Usage: bash run_uv.sh

set -e

echo "=================================================="
echo "  Ensemble Methods Assignment — BSAI23004"
echo "=================================================="

# Check uv
if ! command -v uv &> /dev/null; then
    echo "[ERROR] uv is not installed."
    echo ""
    echo "Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found."
    exit 1
fi

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating virtual environment..."
    uv venv
fi

# Activate environment
source .venv/bin/activate

# Install dependencies
echo "[INFO] Syncing dependencies..."

uv pip install \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    pillow \
    --quiet

echo "[INFO] All dependencies satisfied."

# Run main pipeline
echo ""
echo "[RUN] Starting pipeline..."

uv run main.py

echo ""
echo "=================================================="
echo "  Done! Check:"
echo "    BSAI23004_Ensemble_Report.pdf"
echo "    plots/  (all figures)"
echo "    datasets/  (generated CSVs)"
echo "=================================================="
