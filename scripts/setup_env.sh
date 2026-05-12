#!/usr/bin/env bash
# ── setup_env.sh — Set up development environment ─────────────────────
#
# Usage:
#   bash scripts/setup_env.sh
#
set -euo pipefail

ENV_NAME="hand_pose"

echo "━━━ Hand Pose Pipeline — Environment Setup ━━━"
echo ""

# Check if conda is available
if ! command -v conda &>/dev/null; then
    echo "❌ conda not found. Please install Miniforge/Miniconda first."
    exit 1
fi

# Create or update conda environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "📦 Environment '${ENV_NAME}' exists — updating..."
    pip install -r requirements.txt
else
    echo "📦 Creating conda environment '${ENV_NAME}'..."
    conda create -n "$ENV_NAME" python=3.11 -y
    echo "📦 Installing dependencies..."
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    pip install -r requirements.txt
fi

echo ""
echo "✅ Environment ready!"
echo "   Activate with: conda activate ${ENV_NAME}"
