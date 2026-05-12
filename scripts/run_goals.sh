#!/usr/bin/env bash
# ── run_goals.sh — Goal execution launcher ────────────────────────────
#
# Displays current goals for Claude, and after completion,
# archives them and runs commit_and_push.
#
# Usage:
#   bash scripts/run_goals.sh            # Show goals
#   bash scripts/run_goals.sh --archive  # Archive completed goals and commit
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GOALS_FILE="goals/current_goals.yaml"
HISTORY_DIR="goals/goal_history"

if [[ ! -f "$GOALS_FILE" ]]; then
    echo "❌ No goals file found at $GOALS_FILE"
    echo "   Create one to define goals for Claude."
    exit 1
fi

if [[ "${1:-}" == "--archive" ]]; then
    # Archive completed goals
    mkdir -p "$HISTORY_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    VERSION=$(grep '^version:' "$GOALS_FILE" | sed 's/version: *"\(.*\)"/\1/')
    ARCHIVE_NAME="v${VERSION}_${TIMESTAMP}.yaml"
    
    cp "$GOALS_FILE" "${HISTORY_DIR}/${ARCHIVE_NAME}"
    echo "📁 Archived goals to ${HISTORY_DIR}/${ARCHIVE_NAME}"
    
    # Create a fresh goals template
    cat > "$GOALS_FILE" << 'EOF'
# ── Claude Goal Specification ──────────────────────────────────────────
# Edit this file to define goals for the next Claude session.

version: "NEXT_VERSION"
title: "TITLE"
description: >
  Describe what this set of goals accomplishes.

goals:
  - id: G1
    description: "Goal description"
    priority: high
    status: pending
    acceptance_criteria:
      - "Criterion 1"
      - "Criterion 2"

constraints: []

on_complete: "commit_and_push"
EOF
    echo "📝 Reset goals file — edit it for the next session."
    
    # Auto-commit
    echo ""
    bash scripts/commit_and_push.sh "Goals v${VERSION} completed"
else
    # Display goals for Claude
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  CURRENT GOALS FOR CLAUDE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    cat "$GOALS_FILE"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  To archive after completion: bash scripts/run_goals.sh --archive"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi
