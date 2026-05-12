#!/usr/bin/env bash
# ── commit_and_push.sh — Auto-commit and push to GitHub ───────────────
#
# Usage:
#   bash scripts/commit_and_push.sh                    # auto-generate message
#   bash scripts/commit_and_push.sh "custom message"   # custom message
#   bash scripts/commit_and_push.sh --dry-run           # preview only
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false
MESSAGE=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) MESSAGE="$arg" ;;
    esac
done

# ── Detect current version from latest manifest ──────────────────────
LATEST_VERSION=$(ls -1 versions/v*.md 2>/dev/null | sort -V | tail -1 | sed 's|versions/v\(.*\)\.md|\1|')
if [[ -z "$LATEST_VERSION" ]]; then
    LATEST_VERSION="0.0"
fi

# ── Auto-generate commit message if not provided ─────────────────────
if [[ -z "$MESSAGE" ]]; then
    # Extract title from version manifest
    MANIFEST="versions/v${LATEST_VERSION}.md"
    if [[ -f "$MANIFEST" ]]; then
        TITLE=$(head -1 "$MANIFEST" | sed 's/^# //')
        MESSAGE="${TITLE}"
    else
        MESSAGE="v${LATEST_VERSION}: update"
    fi
fi

# ── Show status ──────────────────────────────────────────────────────
echo "━━━ Commit Summary ━━━"
echo "  Version: v${LATEST_VERSION}"
echo "  Message: ${MESSAGE}"
echo ""
echo "━━━ Changed Files ━━━"
git status --short
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "🔍 Dry run — no changes committed."
    exit 0
fi

# ── Stage, commit, tag, push ─────────────────────────────────────────
git add -A
git commit -m "${MESSAGE}" || { echo "Nothing to commit."; exit 0; }

# Tag if this version isn't already tagged
TAG="v${LATEST_VERSION}"
if ! git tag -l "$TAG" | grep -q "$TAG"; then
    git tag -a "$TAG" -m "${MESSAGE}"
    echo "🏷️  Tagged as ${TAG}"
fi

# Push (with tags)
if git remote | grep -q origin; then
    git push origin main --tags 2>/dev/null || git push origin master --tags 2>/dev/null || {
        echo "⚠️  Push failed — is the remote configured?"
        echo "   Run: git remote add origin git@github.com:optmly/hand_pose_ag.git"
    }
else
    echo "⚠️  No remote 'origin' configured."
    echo "   Run: git remote add origin git@github.com:optmly/hand_pose_ag.git"
fi

echo ""
echo "✅ Committed and pushed v${LATEST_VERSION}"
