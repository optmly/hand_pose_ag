#!/usr/bin/env bash
# ── new_version.sh — Create a new version manifest ────────────────────
#
# Usage:
#   bash scripts/new_version.sh 0.1 "SAM3 Hand Tracking"
#   bash scripts/new_version.sh 1.0 "Production Release"
#
set -euo pipefail

VERSION="${1:?Usage: new_version.sh <version> <title>}"
TITLE="${2:?Usage: new_version.sh <version> <title>}"
DATE=$(date +%Y-%m-%d)
FILE="versions/v${VERSION}.md"

if [[ -f "$FILE" ]]; then
    echo "❌ Version manifest already exists: $FILE"
    exit 1
fi

cat > "$FILE" << EOF
# v${VERSION} — ${TITLE}

## Release Date
${DATE}

## Summary
<!-- Brief description of what this version accomplishes -->

## Features Added
<!-- List of features added or updated -->
- 

## Features Updated
<!-- List of existing features that were modified -->
- 

## How to Run
\`\`\`bash
# Step-by-step instructions to run this version
\`\`\`

## Dependencies
- Python 3.11+

## Breaking Changes
<!-- Any breaking changes from previous versions -->
None.

## Notes
<!-- Additional context, known issues, etc. -->
EOF

echo "✅ Created version manifest: $FILE"
echo "   Edit it to fill in the details."
