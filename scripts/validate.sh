#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
DATASET="${1:-$PROJECT_ROOT/data/sample/classification.csv}"
LABEL="${2:-label}"

log() {
    printf '[VALIDATE] %s\n' "$1"
}

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    exit 1
}

[[ -x "$PYTHON" ]] || fail "Python virtual environment not found. Run setup.sh first."
[[ -f "$DATASET" ]] || fail "Dataset not found: $DATASET"

cd "$PROJECT_ROOT"

log "Dataset: $DATASET"
log "Label column: $LABEL"
log "Running Python dataset validation..."

"$PYTHON" -m ai_pipeline.cli.main validate "$DATASET" --label "$LABEL"

log "Dataset validation completed successfully."
