#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

INPUT_DATASET="${1:-$PROJECT_ROOT/data/sample/classification.csv}"
OUTPUT_DATASET="${2:-$PROJECT_ROOT/data/processed/final_clean.csv}"
LABEL="${3:-label}"

log() {
    printf '[PREPROCESS] %s\n' "$1"
}

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    exit 1
}

[[ -x "$PYTHON" ]] || fail "Python virtual environment not found. Run setup.sh first."
[[ -f "$INPUT_DATASET" ]] || fail "Input dataset not found: $INPUT_DATASET"

cd "$PROJECT_ROOT"

mkdir -p "$(dirname "$OUTPUT_DATASET")"

log "Input dataset: $INPUT_DATASET"
log "Output dataset: $OUTPUT_DATASET"
log "Label column: $LABEL"
log "Running Python preprocessing..."

"$PYTHON" -m ai_pipeline.cli.main preprocess \
    "$INPUT_DATASET" \
    "$OUTPUT_DATASET" \
    --label "$LABEL"

log "Preprocessing completed successfully."
log "Output: $OUTPUT_DATASET"
