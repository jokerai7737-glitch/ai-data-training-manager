#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

DATASET="${1:-$PROJECT_ROOT/data/processed/final_clean.csv}"
MODEL="${2:-$PROJECT_ROOT/models/final_model.json}"
LABEL="${3:-label}"
K="${4:-3}"
TRAIN_RATIO="${5:-0.8}"

log() {
    printf '[TRAIN] %s\n' "$1"
}

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    exit 1
}

[[ -x "$PYTHON" ]] || fail "Python virtual environment not found. Run setup.sh first."
[[ -f "$DATASET" ]] || fail "Training dataset not found: $DATASET"

cd "$PROJECT_ROOT"

mkdir -p "$(dirname "$MODEL")"

log "Training dataset: $DATASET"
log "Model output: $MODEL"
log "Label column: $LABEL"
log "K: $K"
log "Train ratio: $TRAIN_RATIO"
log "Running Python training..."

"$PYTHON" -m ai_pipeline.cli.main train \
    "$DATASET" \
    "$MODEL" \
    --label "$LABEL" \
    --k "$K" \
    --train-ratio "$TRAIN_RATIO"

log "Training completed successfully."
log "Model: $MODEL"
