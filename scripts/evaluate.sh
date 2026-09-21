#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

DATASET="${1:-$PROJECT_ROOT/data/processed/final_clean.csv}"
MODEL="${2:-$PROJECT_ROOT/models/final_model.json}"
RESULTS_DIR="${3:-$PROJECT_ROOT/results}"

log() {
    printf '[EVALUATE] %s\n' "$1"
}

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    exit 1
}

[[ -x "$PYTHON" ]] || fail "Python virtual environment not found. Run setup.sh first."
[[ -f "$DATASET" ]] || fail "Evaluation dataset not found: $DATASET"
[[ -f "$MODEL" ]] || fail "Model not found: $MODEL"

cd "$PROJECT_ROOT"

mkdir -p "$RESULTS_DIR"

log "Evaluation dataset: $DATASET"
log "Model: $MODEL"
log "Results directory: $RESULTS_DIR"
log "Running Python evaluation..."

"$PYTHON" -m ai_pipeline.cli.main evaluate \
    "$DATASET" \
    "$MODEL" \
    --results "$RESULTS_DIR"

log "Evaluation completed successfully."
log "Results: $RESULTS_DIR"
