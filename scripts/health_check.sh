#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

log() {
    printf '[HEALTH] %s\n' "$1"
}

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    exit 1
}

check_file() {
    [[ -f "$1" ]] || fail "Required file not found: $1"
}

check_directory() {
    [[ -d "$1" ]] || fail "Required directory not found: $1"
}

cd "$PROJECT_ROOT"

log "Project root: $PROJECT_ROOT"

log "Checking Python environment..."
[[ -x "$PYTHON" ]] || fail "Python virtual environment not found."

log "Checking project directories..."
for directory in data models results logs backups scripts tests; do
    check_directory "$PROJECT_ROOT/$directory"
done

log "Checking sample dataset..."
check_file "$PROJECT_ROOT/data/sample/classification.csv"

log "Checking processed dataset..."
check_file "$PROJECT_ROOT/data/processed/final_clean.csv"

log "Checking trained model..."
check_file "$PROJECT_ROOT/models/final_model.json"

log "Checking evaluation results..."
RESULT_COUNT="$(find "$PROJECT_ROOT/results" -maxdepth 1 -type f | wc -l)"
[[ "$RESULT_COUNT" -gt 0 ]] || fail "No evaluation results found."

log "Checking backup files..."
BACKUP_COUNT="$(find "$PROJECT_ROOT/backups" -maxdepth 1 -type f -name '*.tar.gz' | wc -l)"
[[ "$BACKUP_COUNT" -gt 0 ]] || fail "No backup archive found."

log "Checking Python package..."
"$PYTHON" -m ai_pipeline.cli.main --help >/dev/null

log "Checking source compilation..."
"$PYTHON" -m compileall -q "$PROJECT_ROOT/src"

log "Health check completed successfully."
