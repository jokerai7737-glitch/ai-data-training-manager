#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
LOG_DIR="$PROJECT_ROOT/logs"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
PIPELINE_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

log() {
    printf '[PIPELINE] %s\n' "$1"
}

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    exit 1
}

on_exit() {
    status=$?

    if [[ "$status" -eq 0 ]]; then
        printf '[PIPELINE] Pipeline finished successfully.\n' | tee -a "$PIPELINE_LOG"
    else
        printf '[ERROR] Pipeline failed with exit code %s.\n' "$status" | tee -a "$PIPELINE_LOG"
    fi
}

trap on_exit EXIT

cd "$PROJECT_ROOT"

log "Project root: $PROJECT_ROOT"
log "Pipeline log: $PIPELINE_LOG"

{
    echo "========================================"
    echo "AI DATA TRAINING MANAGER PIPELINE"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
} | tee -a "$PIPELINE_LOG"

for script in \
    validate.sh \
    preprocess.sh \
    train.sh \
    evaluate.sh \
    backup.sh \
    health_check.sh
do
    [[ -x "$SCRIPTS_DIR/$script" ]] || fail "Required script is missing or not executable: $script"
done

log "Checking available disk space..."
df -h "$PROJECT_ROOT" | tee -a "$PIPELINE_LOG"

log "Stage 1/6 - Dataset validation"
"$SCRIPTS_DIR/validate.sh" | tee -a "$PIPELINE_LOG"

log "Stage 2/6 - Data preprocessing"
"$SCRIPTS_DIR/preprocess.sh" | tee -a "$PIPELINE_LOG"

log "Stage 3/6 - Model training"
"$SCRIPTS_DIR/train.sh" | tee -a "$PIPELINE_LOG"

log "Stage 4/6 - Model evaluation"
"$SCRIPTS_DIR/evaluate.sh" | tee -a "$PIPELINE_LOG"

log "Stage 5/6 - Backup"
"$SCRIPTS_DIR/backup.sh" | tee -a "$PIPELINE_LOG"

log "Stage 6/6 - Final health check"
"$SCRIPTS_DIR/health_check.sh" | tee -a "$PIPELINE_LOG"

echo
echo "========================================"
echo "PIPELINE SUMMARY"
echo "========================================"

log "Processed dataset:"
ls -lh "$PROJECT_ROOT/data/processed/final_clean.csv"

log "Model artifact:"
ls -lh "$PROJECT_ROOT/models/final_model.json"

log "Evaluation results:"
find "$PROJECT_ROOT/results" -maxdepth 1 -type f -printf '%f\n' | sort

log "Backup archives:"
find "$PROJECT_ROOT/backups" -maxdepth 1 -type f -name '*.tar.gz' -printf '%f\n' | sort

log "Pipeline log:"
ls -lh "$PIPELINE_LOG"

echo "========================================"
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
