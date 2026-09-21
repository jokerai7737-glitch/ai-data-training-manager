#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="$PROJECT_ROOT/backups"

log() {
    printf '[BACKUP] %s\n' "$1"
}

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    exit 1
}

cd "$PROJECT_ROOT"

mkdir -p "$BACKUP_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
BACKUP_FILE="$BACKUP_DIR/ai_pipeline_backup_${TIMESTAMP}.tar.gz"

SOURCE_DIRS=(
    "data/processed"
    "models"
    "results"
    "logs"
)

for directory in "${SOURCE_DIRS[@]}"; do
    [[ -d "$directory" ]] || fail "Required directory not found: $directory"
done

log "Project root: $PROJECT_ROOT"
log "Backup file: $BACKUP_FILE"
log "Collecting project artifacts..."

tar -czf "$BACKUP_FILE" "${SOURCE_DIRS[@]}"

[[ -s "$BACKUP_FILE" ]] || fail "Backup file was not created correctly."

log "Backup completed successfully."
log "Backup size: $(du -h "$BACKUP_FILE" | cut -f1)"
log "Backup path: $BACKUP_FILE"
