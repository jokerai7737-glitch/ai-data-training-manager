from datetime import datetime
from pathlib import Path
import shutil


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_backup(
    source: str | Path,
    backup_directory: str | Path,
) -> Path:
    source_path = Path(source)
    backup_dir = ensure_directory(backup_directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    destination = (
        backup_dir
        / f"{source_path.stem}_{timestamp}{source_path.suffix}"
    )

    shutil.copy2(source_path, destination)
    return destination
