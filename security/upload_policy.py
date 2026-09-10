"""Upload validation policy for research datasets and patient files."""

from __future__ import annotations

from pathlib import Path


ALLOWED_DATA_SUFFIXES = {".csv"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def validate_upload_name(name: str) -> None:
    suffix = Path(str(name)).suffix.lower()
    if suffix not in ALLOWED_DATA_SUFFIXES:
        raise ValueError("Only CSV uploads are supported in the current research MVP.")


def validate_upload_size(size_bytes: int) -> None:
    if int(size_bytes) <= 0:
        raise ValueError("Uploaded file is empty.")
    if int(size_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file exceeds the 25 MB research-MVP limit.")
