"""Checksum helpers for model artifacts."""

import hashlib
import logging
from pathlib import Path


log = logging.getLogger(__name__)


def compute_sha256(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA256 digest for a file."""
    path = Path(file_path)
    assert path.exists(), f"file does not exist: {path}"
    assert path.is_file(), f"path is not a file: {path}"
    log.debug(f"computing sha256 for\n    {path}")

    # Stream file bytes to avoid loading large model files into memory.
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        chunk = stream.read(chunk_size)
        while chunk:
            hasher.update(chunk)
            chunk = stream.read(chunk_size)
    return hasher.hexdigest()


def verify_sha256(file_path: str | Path, expected_sha256: str) -> bool:
    """Return True when a file digest matches the expected SHA256."""
    assert expected_sha256, "expected_sha256 cannot be empty"
    actual_sha256 = compute_sha256(file_path)
    is_match = actual_sha256.lower() == expected_sha256.strip().lower()
    log.debug(f"sha256 verification result for\n    {file_path}\n    match={is_match}")
    return is_match


def assert_sha256(file_path: str | Path, expected_sha256: str) -> None:
    """Raise ValueError when the file digest mismatches the expected SHA256."""
    assert expected_sha256, "expected_sha256 cannot be empty"
    actual_sha256 = compute_sha256(file_path)
    if actual_sha256.lower() != expected_sha256.strip().lower():
        raise ValueError(
            f"checksum mismatch for {file_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    log.debug(f"sha256 assertion passed for\n    {file_path}")
