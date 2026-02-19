"""Checksum helpers for model artifacts."""

import hashlib
from pathlib import Path


def compute_sha256(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA256 digest for a file."""
    path = Path(file_path)
    assert path.exists(), f"file does not exist: {path}"
    assert path.is_file(), f"path is not a file: {path}"

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
    return actual_sha256.lower() == expected_sha256.strip().lower()


def assert_sha256(file_path: str | Path, expected_sha256: str) -> None:
    """Raise ValueError when the file digest mismatches the expected SHA256."""
    assert expected_sha256, "expected_sha256 cannot be empty"
    actual_sha256 = compute_sha256(file_path)
    if actual_sha256.lower() != expected_sha256.strip().lower():
        raise ValueError(
            f"checksum mismatch for {file_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

