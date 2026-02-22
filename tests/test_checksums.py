"""Tests for checksum utilities."""

from pathlib import Path

import pytest

from floodsr.checksums import compute_sha256, verify_sha256


pytestmark = pytest.mark.unit


@pytest.fixture(scope="function")
def payload_fp(tmp_path: Path) -> Path:
    """Create a deterministic payload for checksum tests."""
    fp = tmp_path / "payload.bin"
    fp.write_bytes(b"floodsr-test")
    return fp


def test_compute_sha256_returns_hex_digest(payload_fp: Path):
    """Verify digest format for a known payload."""
    digest = compute_sha256(payload_fp)
    assert isinstance(digest, str)
    assert len(digest) == 64


@pytest.mark.parametrize(
    "expected_sha256, expected_match",
    [
        pytest.param(
            "512da56b08416191c606c13875732aa98ea9a2ab4c0f78264a34e22d43b7d170",
            True,
            id="matching_digest",
        ),
        pytest.param("0" * 64, False, id="mismatching_digest"),
    ],
)
def test_verify_sha256_returns_expected_flag(
    payload_fp: Path, expected_sha256: str, expected_match: bool
):
    """Check checksum verification true/false behavior."""
    result = verify_sha256(payload_fp, expected_sha256)
    assert isinstance(result, bool)
    assert result == expected_match
