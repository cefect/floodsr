"""Tests for model registry and fetch workflow."""

import hashlib, json, logging
from pathlib import Path

import pytest

from floodsr.model_registry import get_retrieval_backend, list_models


@pytest.fixture(scope="function")
def models_manifest_fp(tmp_path: Path) -> Path:
    """Create a local manifest that points to a local model file."""
    source_fp = tmp_path / "source_model.onnx"
    source_fp.write_bytes(b"test-model")
    sha256 = hashlib.sha256(source_fp.read_bytes()).hexdigest()

    # Build a one-model manifest that exercises the file backend.
    manifest = {
        "models": {
            "v-test": {
                "file_name": "model.onnx",
                "url": source_fp.as_uri(),
                "sha256": sha256,
                "description": "Local test model.",
            }
        }
    }
    manifest_fp = tmp_path / "models.json"
    manifest_fp.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_fp


def test_list_models_returns_non_empty_records(models_manifest_fp: Path):
    """Ensure model listing returns records from manifest."""
    records = list_models(manifest_fp=models_manifest_fp)
    assert isinstance(records, list)
    assert len(records) > 0


def test_fetch_model_returns_cached_path(tmp_path: Path, models_manifest_fp: Path):
    """Ensure model fetch stores the artifact in cache."""
    model_fp = fetch_model(
        "v-test",
        cache_dir=tmp_path / "cache",
        manifest_fp=models_manifest_fp,
    )
    assert isinstance(model_fp, Path)
    assert model_fp.exists()


def test_fetch_model_fails_on_checksum_mismatch(tmp_path: Path):
    """Ensure fetch fails when downloaded bytes do not match manifest digest."""
    source_fp = tmp_path / "source_model.onnx"
    source_fp.write_bytes(b"bad-hash-model")
    manifest = {
        "models": {
            "v-bad": {
                "file_name": "model.onnx",
                "url": source_fp.as_uri(),
                "sha256": "0" * 64,
                "description": "Mismatched hash model.",
            }
        }
    }
    manifest_fp = tmp_path / "models_bad.json"
    manifest_fp.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        fetch_model(
            "v-bad",
            cache_dir=tmp_path / "cache",
            manifest_fp=manifest_fp,
        )
    assert isinstance(exc_info.value, ValueError)
    assert "checksum mismatch" in str(exc_info.value)


def test_default_manifest_http_links_resolve(tmp_path: Path):
    """Ensure HTTP model URLs in the default manifest resolve with project fetch logic."""
    log = logging.getLogger(__name__)

    # Collect HTTP(S) model URLs from the packaged manifest.
    http_records = [record for record in list_models() if record.url.startswith(("http://", "https://"))]
    if not http_records:
        pytest.skip("no HTTP URLs configured in manifest")
    log.info(f"validating {len(http_records):,} HTTP model URL(s)")

    for record in http_records:
        backend = get_retrieval_backend(record.url)
        assert backend.name == "http"
        log.info(f"fetching model version '{record.version}' from\n    {record.url}")
        destination = tmp_path / record.version / f"{record.file_name}.part"
        try:
            model_fp = backend.retrieve(record.url, destination)
        except RuntimeError as exc:
            reason = str(exc).lower()
            if "temporary failure in name resolution" in reason or "name or service not known" in reason:
                pytest.skip(f"network unavailable for HTTP link validation: {exc}")
            raise

        assert isinstance(model_fp, Path)
        assert model_fp.exists()
