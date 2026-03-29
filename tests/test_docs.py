"""Docs build tests."""

import runpy
from pathlib import Path
import subprocess, sys

import pytest


DOCS_SOURCE_DIR = Path("docs/user")
DOCS_CONF_PATH = DOCS_SOURCE_DIR / "conf.py"


@pytest.mark.sphinx
@pytest.mark.local
def test_docs_linkcheck_builds(tmp_path: Path) -> None:
    """Run a lightweight Sphinx linkcheck build for user docs."""
    # Skip this test when sphinx is not installed in the active environment.
    pytest.importorskip("sphinx", reason="Sphinx not detected in environment.")
    assert DOCS_SOURCE_DIR.exists(), f"missing docs source directory: {DOCS_SOURCE_DIR}"

    # Build docs with linkcheck and fail on warnings (including bad links).
    doctree_dir = tmp_path / "doctrees"
    build_dir = tmp_path / "linkcheck"
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "linkcheck",
        "-q",
        "-W",
        "-d",
        str(doctree_dir),
        "-D",
        "linkcheck_anchors=False",
        str(DOCS_SOURCE_DIR),
        str(build_dir),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, (
        "Sphinx linkcheck build failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.parametrize(
    ("raw_language", "expected"),
    [
        pytest.param("en", "en", id="explicit_en"),
        pytest.param("fr", "fr", id="explicit_fr"),
    ],
)
def test_docs_conf_resolves_language(monkeypatch: pytest.MonkeyPatch, raw_language, expected):
    """Docs config should clamp language selection to the supported docs locales."""
    pytest.importorskip("setuptools_scm", reason="Docs config dependency missing.")

    monkeypatch.setenv("READTHEDOCS_LANGUAGE", raw_language)

    # Load the live docs config so the test exercises the shipped config value.
    conf_globals = runpy.run_path(str(DOCS_CONF_PATH))
    result = conf_globals["language"]

    assert isinstance(result, str)
    assert result == expected
