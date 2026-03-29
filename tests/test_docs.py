"""Docs build tests."""

import runpy
from pathlib import Path
import subprocess, sys

import pytest


DOCS_SOURCE_DIR = Path("docs/user")
DOCS_CONF_PATH = DOCS_SOURCE_DIR / "conf.py"


def _run_sphinx_build(builder: str, source_dir: Path, build_dir: Path, *extra_args: str):
    """Run one Sphinx build and return the completed subprocess result."""
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        builder,
        "-q",
        "-W",
        *extra_args,
        str(source_dir),
        str(build_dir),
    ]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


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
    result = _run_sphinx_build(
        "linkcheck",
        DOCS_SOURCE_DIR,
        build_dir,
        "-d",
        str(doctree_dir),
        "-D",
        "linkcheck_anchors=False",
    )
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


@pytest.mark.sphinx
@pytest.mark.local
@pytest.mark.parametrize(
    ("raw_language", "expected_title"),
    [
        pytest.param("en", "floodsr documentation", id="html_build_en"),
        pytest.param("fr", "Documentation de floodsr", id="html_build_fr"),
    ],
)
def test_docs_html_builds_for_supported_languages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_language: str,
    expected_title: str,
):
    """HTML docs should build for each supported language and render localized metadata."""
    pytest.importorskip("sphinx", reason="Sphinx not detected in environment.")
    monkeypatch.setenv("READTHEDOCS_LANGUAGE", raw_language)

    doctree_dir = tmp_path / f"doctrees_{raw_language}"
    build_dir = tmp_path / f"html_{raw_language}"
    result = _run_sphinx_build("html", DOCS_SOURCE_DIR, build_dir, "-d", str(doctree_dir))
    index_fp = build_dir / "index.html"
    index_html = index_fp.read_text(encoding="utf-8") if index_fp.exists() else ""

    assert result.returncode == 0, (
        f"Sphinx HTML build failed for language={raw_language}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert index_fp.exists(), f"missing built docs index for language={raw_language}:\n    {index_fp}"
    assert f'<html lang="{raw_language}"' in index_html
    assert expected_title in index_html
