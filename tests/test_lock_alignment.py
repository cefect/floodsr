"""Tests that the conda lock file matches the active pytest environment."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

LOCK_FILE = Path(__file__).parent.parent / "container" / "miniforge" / "conda-env-deploy.lock.yml"


def _conda_available():
    """Return whether conda is available in the current environment."""
    return shutil.which("conda") is not None


def _active_env_name():
    """Return the active conda environment name, if conda reports one."""
    result = subprocess.run(
        ["conda", "info", "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout).get("active_prefix_name")


def test_conda_lock_alignment():
    """Conda must report the active pytest environment matches the lock file exactly."""
    print(f"Checking conda availability for lock alignment test.")
    assert _conda_available(), "conda not available in this environment"
    print(f"Checking lock file:\n    {LOCK_FILE}")
    assert LOCK_FILE.exists(), f"Lock file not found: {LOCK_FILE}"
    assert LOCK_FILE.is_file(), f"Lock file is not a file: {LOCK_FILE}"

    env_name = _active_env_name()
    print(f"Active conda environment reported by conda: {env_name}")
    assert env_name not in (None, "base"), (
        "conda active environment is unavailable or not project-specific"
    )

    # Delegate parsing and comparison to conda so build strings are checked too.
    print("Running `conda compare --json` against the active environment.")
    result = subprocess.run(
        ["conda", "compare", "--json", str(LOCK_FILE)],
        capture_output=True,
        text=True,
    )
    message_l = json.loads(result.stdout or "[]")
    print(f"conda compare return code: {result.returncode}")
    if result.stdout:
        print(f"conda compare stdout:\n{result.stdout}")
    if result.stderr:
        print(f"conda compare stderr:\n{result.stderr}")

    assert result.returncode == 0, (
        f"active conda env '{env_name}' diverges from lock file:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert message_l, "conda compare returned no output"


 
