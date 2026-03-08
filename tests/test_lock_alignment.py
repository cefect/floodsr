"""Tests that the conda lock file matches the live deploy environment."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit

LOCK_FILE = Path(__file__).parent.parent / "container" / "miniforge" / "conda-env-deploy.lock.yml"


def _parse_lock_file():
    """Return (env_name, {package: version}) from the lock file."""
    with LOCK_FILE.open() as fh:
        data = yaml.safe_load(fh)
    env_name = data["name"]
    pinned = {}
    for entry in data.get("dependencies", []):
        if not isinstance(entry, str):
            continue
        parts = entry.split("=")
        if len(parts) >= 2:
            pinned[parts[0]] = parts[1]
    return env_name, pinned


def _conda_available():
    return shutil.which("conda") is not None


def _env_exists(env_name):
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    envs = json.loads(result.stdout).get("envs", [])
    return any(Path(e).name == env_name for e in envs)


@pytest.mark.skipif(
    not _conda_available(),
    reason="conda not available in this environment",
)
def test_conda_lock_alignment():
    """Every package pinned in the lock file must match the live deploy env."""
    env_name, pinned = _parse_lock_file()

    if not _env_exists(env_name):
        pytest.skip(f"conda env '{env_name}' does not exist in this environment")

    result = subprocess.run(
        ["conda", "list", "--name", env_name, "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    installed = {pkg["name"]: pkg["version"] for pkg in json.loads(result.stdout)}

    mismatches = []
    for pkg, locked_version in pinned.items():
        if pkg not in installed:
            mismatches.append(f"  {pkg}: locked={locked_version!r} MISSING from env")
        elif installed[pkg] != locked_version:
            mismatches.append(
                f"  {pkg}: locked={locked_version!r} actual={installed[pkg]!r}"
            )

    assert not mismatches, (
        f"conda env '{env_name}' diverges from lock file:\n" + "\n".join(mismatches)
    )


def test_lock_file_is_parseable():
    """Lock file exists and contains at least one pinned dependency."""
    assert LOCK_FILE.exists(), f"Lock file not found: {LOCK_FILE}"
    env_name, pinned = _parse_lock_file()
    assert env_name, "Lock file must declare a non-empty env name"
    assert len(pinned) > 0, "Lock file must contain at least one pinned dependency"
