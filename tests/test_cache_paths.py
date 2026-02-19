"""Tests for cache path helpers."""

from pathlib import Path

import pytest

from floodsr.cache_paths import get_cache_dir


@pytest.mark.parametrize(
    "cache_dir",
    [
        pytest.param("cache_str", id="string_cache_dir"),
        pytest.param(Path("cache_path"), id="path_cache_dir"),
    ],
)
def test_get_cache_dir_returns_created_path(tmp_path: Path, cache_dir: str | Path):
    """Ensure explicit cache directory inputs produce writable paths."""
    cache_arg = str(tmp_path / cache_dir) if isinstance(cache_dir, str) else tmp_path / cache_dir
    result = get_cache_dir(cache_arg)
    assert isinstance(result, Path)
    assert result.exists()

