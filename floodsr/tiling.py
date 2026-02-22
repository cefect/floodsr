"""Shared tiling helpers for model-worker windowing and mosaicing."""

import numpy as np
from tqdm import tqdm


def build_tile_starts(total_size: int, tile_size: int, stride: int) -> list[int]:
    """Build overlap-aware tile starts with guaranteed trailing-edge coverage."""
    assert total_size > 0, f"total_size must be > 0; got {total_size}"
    assert tile_size > 0, f"tile_size must be > 0; got {tile_size}"
    assert stride > 0, f"stride must be > 0; got {stride}"
    starts = list(range(0, max(total_size - tile_size + 1, 1), stride))
    last_start = total_size - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iter_window_origins(
    y_starts: list[int],
    x_starts: list[int],
    *,
    use_progress: bool,
    desc: str = "windowed inference",
):
    """Yield indexed window origins with optional progress rendering."""
    total = len(y_starts) * len(x_starts)
    windows = ((yi, xi, y0, x0) for yi, y0 in enumerate(y_starts) for xi, x0 in enumerate(x_starts))
    if use_progress:
        return tqdm(windows, desc=desc, total=total, unit="window")
    return windows


def build_feather_ramp(tile_size: int, overlap: int) -> np.ndarray:
    """Build one-dimensional symmetric feather weights for tile blending."""
    assert tile_size > 0, f"tile_size must be > 0; got {tile_size}"
    assert overlap >= 0, f"overlap must be >= 0; got {overlap}"
    assert overlap < tile_size, f"overlap must be < tile_size; got overlap={overlap}, tile_size={tile_size}"

    feather_1d = np.ones(tile_size, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
        feather_1d[:overlap] = ramp
        feather_1d[-overlap:] = ramp[::-1]
    return np.clip(feather_1d, 1e-3, 1.0)
