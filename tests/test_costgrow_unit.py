"""Unit tests for CostGrow helper behavior."""

import pytest
np = pytest.importorskip("numpy")

from floodsr.models.CostGrow_Terrain import _compute_cost_surface, _fill_nearest_unmasked, _filter_isolated, ModelWorker


pytestmark = pytest.mark.fast


def test_costgrow_cost_surface_penalizes_below_ground_cells():
    """Ensure the terrain penalty cost is 1 over wet cells and higher below ground."""
    wse = np.array([[5.0, 5.0], [2.0, 1.0]], dtype=np.float32)
    dem = np.array([[4.0, 5.5], [1.0, 3.0]], dtype=np.float32)
    valid = np.array([[True, True], [True, False]])
    cost, delta = _compute_cost_surface(wse, dem, valid)
    assert np.isclose(cost[0, 0], 1.0)
    assert np.isclose(cost[0, 1], 1.5)
    assert np.isclose(cost[1, 0], 1.0)
    assert np.isnan(cost[1, 1])
    assert np.isclose(delta[0, 1], -0.5)


def test_costgrow_filter_isolated_keeps_only_anchor_connected_region():
    """Ensure isolated grown regions are removed when disconnected from anchors."""
    source = np.array(
        [
            [True, True, False, False],
            [False, True, False, True],
            [False, False, False, True],
        ],
        dtype=bool,
    )
    anchor = np.array(
        [
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    filtered = _filter_isolated(source, anchor)
    assert filtered[0, 0]
    assert filtered[0, 1]
    assert filtered[1, 1]
    assert not filtered[1, 3]
    assert not filtered[2, 3]


def test_costgrow_fill_nearest_unmasked_returns_copy_when_array_has_no_mask():
    """Ensure fully wet coarse WSE inputs bypass the nearest-fill transform cleanly."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    masked = np.ma.MaskedArray(arr, mask=np.zeros_like(arr, dtype=bool))
    filled = _fill_nearest_unmasked(masked)
    assert np.array_equal(filled, arr)
    assert filled is not arr


def test_costgrow_worker_is_builtin_and_valid_without_artifact():
    """Ensure the built-in CostGrow worker does not require a model artifact."""
    worker = ModelWorker(model_fp=None)
    assert worker.requires_model_artifact is False
    assert worker.is_valid(None) is True


def test_costgrow_worker_resolves_windowed_path_from_fine_grid_size():
    """Ensure large prepared scenes switch CostGrow into the disk-backed execution path."""
    worker = ModelWorker(model_fp=None)
    assert worker._resolve_execution_path("hard", (2048, 2048)) == "simple"
    assert worker._resolve_execution_path("hard", (4096, 4096)) == "windowed"
    assert worker._resolve_execution_path("feather", (4096, 4096)) == "windowed"
