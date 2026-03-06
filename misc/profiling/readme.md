# HRDEM memory profiling

This folder contains a reproducible memory experiment for the HRDEM fetch paths on the large-grid fixture:

`/workspace/tests/data/fathom_1024/lores.tif`

The experiment runs these three cases in sequence with `use_cache=False`:

1. `non_windowed`
2. `windowed_w512`
3. `windowed_w256`

The experiment writes:

- one output raster per case
- one memory trace CSV per case
- one combined `summary.csv`
- one combined `summary.json`

 

## run

Run the full sequence with the bash entrypoint:

```bash
conda run --no-capture-output -n dev /workspace/misc/profiling/run_profile_hrdem_memory.sh

```

The bash script runs the three cases in order and then writes the combined summary.

## evaluate

The main file to inspect is:

```bash
column -s, -t < /workspace/misc/profiling/output/summary.csv
```

The key fields are:

- `peak_memory_mib`: peak resident memory sampled by `memory_profiler`
- `wall_clock_s`: end-to-end runtime measured around the fetch call
- `fetch_window_size`: fetch tile size used for the case
- `force_tiling`: whether the case was forced onto `_03_read_dem_windowed_tiles_to_vrt`

Inspect one case trace over time:

```bash
head /workspace/misc/profiling/output/windowed_w512_trace.csv
tail /workspace/misc/profiling/output/windowed_w512_trace.csv
```

Quickly sort by peak memory:

```bash
python - <<'PY'
import csv
from pathlib import Path

fp = Path("/workspace/misc/profiling/output/summary.csv")
with fp.open() as f:
    rows = list(csv.DictReader(f))
for row in sorted(rows, key=lambda row: float(row["peak_memory_mib"]), reverse=True):
    print(
        row["case_id"],
        f"peak_memory_mib={float(row['peak_memory_mib']):.2f}",
        f"wall_clock_s={float(row['wall_clock_s']):.2f}",
    )
PY
```

## case definitions

- `non_windowed`
  - `force_tiling=False`
  - `memory_limit_gib=4096.0`
  - intended to hit `_02_read_dem_non_windowed`
- `windowed_w512`
  - `force_tiling=True`
  - `fetch_window_size=512`
  - intended to hit `_03_read_dem_windowed_tiles_to_vrt`
- `windowed_w256`
  - `force_tiling=True`
  - `fetch_window_size=256`
  - intended to hit `_03_read_dem_windowed_tiles_to_vrt`

## notes

- The bash entrypoint runs each case in a fresh `conda run -n dev python` process so memory state does not leak between cases.
- The experiment uses the normal HRDEM fetch code path, including live network access to STAC and project-extent services.
- `memory_profiler` samples process RSS, so values should be treated as approximate but comparable across cases.
- The Python module is intentionally not a CLI; it just exposes fixed-case functions used by the bash entrypoint.
