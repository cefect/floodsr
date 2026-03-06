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

# RESULTS

## 2026-03-06 run

| case_id | status | force_tiling | fetch_window_size | memory_limit_gib | peak_memory_mib | wall_clock_s |
| --- | --- | --- | --- | --- | --- | --- |
| `non_windowed` | `ok` | `FALSE` | `256` | `4096` | `10201.69531` | `206.28154708398506` |
| `windowed_w512` | `ok` | `TRUE` | `512` | `0.01` | `5913.132813` | `211.6905551289965` |
| `windowed_w256` | `ok` | `TRUE` | `256` | `0.01` | `1752.421875` | `260.3550035379885` |
