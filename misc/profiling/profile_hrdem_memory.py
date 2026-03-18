"""Profile one fixed HRDEM fetch case and write memory experiment artifacts."""

import csv, json, logging, sys, time, traceback
from pathlib import Path

from memory_profiler import memory_usage

import floodsr.dem_sources.hrdem_mosaic


DEPTH_LR_FP = Path("/workspace/tests/data/fathom_1024/lores.tif")
OUTPUT_DIR = Path("/workspace/misc/profiling/output")
INTERVAL_S = 0.2
CASE_D = {
    "non_windowed": {
        "force_tiling": False,
        "fetch_window_size": 256,
        "memory_limit_gib": 4096.0,
        "suffix": ".tif",
    },
    "windowed_w512": {
        "force_tiling": True,
        "fetch_window_size": 512,
        "memory_limit_gib": 0.01,
        "suffix": ".vrt",
    },
    "windowed_w256": {
        "force_tiling": True,
        "fetch_window_size": 256,
        "memory_limit_gib": 0.01,
        "suffix": ".vrt",
    },
}


def _setup_logger(
    log_fp: str | Path,
    logger_name: str,
    logger=None,
):
    """Build one INFO-level logger with stream and file handlers."""
    if logger is not None:
        return logger
    log_fp = Path(log_fp).expanduser().resolve()
    log_fp.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_fp)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("[%(levelname)s]%(name)s: %(message)s"))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s]%(name)s: %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main_profile_hrdem_memory(
    case_id: str,
    output_dir: str | Path = OUTPUT_DIR,
    depth_lr_fp: str | Path = DEPTH_LR_FP,
    interval_s: float = INTERVAL_S,
    logger=None,
):
    """
    Profile one fixed HRDEM fetch case and persist the trace and summary rows.

    Parameters
    ----------
    case_id:
        Case key from ``CASE_D``.
    output_dir:
        Directory for the case output raster, trace CSV, and case JSON.
    depth_lr_fp:
        Large-grid low-resolution raster used to drive the fetch.
    interval_s:
        Sampling interval passed to ``memory_profiler.memory_usage``.
    logger:
        Optional logger for progress reporting.

    Returns
    -------
    dict
        Serialized case summary payload.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    depth_lr_fp = Path(depth_lr_fp).expanduser().resolve()
    interval_s = float(interval_s)
    assert case_id in CASE_D, f"case_id must be one of {tuple(CASE_D)}, got {case_id!r}"
    assert depth_lr_fp.exists(), f"depth_lr_fp does not exist: {depth_lr_fp}"
    assert interval_s > 0, f"interval_s must be > 0, got {interval_s}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_fp = output_dir / f"{case_id}.log"
    log = _setup_logger(log_fp, logger_name=f"hrdem_profile.{case_id}", logger=logger)
    log.info("\n" + "=" * 80)
    log.info(f"starting profiling case '{case_id}'")
    log.info(f"depth_lr_fp=\n    {depth_lr_fp}")
    log.info(f"output_dir=\n    {output_dir}")

    # Profile one fetch call in this fresh Python process.
    mem_trace_l, case_result_d = memory_usage(
        (
            _run_fetch_case,
            (),
            {
                "case_id": case_id,
                "case_cfg": CASE_D[case_id],
                "depth_lr_fp": depth_lr_fp,
                "output_dir": output_dir,
                "logger_name": log.name,
            },
        ),
        interval=interval_s,
        timestamps=True,
        retval=True,
        backend="psutil",
    )
    trace_csv_fp = output_dir / f"{case_id}_trace.csv"
    start_ts = float(mem_trace_l[0][1])
    with trace_csv_fp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["elapsed_s", "memory_mib"])
        writer.writeheader()
        for memory_mib, ts in mem_trace_l:
            writer.writerow({"elapsed_s": f"{float(ts) - start_ts:.3f}", "memory_mib": f"{float(memory_mib):.3f}"})

    case_result_d["peak_memory_mib"] = float(max(memory_mib for memory_mib, _ in mem_trace_l))
    case_result_d["trace_csv_fp"] = str(trace_csv_fp)
    case_json_fp = output_dir / f"{case_id}.json"
    case_json_fp.write_text(json.dumps(case_result_d, indent=2), encoding="utf-8")
    log.info(
        f"finished profiling case '{case_id}': wall_clock_s={case_result_d['wall_clock_s']:.2f}, "
        f"peak_memory_mib={case_result_d['peak_memory_mib']:.2f}"
    )
    log.info("=" * 80 + "\n")
    return case_result_d


def main_write_profile_summary(
    output_dir: str | Path = OUTPUT_DIR,
    logger=None,
):
    """Build one combined summary CSV and JSON from per-case profiling outputs."""
    output_dir = Path(output_dir).expanduser().resolve()
    assert output_dir.exists(), f"output_dir does not exist: {output_dir}"
    log = _setup_logger(output_dir / "summary.log", logger_name="hrdem_profile.summary", logger=logger)
    summary_l = []
    for case_id in CASE_D:
        case_json_fp = output_dir / f"{case_id}.json"
        if case_json_fp.exists():
            summary_l.append(json.loads(case_json_fp.read_text(encoding="utf-8")))
    summary_csv_fp = output_dir / "summary.csv"
    summary_json_fp = output_dir / "summary.json"
    if summary_l:
        with summary_csv_fp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case_id",
                    "status",
                    "force_tiling",
                    "fetch_window_size",
                    "memory_limit_gib",
                    "peak_memory_mib",
                    "wall_clock_s",
                    "output_fp",
                    "trace_csv_fp",
                    "log_fp",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_l)
        summary_json_fp.write_text(json.dumps(summary_l, indent=2), encoding="utf-8")
        log.info(f"wrote profiling summary to\n    {summary_csv_fp}")
    return {"summary_csv_fp": str(summary_csv_fp), "summary_json_fp": str(summary_json_fp), "cases": summary_l}


def _run_fetch_case(
    case_id: str,
    case_cfg: dict,
    depth_lr_fp: Path,
    output_dir: Path,
    logger_name: str,
):
    """Execute one fixed HRDEM fetch case and return one serializable payload."""
    assert case_id, "case_id is required"
    assert isinstance(case_cfg, dict), f"case_cfg must be dict, got {type(case_cfg)!r}"
    log = logging.getLogger(logger_name)
    case_output_fp = output_dir / f"{case_id}{case_cfg['suffix']}"
    if case_output_fp.exists():
        case_output_fp.unlink()
    t0 = time.perf_counter()
    try:
        result = floodsr.dem_sources.hrdem_mosaic.main_fetch_hrdem_for_lowres_tile(
            depth_lr_fp=depth_lr_fp,
            output_fp=case_output_fp,
            use_cache=False,
            force_tiling=bool(case_cfg["force_tiling"]),
            fetch_window_size=int(case_cfg["fetch_window_size"]),
            memory_limit_gib=float(case_cfg["memory_limit_gib"]),
            show_progress=True,
            logger=log,
        )
        return {
            "case_id": case_id,
            "status": "ok",
            "force_tiling": bool(case_cfg["force_tiling"]),
            "fetch_window_size": int(case_cfg["fetch_window_size"]),
            "memory_limit_gib": float(case_cfg["memory_limit_gib"]),
            "peak_memory_mib": None,
            "wall_clock_s": float(time.perf_counter() - t0),
            "output_fp": str(result.dem_fp),
            "trace_csv_fp": str(output_dir / f"{case_id}_trace.csv"),
            "log_fp": str(output_dir / f"{case_id}.log"),
            "error": "",
        }
    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "force_tiling": bool(case_cfg["force_tiling"]),
            "fetch_window_size": int(case_cfg["fetch_window_size"]),
            "memory_limit_gib": float(case_cfg["memory_limit_gib"]),
            "peak_memory_mib": None,
            "wall_clock_s": float(time.perf_counter() - t0),
            "output_fp": str(case_output_fp),
            "trace_csv_fp": str(output_dir / f"{case_id}_trace.csv"),
            "log_fp": str(output_dir / f"{case_id}.log"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
