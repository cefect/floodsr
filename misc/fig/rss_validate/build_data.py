"""Build raster and ancillary inputs for RSS validation figure notebooks."""

import argparse, json, logging, shutil, time
from pathlib import Path

from floodsr.model_registry import fetch_model
from floodsr.tohr import tohr
from rss_validate import CASE_D, parse_report_metrics


def main_build_data(project_root: Path, out_dir: Path, force: bool = False, rebuild_model: bool = False) -> dict:
    """Build or copy all prepared data for the configured RSS validation cases."""
    start = time.perf_counter()
    log = logging.getLogger("build_data")
    path_d = {
        "project_root": project_root.expanduser().resolve(),
        "out_dir": out_dir.expanduser().resolve(),
    }
    path_d["data_dir"] = path_d["out_dir"] / "data"
    path_d["data_dir"].mkdir(parents=True, exist_ok=True)
    assert path_d["project_root"].exists(), f"missing project root: {path_d['project_root']}"

    case_manifest_d = {}
    for case_name, case_cfg in CASE_D.items():
        case_manifest_d[case_name] = _1_build_case_data(case_name, case_cfg, path_d, force=force, rebuild_model=rebuild_model, logger=log)

    manifest_d = {
        "cases": case_manifest_d,
        "runtime_s": round(time.perf_counter() - start, 3),
    }
    (path_d["data_dir"] / "manifest.json").write_text(json.dumps(manifest_d, indent=2, sort_keys=True) + "\n")
    log.info(f"wrote root manifest in {manifest_d['runtime_s']:,.2f}s\n    {path_d['data_dir'] / 'manifest.json'}")
    return manifest_d


def _1_build_case_data(case_name: str, case_cfg: dict, path_d: dict, force: bool = False, rebuild_model: bool = False, logger=None) -> dict:
    """Build or copy prepared rasters and references for one validation case."""
    log = logger or logging.getLogger(__name__)
    case_path_d = {
        "case_dir": path_d["project_root"] / "tests" / "data" / case_cfg["case_dir_name"],
        "report_dir": path_d["project_root"] / "report" / "examples_compare" / case_cfg["case_dir_name"],
        "build_dir": path_d["project_root"] / "build" / "examples_compare" / case_cfg["case_dir_name"],
        "case_data_dir": path_d["data_dir"] / case_name,
    }
    case_path_d["rasters_dir"] = case_path_d["case_data_dir"] / "rasters"
    case_path_d["report_reference_dir"] = case_path_d["case_data_dir"] / "report_reference"
    assert case_path_d["case_dir"].is_dir(), f"missing case dir: {case_path_d['case_dir']}"
    assert case_path_d["report_dir"].is_dir(), f"missing report dir: {case_path_d['report_dir']}"
    case_path_d["rasters_dir"].mkdir(parents=True, exist_ok=True)
    case_path_d["report_reference_dir"].mkdir(parents=True, exist_ok=True)
    log.info(f"building {case_name} data under\n    {case_path_d['case_data_dir']}")

    case_spec_fp = case_path_d["case_dir"] / "case_spec.json"
    case_spec = json.loads(case_spec_fp.read_text())
    shutil.copy2(case_spec_fp, case_path_d["case_data_dir"] / "case_spec.json")
    report_md_fp = case_path_d["report_dir"] / f"examples_compare.{case_name}.md"
    shutil.copy2(report_md_fp, case_path_d["report_reference_dir"] / report_md_fp.name)
    for src_fp in sorted((case_path_d["report_dir"] / f"examples_compare.{case_name}_files").glob("*.png")):
        shutil.copy2(src_fp, case_path_d["report_reference_dir"] / src_fp.name)

    input_d = case_spec["inputs"]
    source_raster_d = {
        "lowres_depth": case_path_d["case_dir"] / input_d["lowres_fp"],
        "dem_hr": case_path_d["case_dir"] / input_d["dem_fp"],
        "truth_hr": case_path_d["case_dir"] / input_d["truth_fp"],
    }
    raster_d = {
        "lowres_depth": case_path_d["rasters_dir"] / "lowres_depth.tif",
        "dem_hr": case_path_d["rasters_dir"] / "dem_hr.tif",
        "truth_hr": case_path_d["rasters_dir"] / "truth_hr.tif",
    }
    for label, src_fp in source_raster_d.items():
        assert src_fp.exists(), f"missing source raster for {case_name}/{label}: {src_fp}"
        if force or not raster_d[label].exists():
            shutil.copy2(src_fp, raster_d[label])

    resunet_params = case_spec["expected"]["ResUNet_16x_DEM_default"]["params"]
    resunet_model_version = resunet_params["model_version"]
    raster_d["resunet_prediction"] = case_path_d["rasters_dir"] / "resunet_prediction.tif"
    built_resunet_fp = case_path_d["build_dir"] / f"{Path(input_d['lowres_fp']).stem}_{resunet_model_version}.tif"
    if built_resunet_fp.exists() and not rebuild_model:
        if force or not raster_d["resunet_prediction"].exists():
            shutil.copy2(built_resunet_fp, raster_d["resunet_prediction"])
    elif force or not raster_d["resunet_prediction"].exists():
        resunet_model_fp = fetch_model(resunet_model_version, show_progress=False)
        log.info(f"running {case_name}/{resunet_model_version} to\n    {raster_d['resunet_prediction']}")
        tohr(
            model_version=resunet_model_version,
            model_fp=resunet_model_fp,
            depth_lr_fp=source_raster_d["lowres_depth"],
            dem_hr_fp=source_raster_d["dem_hr"],
            output_fp=raster_d["resunet_prediction"],
            min_depth_threshold=resunet_params.get("min_depth_threshold"),
            show_progress=False,
        )
    assert raster_d["resunet_prediction"].exists(), f"missing ResUNet prediction: {raster_d['resunet_prediction']}"

    costgrow_src_fp = case_path_d["build_dir"] / f"{Path(input_d['lowres_fp']).stem}_CostGrow_Terrain.tif"
    costgrow_available = costgrow_src_fp.exists()
    if costgrow_available:
        raster_d["costgrow_prediction"] = case_path_d["rasters_dir"] / "costgrow_prediction.tif"
        if force or not raster_d["costgrow_prediction"].exists():
            shutil.copy2(costgrow_src_fp, raster_d["costgrow_prediction"])

    report_metrics_d = parse_report_metrics(report_md_fp)
    manifest_d = {
        "case_name": case_name,
        "params": {
            "costgrow_available": costgrow_available,
            "dry_depth_thresh_m": 1e-3,
            "max_depth_m": 5.0,
            "report_vmax_m": case_cfg["report_vmax_m"],
            "resunet": resunet_params,
        },
        "rasters": {key: str(fp.relative_to(case_path_d["case_data_dir"])) for key, fp in raster_d.items()},
        "report_metrics": report_metrics_d,
        "report_reference": sorted(str(fp.relative_to(case_path_d["case_data_dir"])) for fp in case_path_d["report_reference_dir"].glob("*")),
        "source_case_dir": str(case_path_d["case_dir"]),
        "source_report_dir": str(case_path_d["report_dir"]),
    }
    (case_path_d["case_data_dir"] / "manifest.json").write_text(json.dumps(manifest_d, indent=2, sort_keys=True) + "\n")
    log.debug(f"wrote {case_name} manifest")
    return manifest_d


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for building validation figure data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--force", action="store_true", help="recopy prepared data even when targets exist")
    parser.add_argument("--rebuild-model", action="store_true", help="rerun ResUNet instead of copying existing build rasters")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s:%(name)s:%(message)s")
    main_build_data(project_root=args.project_root, out_dir=args.out_dir, force=args.force, rebuild_model=args.rebuild_model)
