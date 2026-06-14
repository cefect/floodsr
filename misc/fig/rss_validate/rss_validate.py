"""Shared helpers for RSS validation figure exports."""

import json, re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

CASE_D = {
    "rss_dudelange_A": {
        "case_dir_name": "rss_dudelange_A",
        "report_vmax_m": 100.3365,
    },
    "rss_mersch_A": {
        "case_dir_name": "rss_mersch_A",
        "report_vmax_m": 12.7052,
    },
}


def parse_report_metrics(report_md_fp: Path) -> dict[str, dict[str, float]]:
    """Extract the rendered metrics table from an examples_compare markdown report."""
    text = report_md_fp.read_text()
    table_start = text.find("<table")
    table_end = text.find("</table>", table_start)
    assert table_start >= 0 and table_end >= 0, f"missing metrics table in {report_md_fp}"
    table_text = text[table_start:table_end]
    header_text = table_text[table_text.find("<thead>"):table_text.find("</thead>")]
    body_text = table_text[table_text.find("<tbody>"):table_text.find("</tbody>")]
    columns = [col.strip() for col in re.findall(r"<th>(.*?)</th>", header_text, flags=re.S)][1:]
    out_d = {col: {} for col in columns}
    for row_text in re.findall(r"<tr>(.*?)</tr>", body_text, flags=re.S):
        row_heads = re.findall(r"<th>(.*?)</th>", row_text, flags=re.S)
        if not row_heads:
            continue
        metric_name = row_heads[0].strip()
        values = re.findall(r"<td>(.*?)</td>", row_text, flags=re.S)
        for col, value in zip(columns, values):
            out_d[col][metric_name] = float(value)
    return out_d


def load_case_data(data_dir: Path, case_name: str) -> dict:
    """Load one prepared case directory into a plotting dictionary."""
    case_data_dir = data_dir / case_name
    manifest_fp = case_data_dir / "manifest.json"
    assert manifest_fp.exists(), f"missing manifest: {manifest_fp}"
    manifest = json.loads(manifest_fp.read_text())
    raster_fp_d = {key: case_data_dir / fp for key, fp in manifest["rasters"].items()}

    depth_lr_raw, depth_lr_profile = read_raster(raster_fp_d["lowres_depth"])
    dem_hr_raw, dem_hr_profile = read_raster(raster_fp_d["dem_hr"])
    truth_raw, truth_profile = read_raster(raster_fp_d["truth_hr"])
    resunet_pred, pred_profile = read_raster(raster_fp_d["resunet_prediction"])
    truth_aligned = reproject_to_profile(raster_fp_d["truth_hr"], pred_profile, Resampling.nearest)
    lr_nearest = reproject_to_profile(raster_fp_d["lowres_depth"], pred_profile, Resampling.nearest)
    lr_bilinear = reproject_to_profile(raster_fp_d["lowres_depth"], pred_profile, Resampling.bilinear)
    costgrow_pred = None
    if "costgrow_prediction" in raster_fp_d:
        costgrow_pred = reproject_to_profile(raster_fp_d["costgrow_prediction"], pred_profile, Resampling.nearest)

    return {
        "case_name": case_name,
        "manifest": manifest,
        "raster_fp_d": raster_fp_d,
        "raw": {
            "lowres_depth": (depth_lr_raw, depth_lr_profile),
            "truth_hr": (truth_raw, truth_profile),
            "dem_hr": (dem_hr_raw, dem_hr_profile),
        },
        "aligned": {
            "truth_hr": truth_aligned,
            "lowres_nearest": lr_nearest,
            "lowres_bilinear": lr_bilinear,
            "resunet_prediction": resunet_pred,
            "costgrow_prediction": costgrow_pred,
            "profile": pred_profile,
        },
        "report_metrics": manifest["report_metrics"],
    }


def read_raster(fp: Path) -> tuple[np.ndarray, dict]:
    """Read one raster band and replace nodata with zero."""
    with rasterio.open(fp) as ds:
        arr = ds.read(1).astype(np.float32, copy=False)
        profile = ds.profile.copy()
        nodata = ds.nodata
    if nodata is not None:
        arr = np.where(np.isclose(arr, nodata), 0.0, arr).astype(np.float32, copy=False)
    return arr, profile


def reproject_to_profile(src_fp: Path, dst_profile: dict, resampling: Resampling) -> np.ndarray:
    """Reproject one raster to the grid described by a destination profile."""
    with rasterio.open(src_fp) as src_ds:
        out = np.zeros((int(dst_profile["height"]), int(dst_profile["width"])), dtype=np.float32)
        reproject(
            source=rasterio.band(src_ds, 1),
            destination=out,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            src_nodata=src_ds.nodata,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            dst_nodata=0.0,
            resampling=resampling,
        )
    return out.astype(np.float32, copy=False)


def compute_metrics(reference_depth_m: np.ndarray, estimate_depth_m: np.ndarray, max_depth: float, dry_depth_thresh_m: float) -> dict[str, float]:
    """Compute the report-style depth metrics for one aligned estimate."""
    reference_np = reference_depth_m.astype(np.float32, copy=False)
    estimate_np = estimate_depth_m.astype(np.float32, copy=False)
    diff = estimate_np - reference_np
    wet_mask = reference_np >= dry_depth_thresh_m
    mse_all = float(np.mean(np.square(diff), dtype=np.float64))
    rmse_all = float(np.sqrt(mse_all))
    mae_all = float(np.mean(np.abs(diff), dtype=np.float64))
    bias_all = float(np.mean(diff, dtype=np.float64))
    rmse_wet = float(np.sqrt(np.mean(np.square(diff[wet_mask]), dtype=np.float64))) if wet_mask.any() else np.nan
    psnr = np.inf if mse_all <= 0.0 else float(20.0 * np.log10(max_depth) - 10.0 * np.log10(mse_all))
    ref64 = reference_np.astype(np.float64, copy=False)
    est64 = estimate_np.astype(np.float64, copy=False)
    mu_x = float(ref64.mean())
    mu_y = float(est64.mean())
    sigma_x = float(ref64.var())
    sigma_y = float(est64.var())
    sigma_xy = float(((ref64 - mu_x) * (est64 - mu_y)).mean())
    c1 = float((0.01 * max_depth) ** 2)
    c2 = float((0.03 * max_depth) ** 2)
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return {"psnr": psnr, "ssim": float(ssim), "mae_m": mae_all, "mase_m": mae_all, "rmse_m": rmse_all, "rmse_wet_m": rmse_wet, "bias_m": bias_all}


def plot_input_diagnostics(case_d: dict) -> plt.Figure:
    """Create the report-style raw input diagnostic figure."""
    dry_depth_thresh_m = float(case_d["manifest"]["params"]["dry_depth_thresh_m"])
    plot_specs_raw = [
        ("Low-res depth", *case_d["raw"]["lowres_depth"], "viridis", True, dry_depth_thresh_m),
        ("Hires truth depth", *case_d["raw"]["truth_hr"], "viridis", True, dry_depth_thresh_m),
        ("High-res DEM", *case_d["raw"]["dem_hr"], "terrain", False, None),
    ]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
    for row_idx, (title, arr, profile, cmap, use_dry_mask, dry_thresh) in enumerate(plot_specs_raw):
        vals = arr[np.isfinite(arr)]
        ax_hist = axes[row_idx, 0]
        ax_raster = axes[row_idx, 1]
        ax_hist.hist(vals, bins=60, color="steelblue", alpha=0.9)
        if use_dry_mask:
            ax_hist.axvline(dry_thresh, color="red", linestyle="--", linewidth=1.5)
        ax_hist.set_title(f"{title} histogram")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(color="lightgrey", linestyle="-", linewidth=0.7)
        text_lines = [
            f"shape: {arr.shape}",
            f"res(x,y): {fmt_res(profile)}",
            f"min: {vals.min():.3f}",
            f"max: {vals.max():.3f}",
            f"mean: {vals.mean():.3f}",
            f"std: {vals.std():.3f}",
        ]
        ax_hist.text(0.98, 0.95, chr(10).join(text_lines), transform=ax_hist.transAxes, fontsize=9, va="top", ha="right")
        raster_arr = np.ma.masked_where(arr < dry_thresh, arr) if use_dry_mask else arr
        im = ax_raster.imshow(raster_arr, cmap=cmap)
        ax_raster.set_title(f"{title} raster")
        ax_raster.set_axis_off()
        fig.colorbar(im, ax=ax_raster, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_comparison(case_d: dict) -> tuple[plt.Figure, pd.DataFrame]:
    """Create the report-style final comparison figure and metrics table."""
    params = case_d["manifest"]["params"]
    dry_depth_thresh_m = float(params["dry_depth_thresh_m"])
    max_depth = float(params["max_depth_m"])
    aligned_d = case_d["aligned"]
    metrics_d = {
        "Low-res depth (nearest)": compute_metrics(aligned_d["truth_hr"], aligned_d["lowres_nearest"], max_depth, dry_depth_thresh_m),
        "Low-res depth (bilinear)": compute_metrics(aligned_d["truth_hr"], aligned_d["lowres_bilinear"], max_depth, dry_depth_thresh_m),
        "ResUNet_16x_DEM prediction": compute_metrics(aligned_d["truth_hr"], aligned_d["resunet_prediction"], max_depth, dry_depth_thresh_m),
    }
    report_metric_d = case_d["report_metrics"]
    if "Bilinear" in report_metric_d:
        metrics_d["Low-res depth (bilinear)"].update(report_metric_d["Bilinear"])
    if "ResUNet_16x_DEM" in report_metric_d:
        metrics_d["ResUNet_16x_DEM prediction"].update(report_metric_d["ResUNet_16x_DEM"])
    rows = [
        ("Hires truth", aligned_d["truth_hr"], None),
        ("Low-res depth (nearest)", aligned_d["lowres_nearest"], "Low-res depth (nearest)"),
        ("Low-res depth (bilinear)", aligned_d["lowres_bilinear"], "Low-res depth (bilinear)"),
        ("ResUNet_16x_DEM prediction", aligned_d["resunet_prediction"], "ResUNet_16x_DEM prediction"),
    ]
    if aligned_d["costgrow_prediction"] is not None:
        metrics_d["CostGrow_Terrain prediction"] = compute_metrics(aligned_d["truth_hr"], aligned_d["costgrow_prediction"], max_depth, dry_depth_thresh_m)
        if "CostGrow_Terrain" in report_metric_d:
            metrics_d["CostGrow_Terrain prediction"].update(report_metric_d["CostGrow_Terrain"])
        rows.append(("CostGrow_Terrain prediction", aligned_d["costgrow_prediction"], "CostGrow_Terrain prediction"))

    vmax_depth = float(params["report_vmax_m"])
    fig, axes = plt.subplots(nrows=len(rows), ncols=2, figsize=(11, 16), sharex="col")
    for row_idx, (title, img, metric_key) in enumerate(rows):
        ax_hist = axes[row_idx, 0]
        ax_img = axes[row_idx, 1]
        masked = np.ma.masked_where(img < dry_depth_thresh_m, img)
        vals = masked.compressed()
        ax_hist.hist(vals, bins=50, density=True, color="steelblue", alpha=0.9)
        ax_hist.axvline(dry_depth_thresh_m, color="red", linestyle="--", linewidth=1.2)
        ax_hist.set_ylabel("Density")
        ax_hist.set_xlim(0.0, vmax_depth)
        ax_hist.grid(color="lightgrey", linestyle="-", linewidth=0.7)
        if row_idx == len(rows) - 1:
            ax_hist.set_xlabel("Depth (m)")
        info_lines = [stats_text(vals)]
        if metric_key is not None:
            metric_group = metrics_d[metric_key]
            info_lines.extend([
                f"PSNR={fmt_metric(metric_group['psnr'], 2)} dB",
                f"SSIM={fmt_metric(metric_group['ssim'])}",
                f"MAE={fmt_metric(metric_group['mae_m'])} m",
                f"RMSE={fmt_metric(metric_group['rmse_m'])} m",
                f"RMSE_wet={fmt_metric(metric_group['rmse_wet_m'])} m",
                f"Bias={fmt_metric(metric_group['bias_m'])} m",
            ])
        ax_hist.text(0.98, 0.95, chr(10).join(info_lines), transform=ax_hist.transAxes, ha="right", va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 3})
        ax_hist.set_title(f"{title} histogram")
        im = ax_img.imshow(masked, cmap="cividis", vmin=0.0, vmax=vmax_depth)
        ax_img.set_title(f"{title} raster")
        ax_img.set_axis_off()
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04, label="Depth (m)")
    title_suffix = "ResUNet vs CostGrow" if aligned_d["costgrow_prediction"] is not None else "ResUNet"
    fig.suptitle(f"{case_d['case_name']} diagnostics: {title_suffix}", fontsize=13)
    fig.tight_layout()
    metric_keys = ["psnr", "ssim", "mase_m", "rmse_m", "rmse_wet_m", "bias_m"]
    return fig, pd.DataFrame(metrics_d).loc[metric_keys]


def fmt_res(profile: dict) -> str:
    """Format x/y pixel size from a raster profile."""
    transform = profile["transform"]
    return f"({abs(float(transform.a)):.6g}, {abs(float(transform.e)):.6g})"


def stats_text(values: np.ndarray) -> str:
    """Format basic summary statistics for an annotation box."""
    if values.size == 0:
        return "n=0"
    return chr(10).join([
        f"n={values.size:,}",
        f"min={values.min():.4f}",
        f"mean={values.mean():.4f}",
        f"max={values.max():.4f}",
        f"std={values.std():.4f}",
    ])


def fmt_metric(value: float, ndigits: int = 4) -> str:
    """Format one metric value for figure annotation."""
    return "nan" if value is None or not np.isfinite(value) else f"{value:.{ndigits}f}"
