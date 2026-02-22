"""Evaluation and error-metric helpers for analysis workflows."""

import numpy as np


def compute_depth_error_metrics(
    reference_depth_m: np.ndarray,
    estimate_depth_m: np.ndarray,
    max_depth: float,
    dry_depth_thresh_m: float = 1e-3,
) -> dict[str, float]:
    """Compute pairwise depth error metrics for one tile."""
    assert reference_depth_m.ndim == 2, f"reference depth must be 2D; got {reference_depth_m.shape}"
    assert estimate_depth_m.shape == reference_depth_m.shape, (
        f"estimate shape {estimate_depth_m.shape} must match reference shape {reference_depth_m.shape}"
    )
    assert max_depth > 0, f"max_depth must be > 0; got {max_depth}"

    # Normalize array dtypes once for deterministic metric math.
    reference_np = reference_depth_m.astype(np.float32, copy=False)
    estimate_np = estimate_depth_m.astype(np.float32, copy=False)

    # Compute all pixel-wise deltas and wet/dry masks.
    diff = estimate_np - reference_np
    wet_mask = reference_np >= dry_depth_thresh_m
    wet_pixel_count = int(wet_mask.sum())
    total_pixels = int(reference_np.size)
    dry_pixel_count = int(total_pixels - wet_pixel_count)

    # Aggregate core error statistics in depth units.
    mse_all = float(np.mean(np.square(diff), dtype=np.float64))
    rmse_all = float(np.sqrt(mse_all))
    mae_all = float(np.mean(np.abs(diff), dtype=np.float64))
    bias_all = float(np.mean(diff, dtype=np.float64))

    if wet_pixel_count > 0:
        rmse_wet = float(np.sqrt(np.mean(np.square(diff[wet_mask]), dtype=np.float64)))
    else:
        rmse_wet = np.nan

    if mse_all <= 0.0:
        psnr = np.inf
    else:
        psnr = float(20.0 * np.log10(max_depth) - 10.0 * np.log10(mse_all))

    # Compute SSIM over the full tile using max_depth-derived constants.
    ref64 = reference_np.astype(np.float64, copy=False)
    est64 = estimate_np.astype(np.float64, copy=False)
    mu_x = float(ref64.mean())
    mu_y = float(est64.mean())
    sigma_x = float(ref64.var())
    sigma_y = float(est64.var())
    sigma_xy = float(((ref64 - mu_x) * (est64 - mu_y)).mean())

    c1 = float((0.01 * max_depth) ** 2)
    c2 = float((0.03 * max_depth) ** 2)
    ssim_num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    ssim_den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = float(ssim_num / ssim_den) if ssim_den != 0.0 else np.nan

    return {
        "psnr": psnr,
        "ssim": ssim,
        "rmse_m": rmse_all,
        "rmse_wet_m": rmse_wet,
        "mae_m": mae_all,
        "mase_m": mae_all,
        "bias_m": bias_all,
        "mse_m2": mse_all,
        "dry_pixel_count": dry_pixel_count,
        "wet_pixel_count": wet_pixel_count,
    }
