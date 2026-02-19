"""ONNX Runtime CPU engine implementation for FloodSR."""

import logging, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from floodsr.engine.base import EngineBase
from floodsr.inference import invert_depth_log1p_np, normalize_dem, replace_nodata_with_zero, scale_depth_log1p_np


@dataclass(frozen=True)
class ModelIOContract:
    """Resolved model tensor names and spatial dimensions."""

    depth_input_name: str
    dem_input_name: str
    output_name: str
    depth_lr_hwc: tuple[int, int, int]
    dem_hr_hwc: tuple[int, int, int]
    output_hwc: tuple[int, int, int]
    scale: int


class EngineORT(EngineBase):
    """CPU ONNX Runtime engine with notebook-compatible model I/O wiring."""

    def __init__(
        self,
        model_fp: str | Path,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        logger=None,
    ):
        """Initialize and load an ORT session."""
        self._model_fp = Path(model_fp).expanduser().resolve()
        assert self._model_fp.exists(), f"model file does not exist: {self._model_fp}"
        assert providers, "providers cannot be empty"
        self.providers = tuple(providers)
        self.log = logger or logging.getLogger(__name__)
        self.session: ort.InferenceSession | None = None
        self.contract: ModelIOContract | None = None
        self.load()

    def model_path(self) -> Path:
        """Return the model path used by this engine."""
        return self._model_fp

    def load(self) -> None:
        """Load model session and resolve model I/O contract."""
        self.log.debug(f"loading ORT session from\n    {self._model_fp}")
        self.session = ort.InferenceSession(self._model_fp.as_posix(), providers=list(self.providers))
        self.contract = self._resolve_contract()
        self.log.info(
            f"loaded ORT model '{self._model_fp.name}' with providers={self.session.get_providers()} "
            f"and scale={self.contract.scale}"
        )

    def _resolve_hwc(self, dims: list[Any], tensor_name: str) -> tuple[int, int, int]:
        """Resolve fixed HWC dimensions from a rank-4 NHWC tensor shape."""
        assert len(dims) == 4, f"{tensor_name} must be rank-4 NHWC; got {dims}"
        h, w, c = dims[1], dims[2], dims[3]
        assert isinstance(h, int) and h > 0, f"{tensor_name} height must be fixed int; got {h}"
        assert isinstance(w, int) and w > 0, f"{tensor_name} width must be fixed int; got {w}"
        assert isinstance(c, int) and c == 1, f"{tensor_name} channels must be 1; got {c}"
        return (h, w, c)

    def _resolve_contract(self) -> ModelIOContract:
        """Extract model names and dimensions from ORT metadata."""
        assert self.session is not None, "session must be loaded before resolving contract"
        input_meta_d = {node.name: list(node.shape) for node in self.session.get_inputs()}
        output_meta_l = list(self.session.get_outputs())
        assert "depth_lr" in input_meta_d, "model input 'depth_lr' not found"
        assert "dem_hr" in input_meta_d, "model input 'dem_hr' not found"
        assert len(output_meta_l) > 0, "model outputs are empty"

        output_name = output_meta_l[0].name
        output_dims = list(output_meta_l[0].shape)
        depth_lr_hwc = self._resolve_hwc(input_meta_d["depth_lr"], "depth_lr")
        dem_hr_hwc = self._resolve_hwc(input_meta_d["dem_hr"], "dem_hr")
        output_hwc = self._resolve_hwc(output_dims, output_name)
        assert dem_hr_hwc == output_hwc, f"DEM input shape {dem_hr_hwc} must match output shape {output_hwc}"
        assert dem_hr_hwc[0] % depth_lr_hwc[0] == 0, (
            f"HR/LR height ratio must be integer; got HR={dem_hr_hwc}, LR={depth_lr_hwc}"
        )
        scale = int(dem_hr_hwc[0] // depth_lr_hwc[0])
        return ModelIOContract(
            depth_input_name="depth_lr",
            dem_input_name="dem_hr",
            output_name=output_name,
            depth_lr_hwc=depth_lr_hwc,
            dem_hr_hwc=dem_hr_hwc,
            output_hwc=output_hwc,
            scale=scale,
        )

    def _build_feed_dict(self, depth_lr_nhwc: np.ndarray, dem_hr_nhwc: np.ndarray) -> dict[str, np.ndarray]:
        """Build feed dictionary and validate static input dimensions."""
        assert self.session is not None, "session must be loaded before inference"
        assert self.contract is not None, "model contract must be available before inference"
        feed_dict: dict[str, np.ndarray] = {}
        for input_meta in self.session.get_inputs():
            if input_meta.name == self.contract.depth_input_name:
                input_ar = depth_lr_nhwc
            elif input_meta.name == self.contract.dem_input_name:
                input_ar = dem_hr_nhwc
            else:
                raise AssertionError(f"unexpected model input name: {input_meta.name}")

            for axis, (got, exp) in enumerate(zip(input_ar.shape, input_meta.shape)):
                if isinstance(exp, int) and exp > 0:
                    assert got == exp, (
                        f"input {input_meta.name} axis {axis} expects {exp}, got {got}; "
                        f"expected shape={input_meta.shape}, got={input_ar.shape}"
                    )
            feed_dict[input_meta.name] = input_ar
        assert self.contract.depth_input_name in feed_dict, "missing depth input for model"
        assert self.contract.dem_input_name in feed_dict, "missing dem input for model"
        return feed_dict

    def run_tile(
        self,
        depth_lr_m: np.ndarray,
        dem_hr_m: np.ndarray,
        max_depth: float = 5.0,
        dem_pct_clip: float = 95.0,
        dem_ref_stats: dict[str, float] | None = None,
        depth_lr_nodata: float | None = None,
        dem_hr_nodata: float | None = None,
        logger=None,
    ) -> dict[str, Any]:
        """Run one inference pass from raw depth/DEM arrays to predicted depth."""
        log = logger or self.log
        assert self.session is not None, "session must be loaded before inference"
        assert self.contract is not None, "model contract must be available before inference"
        start = time.perf_counter()
        log.debug(
            f"run_tile start: depth_lr_shape={np.shape(depth_lr_m)}, dem_hr_shape={np.shape(dem_hr_m)}, "
            f"max_depth={max_depth}, dem_pct_clip={dem_pct_clip}"
        )

        # Match notebook nodata policy before normalization.
        depth_lr_raw = replace_nodata_with_zero(depth_lr_m, depth_lr_nodata)
        dem_hr_raw = replace_nodata_with_zero(dem_hr_m, dem_hr_nodata)
        assert np.isfinite(depth_lr_raw).all(), "low-res depth contains non-finite values after nodata replacement"
        assert np.isfinite(dem_hr_raw).all(), "DEM contains non-finite values after nodata replacement"

        # Match notebook preprocessing and NHWC tensor formatting.
        depth_lr_norm = scale_depth_log1p_np(depth_lr_raw, max_depth=float(max_depth))
        dem_hr_norm, dem_stats_used = normalize_dem(dem_hr_raw, pct_clip=float(dem_pct_clip), ref_stats=dem_ref_stats)
        assert depth_lr_norm is not None, "depth normalization returned None"
        assert dem_hr_norm is not None, "DEM normalization returned None"
        assert dem_stats_used is not None, "DEM normalization did not return stats"
        depth_lr_nhwc = depth_lr_norm[np.newaxis, :, :, np.newaxis].astype(np.float32, copy=False)
        dem_hr_nhwc = dem_hr_norm[np.newaxis, :, :, np.newaxis].astype(np.float32, copy=False)
        assert depth_lr_nhwc.shape[1:] == self.contract.depth_lr_hwc, (
            f"depth tensor shape {depth_lr_nhwc.shape[1:]} != expected {self.contract.depth_lr_hwc}"
        )
        assert dem_hr_nhwc.shape[1:] == self.contract.dem_hr_hwc, (
            f"DEM tensor shape {dem_hr_nhwc.shape[1:]} != expected {self.contract.dem_hr_hwc}"
        )

        # Run the model and map normalized output back to depth units.
        feed_dict = self._build_feed_dict(depth_lr_nhwc, dem_hr_nhwc)
        outputs = self.session.run([self.contract.output_name], feed_dict)
        assert len(outputs) > 0, "model returned zero outputs"
        depth_hr_pred_norm = outputs[0][0, :, :, 0].astype(np.float32, copy=False)
        depth_hr_pred_m = invert_depth_log1p_np(depth_hr_pred_norm, max_depth=float(max_depth))
        assert depth_hr_pred_m is not None, "inverted prediction cannot be None"
        assert depth_hr_pred_m.shape == self.contract.output_hwc[:2], (
            f"prediction shape {depth_hr_pred_m.shape} != expected {self.contract.output_hwc[:2]}"
        )

        runtime_s = time.perf_counter() - start
        log.info(
            f"run_tile complete in {runtime_s:.3f}s; "
            f"pred_min={float(depth_hr_pred_m.min()):.6f}, pred_max={float(depth_hr_pred_m.max()):.6f}"
        )
        return {
            "prediction_m": depth_hr_pred_m.astype(np.float32, copy=False),
            "prediction_norm": depth_hr_pred_norm.astype(np.float32, copy=False),
            "dem_stats_used": dem_stats_used,
            "runtime_s": float(runtime_s),
        }

