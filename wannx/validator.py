"""ONNX model validation and benchmarking."""

import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import onnx
import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_providers() -> List[str]:
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def validate(onnx_path: str) -> Dict[str, Any]:
    """Validate an ONNX file: structure check, trial inference, quick benchmark.

    Returns a dict with validation results.
    """
    result: Dict[str, Any] = {
        "path": onnx_path,
        "valid": False,
        "structure_ok": False,
        "inference_ok": False,
        "size_mb": 0.0,
        "inputs": [],
        "outputs": [],
        "errors": [],
    }

    if not os.path.isfile(onnx_path):
        result["errors"].append(f"File not found: {onnx_path}")
        return result

    result["size_mb"] = os.path.getsize(onnx_path) / (1024 * 1024)

    # ---- structure check ------------------------------------------------ #
    try:
        # Use path-based checker to support >2GB models with external data.
        onnx.checker.check_model(onnx_path)
        result["structure_ok"] = True
    except Exception as exc:
        result["errors"].append(f"Structure check failed: {exc}")
        return result

    # ---- trial inference ------------------------------------------------ #
    try:
        providers = get_providers()
        sess = ort.InferenceSession(onnx_path, providers=providers)

        result["inputs"] = [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in sess.get_inputs()
        ]
        result["outputs"] = [
            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in sess.get_outputs()
        ]

        # Build dummy feeds
        feeds: Dict[str, np.ndarray] = {}
        for inp in sess.get_inputs():
            shape = _resolve_shape(inp.shape)
            dtype = _onnx_type_to_numpy(inp.type)
            if np.issubdtype(dtype, np.integer):
                feeds[inp.name] = np.random.randint(0, 100, shape).astype(dtype)
            else:
                feeds[inp.name] = np.random.randn(*shape).astype(dtype)

        outputs = sess.run(None, feeds)
        result["inference_ok"] = True
        result["output_shapes"] = [list(o.shape) for o in outputs]

        # Quick benchmark (5 iterations)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            sess.run(None, feeds)
            times.append((time.perf_counter() - t0) * 1000)
        result["avg_inference_ms"] = float(np.mean(times))
        result["std_inference_ms"] = float(np.std(times))

    except Exception as exc:
        result["errors"].append(f"Inference failed: {exc}")
        return result

    result["valid"] = result["structure_ok"] and result["inference_ok"]
    return result


def _resolve_shape(shape) -> List[int]:
    """Replace dynamic / symbolic dims with small concrete values."""
    resolved = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            resolved.append(d)
        elif isinstance(d, str):
            if "batch" in d:
                resolved.append(1)
            elif "seq" in d or "len" in d:
                resolved.append(16)
            elif "frame" in d:
                resolved.append(4)
            elif "height" in d:
                resolved.append(32)
            elif "width" in d:
                resolved.append(32)
            else:
                resolved.append(4)
        else:
            resolved.append(4)
    return resolved


def _onnx_type_to_numpy(onnx_type: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(onnx_type, np.float32)
