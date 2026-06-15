"""INT8 weight-only quantization for large ONNX models with external data.

ORT's quantize_dynamic segfaults on models >2GB (protobuf limit).
This module works around that by:
  1. Loading the ONNX graph WITHOUT external data (stays <2GB)
  2. Reading each weight tensor from the external data file
  3. Quantizing to INT8 with per-tensor symmetric scaling
  4. Inserting DequantizeLinear nodes so ORT can dequantize at runtime
  5. Writing a new consolidated external data file

The result is a valid ONNX model that ORT loads natively.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import (
    ExternalDataInfo,
    set_external_data,
    uses_external_data,
    _get_all_tensors,
)

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Helpers                                                                  #
# ====================================================================== #

_ONNX_DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.BFLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.INT8: 1,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.UINT8: 1,
    TensorProto.BOOL: 1,
}


def _expected_byte_size(tensor):
    """Compute expected raw data size from dims and dtype."""
    elem_size = _ONNX_DTYPE_BYTES.get(tensor.data_type, 0)
    num_elems = 1
    for d in tensor.dims:
        num_elems *= d
    return num_elems * elem_size


def _read_external_tensor(tensor, base_dir):
    """Read raw bytes of an externally-stored tensor."""
    info = ExternalDataInfo(tensor)
    path = (
        info.location if os.path.isabs(info.location)
        else os.path.join(base_dir, info.location)
    )
    expected = _expected_byte_size(tensor)
    with open(path, "rb") as f:
        if info.offset is not None:
            f.seek(info.offset)
        if info.length is not None and info.length > 0:
            return f.read(info.length)
        if expected > 0:
            return f.read(expected)
        return f.read()


def _tensor_to_numpy(tensor, raw_data):
    """Convert raw bytes to numpy array based on tensor dtype."""
    dtype = tensor.data_type
    if dtype == TensorProto.FLOAT:
        return np.frombuffer(raw_data, dtype=np.float32).reshape(tensor.dims)
    elif dtype == TensorProto.FLOAT16:
        return np.frombuffer(raw_data, dtype=np.float16).reshape(tensor.dims)
    elif dtype == TensorProto.BFLOAT16:
        bf16 = np.frombuffer(raw_data, dtype=np.uint16)
        fp32_bits = bf16.astype(np.uint32) << 16
        fp32 = np.frombuffer(fp32_bits.tobytes(), dtype=np.float32)
        return fp32.reshape(tensor.dims)
    elif dtype == TensorProto.DOUBLE:
        return np.frombuffer(raw_data, dtype=np.float64).reshape(tensor.dims)
    else:
        return None


def _quantize_to_int8(arr):
    """Per-tensor symmetric INT8 quantization. Returns (int8_arr, scale)."""
    arr_fp32 = arr.astype(np.float32)
    abs_max = np.max(np.abs(arr_fp32))
    if abs_max == 0:
        scale = np.float32(1.0)
    else:
        scale = np.float32(abs_max / 127.0)
    quantized = np.clip(np.round(arr_fp32 / scale), -128, 127).astype(np.int8)
    return quantized, scale


def _is_weight_tensor(name):
    """Heuristic: only quantize weight matrices, not biases/norms/embeddings."""
    n = name.lower()
    for skip in (".q.weight", ".k.weight", ".v.weight"):
        if skip in n:
            return False
    for skip in ("bias", "norm", "embed", "position", "relative_attention"):
        if skip in n:
            return False
    for keep in ("weight", "kernel"):
        if keep in n:
            return True
    return False


def _find_consumers(graph, tensor_name):
    """Find all nodes that consume a given tensor name as input."""
    consumers = []
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == tensor_name:
                consumers.append((node, i))
    return consumers


# ====================================================================== #
#  Main quantization                                                        #
# ====================================================================== #

def quantize_large_model(input_path, output_path):
    """Quantize a large ONNX model with external data to INT8.

    Inserts DequantizeLinear nodes for each quantized weight so ORT
    can run the model natively.
    """
    base_dir = os.path.dirname(os.path.abspath(input_path))
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    sidecar_name = os.path.basename(output_path) + ".data"
    sidecar_path = os.path.join(out_dir, sidecar_name)

    logger.info("Loading ONNX graph (without external data)...")
    model = onnx.load(input_path, load_external_data=False)
    graph = model.graph

    # Build map: initializer name -> initializer proto
    init_map: Dict[str, TensorProto] = {}
    for init in graph.initializer:
        init_map[init.name] = init

    # Collect quantization info: (tensor_name, int8_bytes, scale, orig_dims)
    quant_info: List[Tuple[str, bytes, float, list]] = []
    quantized_names: Set[str] = set()
    quantized_count = 0
    skipped_count = 0
    offset = 0

    # Phase 1: Quantize weights and write external data
    logger.info("Phase 1: Quantizing weights...")
    with open(sidecar_path, "wb") as sidecar_f:
        for tensor in _get_all_tensors(model):
            if not uses_external_data(tensor):
                continue

            raw_data = _read_external_tensor(tensor, base_dir)
            name = tensor.name

            arr = _tensor_to_numpy(tensor, raw_data)
            should_quantize = (
                arr is not None
                and arr.ndim >= 2
                and _is_weight_tensor(name)
            )

            if should_quantize:
                q_arr, scale = _quantize_to_int8(arr)
                new_data = q_arr.tobytes()
                tensor.data_type = TensorProto.INT8
                quant_info.append((name, scale, list(tensor.dims)))
                quantized_names.add(name)
                quantized_count += 1
                logger.info(
                    "  Quantized %-60s %s -> int8 (scale=%.6f)",
                    name, list(tensor.dims), float(scale),
                )
            else:
                new_data = raw_data
                skipped_count += 1

            sidecar_f.write(new_data)
            if not tensor.HasField("raw_data"):
                tensor.raw_data = b""
            set_external_data(
                tensor,
                location=sidecar_name,
                offset=offset,
                length=len(new_data),
            )
            offset += len(new_data)

    logger.info(
        "Quantized %d tensors, skipped %d", quantized_count, skipped_count
    )

    # Phase 2: Insert DequantizeLinear nodes
    logger.info("Phase 2: Inserting DequantizeLinear nodes...")
    new_nodes = list(graph.node)

    for orig_name, scale_val, dims in quant_info:
        dq_output = orig_name + "_dequantized"
        scale_name = orig_name + "_scale"
        zp_name = orig_name + "_zero_point"

        # Add scale initializer (scalar float32)
        scale_tensor = numpy_helper.from_array(
            np.array(scale_val, dtype=np.float32), name=scale_name
        )
        graph.initializer.append(scale_tensor)

        # Add zero_point initializer (scalar int8, symmetric = 0)
        zp_tensor = numpy_helper.from_array(
            np.array(0, dtype=np.int8), name=zp_name
        )
        graph.initializer.append(zp_tensor)

        # Create DequantizeLinear node: int8_weight, scale, zp -> float_weight
        dq_node = helper.make_node(
            "DequantizeLinear",
            inputs=[orig_name, scale_name, zp_name],
            outputs=[dq_output],
            name=f"DequantizeLinear_{orig_name}",
        )

        # Rewire all consumers of the original weight to use dequantized output
        for node in new_nodes:
            for i, inp in enumerate(node.input):
                if inp == orig_name:
                    node.input[i] = dq_output

        new_nodes.append(dq_node)

    # Replace graph nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    logger.info("Phase 3: Saving quantized model...")
    with open(output_path, "wb") as f:
        f.write(model.SerializeToString())

    graph_size = os.path.getsize(output_path) / 2**20
    data_size = os.path.getsize(sidecar_path) / 2**30
    logger.info(
        "Output: %s (%.1f MB graph, %.1f GB data)",
        output_path, graph_size, data_size,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output_int8.onnx>")
        sys.exit(1)
    quantize_large_model(sys.argv[1], sys.argv[2])
