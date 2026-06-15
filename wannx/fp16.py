"""FP16 conversion helpers for large ONNX models with external data.

There are two conversion modes:

1. storage-only fp16:
   converts float32 initializers to float16 but inserts Cast(to=FLOAT)
   nodes so the graph still computes in float32. This is numerically
   conservative and reduces sidecar size, but keeps the runtime graph
   fp32-heavy.

2. native fp16:
   converts the graph itself to float16 (with ONNX Runtime's official
   float16 graph converter). This is the mode needed for large DiT
   models where startup/runtime cost matters, because it removes the
   per-weight fp16->fp32 Cast pattern.
"""

import logging
import os
import sys
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import (
    ExternalDataInfo,
    set_external_data,
    uses_external_data,
    _get_all_tensors,
)
from onnxruntime.transformers.float16 import (
    convert_float_to_float16 as ort_convert_float_to_float16,
)

logger = logging.getLogger(__name__)

_EXTERNALIZE_ATTR_THRESHOLD_BYTES = 1 << 20


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


def _read_tensor_data(tensor, base_dir):
    """Read raw bytes for an initializer stored internally or externally."""
    if uses_external_data(tensor):
        return _read_external_tensor(tensor, base_dir)
    if tensor.HasField("raw_data") and tensor.raw_data:
        return tensor.raw_data
    arr = numpy_helper.to_array(tensor)
    return arr.tobytes()


def _to_fp16_bytes(raw_data):
    arr = np.frombuffer(raw_data, dtype=np.float32)
    return arr.astype(np.float16).tobytes()


def _should_externalize_attribute_tensor(tensor, raw_data: bytes) -> bool:
    """Keep large/externally-stored attribute tensors out of the protobuf graph.

    Large DiT exports often store model weights inside Constant-node attribute
    tensors rather than graph.initializer. Re-embedding those tensors into the
    protobuf graph makes serialization explode for big profiles (for example
    14B/81f). Preserve existing external-data layout for those tensors and
    externalize any large attribute payloads as a safety fallback.
    """
    return uses_external_data(tensor) or len(raw_data) >= _EXTERNALIZE_ATTR_THRESHOLD_BYTES


def _iter_graphs(model: onnx.ModelProto) -> Iterable[onnx.GraphProto]:
    """Yield the main graph and any nested subgraphs."""
    stack = [model.graph]
    while stack:
        graph = stack.pop()
        yield graph
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    stack.append(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    stack.extend(attr.graphs)


def _convert_value_info_to_fp16(value_info):
    if value_info.type.HasField("tensor_type"):
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
    if value_info.type.HasField("sequence_type"):
        elem = value_info.type.sequence_type.elem_type
        if elem.HasField("tensor_type") and elem.tensor_type.elem_type == TensorProto.FLOAT:
            elem.tensor_type.elem_type = TensorProto.FLOAT16


def _iter_initializer_tensors(model: onnx.ModelProto):
    for graph in _iter_graphs(model):
        for tensor in graph.initializer:
            yield tensor


def _should_convert(tensor):
    """Decide whether to convert this tensor to fp16.

    Converts float32 weight matrices (2D+).
    Skips biases, norms, embeddings, and small tensors for numerical stability.
    """
    if tensor.data_type != TensorProto.FLOAT:
        return False
    # Only convert 2D+ tensors (weight matrices / conv kernels)
    if len(tensor.dims) < 2:
        return False
    name = tensor.name.lower()
    for skip in ("bias", "norm", "embed", "position", "relative_attention",
                 "modulation"):
        if skip in name:
            return False
    return True


# ====================================================================== #
#  Main conversion                                                          #
# ====================================================================== #

def _save_external_model(model: onnx.ModelProto, output_path: str):
    """Save ONNX model with a single external-data sidecar."""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    sidecar_name = os.path.basename(output_path) + ".data"
    sidecar_path = os.path.join(out_dir, sidecar_name)
    if os.path.isfile(sidecar_path):
        os.remove(sidecar_path)
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=sidecar_name,
        size_threshold=0,
        convert_attribute=False,
    )
    return sidecar_path


def convert_to_fp16_storage(input_path, output_path):
    """Convert float32 weights in a large ONNX model to float16.

    Inserts Cast(to=FLOAT) nodes for each converted weight so the
    graph remains numerically valid. ORT fuses these on GPU.
    """
    base_dir = os.path.dirname(os.path.abspath(input_path))
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    sidecar_name = os.path.basename(output_path) + ".data"
    sidecar_path = os.path.join(out_dir, sidecar_name)

    logger.info("Loading ONNX graph (without external data)...")
    model = onnx.load(input_path, load_external_data=False)
    graph = model.graph
    initializer_names: Set[str] = {t.name for t in _iter_initializer_tensors(model)}
    constant_value_outputs: Dict[str, str] = {}
    for subgraph in _iter_graphs(model):
        for node in subgraph.node:
            if node.op_type != "Constant" or not node.output:
                continue
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.name == "value" and attr.t.name:
                    constant_value_outputs[attr.t.name] = node.output[0]

    # Collect conversion info: (tensor_name, dims)
    convert_info: List[Tuple[str, list]] = []
    converted_count = 0
    skipped_count = 0
    offset = 0

    # Phase 1: Convert weights and write external data
    logger.info("Phase 1: Converting weights to FP16...")
    with open(sidecar_path, "wb") as sidecar_f:
        for tensor in _get_all_tensors(model):
            name = tensor.name
            raw_data = _read_tensor_data(tensor, base_dir)

            if _should_convert(tensor):
                # Read as fp32, convert to fp16
                arr = np.frombuffer(raw_data, dtype=np.float32)
                arr_fp16 = arr.astype(np.float16)
                new_data = arr_fp16.tobytes()
                tensor.data_type = TensorProto.FLOAT16
                convert_info.append((constant_value_outputs.get(name, name), list(tensor.dims)))
                converted_count += 1
                logger.debug(
                    "  FP16 %-60s %s (%.1f MB -> %.1f MB)",
                    name, list(tensor.dims),
                    len(raw_data) / 2**20, len(new_data) / 2**20,
                )
            else:
                new_data = raw_data
                skipped_count += 1

            if tensor.name in initializer_names:
                sidecar_f.write(new_data)
                if tensor.HasField("raw_data"):
                    tensor.ClearField("raw_data")
                tensor.raw_data = b""
                set_external_data(
                    tensor,
                    location=sidecar_name,
                    offset=offset,
                    length=len(new_data),
                )
                offset += len(new_data)
            else:
                # Preserve large/external Constant attrs outside the protobuf
                # graph, otherwise huge DiT exports become unserializable.
                if _should_externalize_attribute_tensor(tensor, new_data) or _should_convert(tensor):
                    sidecar_f.write(new_data)
                    if tensor.HasField("raw_data"):
                        tensor.ClearField("raw_data")
                    tensor.raw_data = b""
                    set_external_data(
                        tensor,
                        location=sidecar_name,
                        offset=offset,
                        length=len(new_data),
                    )
                    offset += len(new_data)
                else:
                    tensor.ClearField("external_data")
                    tensor.data_location = TensorProto.DEFAULT
                    if tensor.HasField("raw_data"):
                        tensor.ClearField("raw_data")
                    tensor.raw_data = new_data

    logger.info(
        "Converted %d tensors to FP16, skipped %d",
        converted_count, skipped_count,
    )

    # Phase 2: Insert Cast nodes (fp16 -> fp32)
    logger.info("Phase 2: Inserting Cast nodes...")
    original_nodes = list(graph.node)
    cast_nodes = []
    produced_outputs = {out for node in original_nodes for out in node.output}
    cast_nodes_by_input: Dict[str, onnx.NodeProto] = {}

    for orig_name, dims in convert_info:
        cast_output = orig_name + "_fp32"

        # Create Cast node: fp16_weight -> fp32
        cast_node = helper.make_node(
            "Cast",
            inputs=[orig_name],
            outputs=[cast_output],
            name=f"Cast_fp16to32_{orig_name}",
            to=TensorProto.FLOAT,
        )

        # Rewire all consumers of the original weight to use cast output
        for node in original_nodes:
            for i, inp in enumerate(node.input):
                if inp == orig_name:
                    node.input[i] = cast_output

        cast_nodes.append(cast_node)
        cast_nodes_by_input[orig_name] = cast_node

    # Replace graph nodes
    del graph.node[:]
    for cast_node in cast_nodes:
        if cast_node.input[0] not in produced_outputs:
            graph.node.append(cast_node)
    for node in original_nodes:
        graph.node.append(node)
        for out in node.output:
            cast_node = cast_nodes_by_input.get(out)
            if cast_node is not None:
                graph.node.append(cast_node)

    logger.info("Phase 3: Saving FP16 model...")
    with open(output_path, "wb") as f:
        f.write(model.SerializeToString())

    graph_size = os.path.getsize(output_path) / 2**20
    data_size = os.path.getsize(sidecar_path) / 2**30
    logger.info(
        "Output: %s (%.1f MB graph, %.1f GB data)",
        output_path, graph_size, data_size,
    )


def insert_fp16_initializer_casts(input_path, output_path=None):
    """Insert fp16->fp32 Cast nodes for existing fp16 initializers.

    This is useful for graphs that were exported directly in fp16 but still
    contain float32 activation paths. Rewriting all initializer consumers
    through Cast nodes restores the historical "storage-only fp16" behavior
    without touching the external weight data.
    """
    from .converters.base import _write_model_proto_preserve_external_data

    if output_path is None:
        output_path = input_path

    model = onnx.load_model(input_path, load_external_data=False)
    try:
        inferred_model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred_model = model
    graph = model.graph
    inferred_graph = inferred_model.graph

    produced_outputs = {out for node in graph.node for out in node.output}
    original_nodes = list(graph.node)
    cast_nodes = []
    cast_nodes_by_input: Dict[str, onnx.NodeProto] = {}

    type_map: Dict[str, int] = {t.name: t.data_type for t in graph.initializer}

    def _record_value_info(value_info):
        tt = value_info.type.tensor_type
        if tt.elem_type:
            type_map[value_info.name] = tt.elem_type

    for value_info in list(inferred_graph.input) + list(inferred_graph.value_info) + list(inferred_graph.output):
        _record_value_info(value_info)

    producer_by_output = {out: node for node in original_nodes for out in node.output}

    def _initializer_needs_fp32_cast(name: str) -> bool:
        for node in original_nodes:
            if name not in node.input:
                continue
            for other in node.input:
                if other == name:
                    continue
                other_type = type_map.get(other)
                if other_type == TensorProto.FLOAT:
                    return True
        return False

    for tensor in graph.initializer:
        if tensor.data_type != TensorProto.FLOAT16:
            continue
        orig_name = tensor.name
        if not _initializer_needs_fp32_cast(orig_name):
            continue
        cast_output = orig_name + "_fp32"
        cast_node = helper.make_node(
            "Cast",
            inputs=[orig_name],
            outputs=[cast_output],
            name=f"Cast_fp16to32_{orig_name}",
            to=TensorProto.FLOAT,
        )
        for node in original_nodes:
            should_rewire = False
            if orig_name in node.input:
                for other in node.input:
                    if other == orig_name:
                        continue
                    if type_map.get(other) == TensorProto.FLOAT:
                        should_rewire = True
                        break
            if not should_rewire:
                continue
            for i, inp in enumerate(node.input):
                if inp == orig_name:
                    node.input[i] = cast_output
        cast_nodes.append(cast_node)
        cast_nodes_by_input[orig_name] = cast_node

    passthrough_weight_ops = {"Transpose", "Identity"}

    for node in original_nodes:
        consumer_has_float = any(type_map.get(inp) == TensorProto.FLOAT for inp in node.input)
        if not consumer_has_float:
            continue
        for i, inp in enumerate(node.input):
            producer = producer_by_output.get(inp)
            if producer is None or producer.op_type not in passthrough_weight_ops:
                continue
            if not producer.input:
                continue
            producer_inp = producer.input[0]
            producer_inp_type = type_map.get(producer_inp)
            if producer_inp_type != TensorProto.FLOAT16:
                continue
            cast_output = inp + "_fp32"
            if inp not in cast_nodes_by_input:
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[inp],
                    outputs=[cast_output],
                    name=f"Cast_fp16to32_{inp}",
                    to=TensorProto.FLOAT,
                )
                cast_nodes.append(cast_node)
                cast_nodes_by_input[inp] = cast_node
            node.input[i] = cast_output

    del graph.node[:]
    for cast_node in cast_nodes:
        if cast_node.input[0] not in produced_outputs:
            graph.node.append(cast_node)
    for node in original_nodes:
        graph.node.append(node)
        for out in node.output:
            cast_node = cast_nodes_by_input.get(out)
            if cast_node is not None:
                graph.node.append(cast_node)

    _write_model_proto_preserve_external_data(model, output_path)


def convert_to_native_fp16(
    input_path,
    output_path,
    *,
    keep_io_types=False,
    disable_shape_infer=True,
    op_block_list=None,
    node_block_list=None,
    force_fp16_initializers=True,
):
    """Convert an ONNX graph to true native fp16.

    Unlike storage-only conversion, this rewrites the graph itself to use
    fp16 where supported, so large models do not keep per-weight Cast
    nodes back to fp32 at runtime.
    """
    logger.info("Loading and converting ONNX graph to native FP16...")
    model = ort_convert_float_to_float16(
        input_path,
        keep_io_types=keep_io_types,
        disable_shape_infer=disable_shape_infer,
        op_block_list=op_block_list,
        node_block_list=node_block_list,
        force_fp16_initializers=force_fp16_initializers,
    )

    logger.info("Saving native FP16 model...")
    sidecar_path = _save_external_model(model, output_path)
    graph_size = os.path.getsize(output_path) / 2**20
    data_size = os.path.getsize(sidecar_path) / 2**30 if os.path.isfile(sidecar_path) else 0
    logger.info(
        "Output: %s (%.1f MB graph, %.1f GB data)",
        output_path, graph_size, data_size,
    )


def convert_to_native_fp16_streaming(input_path, output_path):
    """Convert an ONNX graph to native fp16 without loading all weights at once.

    This is a DiT-oriented fallback for very large external-data models where
    the generic ORT float16 graph converter is too memory-hungry.
    """
    base_dir = os.path.dirname(os.path.abspath(input_path))
    model = onnx.load(input_path, load_external_data=False)
    initializer_names: Set[str] = {t.name for t in _iter_initializer_tensors(model)}
    for graph in _iter_graphs(model):
        # Keep existing mixed-precision intent inside the graph. In particular,
        # preserve original Cast(to=FLOAT) nodes and Constant-node attribute
        # tensors; only the public I/O contract is rewritten to fp16.
        for value_info in list(graph.input) + list(graph.output):
            _convert_value_info_to_fp16(value_info)

    sidecar_name = os.path.basename(output_path) + ".data"
    sidecar_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), sidecar_name)
    logger.info("Streaming native FP16 conversion...")
    offset = 0
    current_tensor_name = None
    with open(sidecar_path, "wb") as sidecar_f:
        processed_names: Set[str] = set()

        try:
            for tensor in _iter_initializer_tensors(model):
                current_tensor_name = tensor.name
                raw_data = _read_tensor_data(tensor, base_dir)
                if tensor.data_type == TensorProto.FLOAT:
                    raw_data = _to_fp16_bytes(raw_data)
                    tensor.data_type = TensorProto.FLOAT16

                sidecar_f.write(raw_data)
                sidecar_f.flush()
                if tensor.HasField("raw_data"):
                    tensor.ClearField("raw_data")
                tensor.raw_data = b""
                set_external_data(
                    tensor,
                    location=sidecar_name,
                    offset=offset,
                    length=len(raw_data),
                )
                offset += len(raw_data)
                processed_names.add(tensor.name)

            for tensor in _get_all_tensors(model):
                current_tensor_name = tensor.name
                if tensor.name in processed_names:
                    continue
                raw_data = _read_tensor_data(tensor, base_dir)
                should_externalize_attr = _should_externalize_attribute_tensor(tensor, raw_data)
                if tensor.name in initializer_names:
                    if tensor.data_type == TensorProto.FLOAT:
                        raw_data = _to_fp16_bytes(raw_data)
                        tensor.data_type = TensorProto.FLOAT16
                    sidecar_f.write(raw_data)
                    sidecar_f.flush()
                    if tensor.HasField("raw_data"):
                        tensor.ClearField("raw_data")
                    tensor.raw_data = b""
                    set_external_data(
                        tensor,
                        location=sidecar_name,
                        offset=offset,
                        length=len(raw_data),
                    )
                    offset += len(raw_data)
                else:
                    # Preserve Constant-node attribute tensors that already live
                    # in external data, and externalize any large attribute
                    # payloads so huge DiT exports remain serializable. Only
                    # weight-like attrs are downcasted; small scalar/vector
                    # constants stay embedded and keep their original dtype.
                    if tensor.data_type == TensorProto.FLOAT and _should_convert(tensor):
                        raw_data = _to_fp16_bytes(raw_data)
                        tensor.data_type = TensorProto.FLOAT16

                    if should_externalize_attr or _should_convert(tensor):
                        sidecar_f.write(raw_data)
                        sidecar_f.flush()
                        if tensor.HasField("raw_data"):
                            tensor.ClearField("raw_data")
                        tensor.raw_data = b""
                        set_external_data(
                            tensor,
                            location=sidecar_name,
                            offset=offset,
                            length=len(raw_data),
                        )
                        offset += len(raw_data)
                    else:
                        tensor.ClearField("external_data")
                        tensor.data_location = TensorProto.DEFAULT
                        if tensor.HasField("raw_data"):
                            tensor.ClearField("raw_data")
                        tensor.raw_data = raw_data
        except Exception as exc:
            raise RuntimeError(
                f"Streaming native-fp16 conversion failed at tensor '{current_tensor_name}' "
                f"(offset={offset})"
            ) from exc

    try:
        with open(output_path, "wb") as f:
            f.write(model.SerializeToString())
    except Exception as exc:
        raise RuntimeError(
            f"Streaming native-fp16 serialization failed after tensor '{current_tensor_name}' "
            f"(offset={offset})"
        ) from exc

    graph_size = os.path.getsize(output_path) / 2**20
    data_size = os.path.getsize(sidecar_path) / 2**30 if os.path.isfile(sidecar_path) else 0
    logger.info(
        "Output: %s (%.1f MB graph, %.1f GB data)",
        output_path, graph_size, data_size,
    )


def convert_to_fp16(input_path, output_path, *, native=False, streaming=False):
    """Convert ONNX model to fp16.

    Args:
        native: when True, convert the graph to real fp16; when False,
        keep the historical storage-only fp16 + Cast-to-fp32 behavior.
    """
    if native and streaming:
        return convert_to_native_fp16_streaming(input_path, output_path)
    if native:
        return convert_to_native_fp16(input_path, output_path)
    return convert_to_fp16_storage(input_path, output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output_fp16.onnx>")
        sys.exit(1)
    convert_to_fp16(sys.argv[1], sys.argv[2])
