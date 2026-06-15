"""Synthetic ONNX conversion tests.

Creates tiny model instances (no real checkpoints), wraps them in the
ONNX wrapper classes from wannx/converters/, exports to ONNX, validates
with onnx.checker, and runs inference with onnxruntime.
"""

import os
import sys
import tempfile
import traceback

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup: let Python find the WAN model sources and wannx package
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_WAN_ROOT = os.path.join(_ROOT, "models", "wan2.1")

for p in (_ROOT, _WAN_ROOT):
    p = os.path.normpath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

# Apply all runtime patches BEFORE importing any WAN modules
from wannx.patcher import apply_patches  # noqa: E402

apply_patches()

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export_and_validate(wrapper, dummy_inputs, input_names, output_names, name):
    """Export wrapper to ONNX, validate, and run ORT inference.

    Returns (ort_outputs, onnx_path) on success, raises on failure.
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"wannx_test_{name}_")
    onnx_path = os.path.join(tmp_dir, f"{name}.onnx")

    # Export
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )

    # Validate
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # Run with ORT
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {}
    for inp, tensor in zip(sess.get_inputs(), dummy_inputs):
        arr = tensor.detach().cpu().numpy()
        feeds[inp.name] = arr

    ort_outputs = sess.run(None, feeds)
    return ort_outputs, onnx_path


def _make_external_matmul_model(tmp_dir: str) -> str:
    """Create a tiny ONNX model with external initializer data."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    w_arr = (np.arange(16, dtype=np.float32).reshape(4, 4) / 8.0).copy()
    w = numpy_helper.from_array(w_arr, name="w")
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="matmul")
    graph = helper.make_graph([node], "matmul_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = onnx.IR_VERSION
    path = os.path.join(tmp_dir, "matmul.onnx")
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="matmul.onnx.data",
        size_threshold=0,
        convert_attribute=False,
    )
    return path


# =========================================================================
# Individual module tests
# =========================================================================

def test_dit():
    """DiT (WanModel) -> ONNX round-trip."""
    from wan.modules.model import WanModel
    from wannx.converters.dit import _ONNXWanModel

    dim = 64
    num_heads = 4
    patch_size = (1, 2, 2)
    in_dim = 8
    out_dim = 8
    text_dim = 64
    text_len = 16

    model = WanModel(
        model_type="t2v",
        patch_size=patch_size,
        text_len=text_len,
        in_dim=in_dim,
        dim=dim,
        ffn_dim=256,
        freq_dim=32,
        text_dim=text_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_layers=2,
    )
    model.eval()

    wrapper = _ONNXWanModel(model)
    wrapper.eval()

    # Input shapes: x[1,in_dim,F,H,W], t[1], ctx[1,text_len,text_dim]
    F_, H, W = 4, 8, 8
    x = torch.randn(1, in_dim, F_, H, W)
    t = torch.tensor([500.0])  # float32 — flow matching uses continuous timesteps
    ctx = torch.randn(1, text_len, text_dim)

    ort_out, path = _export_and_validate(
        wrapper, (x, t, ctx),
        input_names=["latent_input", "timestep", "text_embeddings"],
        output_names=["noise_prediction"],
        name="dit",
    )

    out = ort_out[0]
    # Output: [1, out_dim, F, H, W] — unpatchify reconstructs full resolution
    expected = (1, out_dim, F_, H, W)
    assert out.shape == expected, f"DiT output {out.shape} != {expected}"
    return path


def test_dit_dynamic_axes_fixed_by_default():
    from wannx.converters.dit import DiTConverter

    converter = DiTConverter()

    axes = converter.dynamic_axes({})
    assert axes == {}

    text_axes = converter.dynamic_axes({"dynamic_text_len": True})
    assert text_axes == {"text_embeddings": {1: "seq_len"}}

    dyn_axes = converter.dynamic_axes({
        "dynamic_text_len": True,
        "dynamic_frames": True,
        "dynamic_resolution": True,
    })
    assert dyn_axes["latent_input"][2] == "latent_frames"
    assert dyn_axes["latent_input"][3] == "latent_height"
    assert dyn_axes["latent_input"][4] == "latent_width"
    assert dyn_axes["noise_prediction"][2] == "latent_frames_out"
    assert dyn_axes["noise_prediction"][3] == "latent_height_out"
    assert dyn_axes["noise_prediction"][4] == "latent_width_out"


def test_quantize_skips_attention_qkv_weights():
    from wannx.quantize import _is_weight_tensor

    assert _is_weight_tensor("model.blocks.0.ffn.0.weight")
    assert not _is_weight_tensor("model.blocks.39.self_attn.q.weight")
    assert not _is_weight_tensor("model.blocks.39.self_attn.k.weight")
    assert not _is_weight_tensor("model.blocks.39.self_attn.v.weight")
    assert not _is_weight_tensor("model.blocks.39.cross_attn.q.weight")
    assert not _is_weight_tensor("model.blocks.39.cross_attn.k.weight")
    assert not _is_weight_tensor("model.blocks.39.cross_attn.v.weight")


@pytest.mark.parametrize(
    "converter_name",
    ["convert_to_native_fp16", "convert_to_native_fp16_streaming"],
)
def test_native_fp16_conversion_removes_weight_casts(converter_name):
    from wannx import fp16 as fp16_mod

    tmp_dir = tempfile.mkdtemp(prefix="wannx_test_native_fp16_")
    input_path = _make_external_matmul_model(tmp_dir)
    output_path = os.path.join(tmp_dir, "matmul_fp16.onnx")

    getattr(fp16_mod, converter_name)(input_path, output_path)

    onnx.checker.check_model(output_path)
    model = onnx.load(output_path)

    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT16
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
    assert any(init.name == "w" and init.data_type == TensorProto.FLOAT16 for init in model.graph.initializer)
    cast_nodes = [n for n in model.graph.node if n.op_type == "Cast"]
    assert cast_nodes == []


def test_native_fp16_streaming_preserves_original_float_casts_and_constant_attrs():
    from wannx.fp16 import convert_to_native_fp16_streaming

    with tempfile.TemporaryDirectory(prefix="wannx_test_native_fp16_streaming_") as tmp:
        input_path = os.path.join(tmp, "input.onnx")
        output_path = os.path.join(tmp, "output.onnx")

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4])

        weight = numpy_helper.from_array(
            np.arange(16, dtype=np.float32).reshape(4, 4),
            name="weight",
        )
        const_bias = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_bias"],
            value=numpy_helper.from_array(
                np.array([[1.0, -1.0, 0.5, -0.5]], dtype=np.float32),
                name="const_bias_value",
            ),
        )
        matmul = helper.make_node("MatMul", inputs=["x", "weight"], outputs=["mm"])
        cast_up = helper.make_node(
            "Cast",
            inputs=["mm"],
            outputs=["mm_fp32"],
            name="cast_up_to_float",
            to=TensorProto.FLOAT,
        )
        add = helper.make_node("Add", inputs=["mm_fp32", "const_bias"], outputs=["sum_fp32"])
        cast_down = helper.make_node(
            "Cast",
            inputs=["sum_fp32"],
            outputs=["y"],
            name="cast_down_to_fp16",
            to=TensorProto.FLOAT16,
        )

        graph = helper.make_graph(
            [const_bias, matmul, cast_up, add, cast_down],
            "native_fp16_streaming_graph",
            [x],
            [y],
            [weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
        model.ir_version = 8
        onnx.save_model(
            model,
            input_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="input.onnx.data",
            size_threshold=0,
        )

        convert_to_native_fp16_streaming(input_path, output_path)
        onnx.checker.check_model(output_path)

        converted = onnx.load(output_path, load_external_data=False)
        assert converted.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        assert converted.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16

        cast_nodes = {node.name: node for node in converted.graph.node if node.op_type == "Cast"}
        cast_up_attr = next(attr for attr in cast_nodes["cast_up_to_float"].attribute if attr.name == "to")
        cast_down_attr = next(attr for attr in cast_nodes["cast_down_to_fp16"].attribute if attr.name == "to")
        assert cast_up_attr.i == TensorProto.FLOAT
        assert cast_down_attr.i == TensorProto.FLOAT16

        constant_nodes = [node for node in converted.graph.node if node.op_type == "Constant"]
        assert len(constant_nodes) == 1
        const_value = next(attr.t for attr in constant_nodes[0].attribute if attr.name == "value")
        assert const_value.data_type == TensorProto.FLOAT
        assert len(const_value.external_data) == 0
        assert const_value.raw_data


def test_native_fp16_streaming_preserves_external_constant_weight_attrs():
    from wannx.fp16 import convert_to_native_fp16_streaming

    with tempfile.TemporaryDirectory(prefix="wannx_test_native_fp16_streaming_extconst_") as tmp:
        input_path = os.path.join(tmp, "input.onnx")
        output_path = os.path.join(tmp, "output.onnx")

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4])

        const_weight = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["weight"],
            value=numpy_helper.from_array(
                np.arange(16, dtype=np.float32).reshape(4, 4),
                name="constant_weight",
            ),
        )
        matmul = helper.make_node("MatMul", inputs=["x", "weight"], outputs=["mm_fp32"])
        cast_down = helper.make_node(
            "Cast",
            inputs=["mm_fp32"],
            outputs=["y"],
            name="cast_down_to_fp16",
            to=TensorProto.FLOAT16,
        )

        graph = helper.make_graph(
            [const_weight, matmul, cast_down],
            "native_fp16_streaming_extconst_graph",
            [x],
            [y],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
        model.ir_version = 8
        onnx.save_model(
            model,
            input_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="input.onnx.data",
            size_threshold=0,
            convert_attribute=True,
        )

        convert_to_native_fp16_streaming(input_path, output_path)
        onnx.checker.check_model(output_path)

        converted = onnx.load(output_path, load_external_data=False)
        constant_nodes = [node for node in converted.graph.node if node.op_type == "Constant"]
        assert len(constant_nodes) == 1
        const_value = next(attr.t for attr in constant_nodes[0].attribute if attr.name == "value")
        assert const_value.data_type == TensorProto.FLOAT16
        assert len(const_value.external_data) > 0
        assert len(const_value.raw_data) == 0


def test_storage_fp16_preserves_external_constant_weight_attrs():
    from wannx.fp16 import convert_to_fp16

    with tempfile.TemporaryDirectory(prefix="wannx_test_storage_fp16_extconst_") as tmp:
        input_path = os.path.join(tmp, "input.onnx")
        output_path = os.path.join(tmp, "output.onnx")

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])

        const_weight = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["weight"],
            value=numpy_helper.from_array(
                np.arange(16, dtype=np.float32).reshape(4, 4),
                name="constant_weight",
            ),
        )
        matmul = helper.make_node("MatMul", inputs=["x", "weight"], outputs=["y"])

        graph = helper.make_graph(
            [const_weight, matmul],
            "storage_fp16_extconst_graph",
            [x],
            [y],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
        model.ir_version = 8
        onnx.save_model(
            model,
            input_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="input.onnx.data",
            size_threshold=0,
            convert_attribute=True,
        )

        convert_to_fp16(input_path, output_path)
        onnx.checker.check_model(output_path)

        converted = onnx.load(output_path, load_external_data=False)
        constant_nodes = [node for node in converted.graph.node if node.op_type == "Constant"]
        assert len(constant_nodes) == 1
        const_value = next(attr.t for attr in constant_nodes[0].attribute if attr.name == "value")
        assert const_value.data_type == TensorProto.FLOAT16
        assert len(const_value.external_data) > 0
        assert len(const_value.raw_data) == 0
        cast_nodes = [node for node in converted.graph.node if node.op_type == "Cast"]
        assert cast_nodes


def test_vae_encoder():
    """VAE Encoder -> ONNX round-trip."""
    from wan.modules.vae import WanVAE_
    from wannx.converters.vae import _ONNXVAEEncoder

    z_dim = 4
    vae_model = WanVAE_(
        dim=8,
        z_dim=z_dim,
        dim_mult=[1, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True],
        dropout=0.0,
    )
    vae_model.eval()

    # Build mock object matching what _ONNXVAEEncoder.__init__ expects:
    #   vae.model = WanVAE_
    #   vae.mean, vae.std = Tensors
    class _MockVAE:
        pass

    mock_vae = _MockVAE()
    mock_vae.model = vae_model
    mock_vae.mean = torch.zeros(z_dim)
    mock_vae.std = torch.ones(z_dim)

    wrapper = _ONNXVAEEncoder(mock_vae)
    wrapper.eval()

    x = torch.randn(1, 3, 5, 16, 16)

    ort_out, path = _export_and_validate(
        wrapper, (x,),
        input_names=["input_video"],
        output_names=["latent"],
        name="vae_encoder",
    )

    out = ort_out[0]
    assert out.shape[0] == 1, f"batch dim: {out.shape}"
    assert out.shape[1] == z_dim, f"channels: {out.shape[1]} != {z_dim}"
    # Spatial dims should be reduced (16 -> 4 with 2 levels of 2x downsample)
    assert out.shape[3] < 16 and out.shape[4] < 16, f"spatial not reduced: {out.shape}"
    return path


def test_vae_decoder():
    """VAE Decoder -> ONNX round-trip."""
    from wan.modules.vae import WanVAE_
    from wannx.converters.vae import _ONNXVAEDecoder

    z_dim = 4
    vae_model = WanVAE_(
        dim=8,
        z_dim=z_dim,
        dim_mult=[1, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True],
        dropout=0.0,
    )
    vae_model.eval()

    class _MockVAE:
        pass

    mock_vae = _MockVAE()
    mock_vae.model = vae_model
    mock_vae.mean = torch.zeros(z_dim)
    mock_vae.std = torch.ones(z_dim)

    wrapper = _ONNXVAEDecoder(mock_vae)
    wrapper.eval()

    z = torch.randn(1, z_dim, 2, 4, 4)

    ort_out, path = _export_and_validate(
        wrapper, (z,),
        input_names=["latent"],
        output_names=["output_video"],
        name="vae_decoder",
    )

    out = ort_out[0]
    assert out.shape[0] == 1, f"batch dim: {out.shape}"
    assert out.shape[1] == 3, f"channels: {out.shape[1]} != 3"
    # Spatial dims should be larger than latent
    assert out.shape[3] > 4 and out.shape[4] > 4, f"spatial not upsampled: {out.shape}"
    return path


def test_vae_wrappers_match_reference_cached_path():
    """Wrapper outputs should match WanVAE_'s cached reference path."""
    from wan.modules.vae import WanVAE_
    from wannx.converters.vae import _ONNXVAEEncoder, _ONNXVAEDecoder

    torch.manual_seed(0)
    z_dim = 4
    vae_model = WanVAE_(
        dim=8,
        z_dim=z_dim,
        dim_mult=[1, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True],
        dropout=0.0,
    )
    vae_model.eval()

    class _MockVAE:
        pass

    mock_vae = _MockVAE()
    mock_vae.model = vae_model
    mock_vae.mean = torch.randn(z_dim)
    mock_vae.std = torch.rand(z_dim) + 0.5

    enc_wrapper = _ONNXVAEEncoder(mock_vae).eval()
    dec_wrapper = _ONNXVAEDecoder(mock_vae).eval()
    scale = [mock_vae.mean, 1.0 / mock_vae.std]

    x = torch.randn(1, 3, 5, 16, 16)
    z = torch.randn(1, z_dim, 2, 4, 4)

    with torch.no_grad():
        ref_latent = vae_model.encode(x, scale)
        wrapped_latent = enc_wrapper(x)
        ref_video = vae_model.decode(z, scale).clamp(-1, 1)
        wrapped_video = dec_wrapper(z)

    assert wrapped_latent.shape == ref_latent.shape
    assert wrapped_video.shape == ref_video.shape
    torch.testing.assert_close(wrapped_latent, ref_latent, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(wrapped_video, ref_video, rtol=1e-5, atol=1e-5)
    return "ok"


def test_t5_encoder():
    """T5 Encoder -> ONNX round-trip."""
    from wan.modules.t5 import T5Encoder
    from wannx.converters.t5 import _ONNXT5Encoder

    dim = 64
    encoder = T5Encoder(
        vocab=256,
        dim=dim,
        dim_attn=64,
        dim_ffn=256,
        num_heads=4,
        num_layers=2,
        num_buckets=8,
    )
    encoder.eval()

    # _ONNXT5Encoder expects obj.model = T5Encoder
    class _MockT5:
        pass

    mock = _MockT5()
    mock.model = encoder

    wrapper = _ONNXT5Encoder(mock)
    wrapper.eval()

    seq_len = 16
    ids = torch.randint(0, 256, (1, seq_len))
    mask = torch.ones(1, seq_len, dtype=torch.long)

    ort_out, path = _export_and_validate(
        wrapper, (ids, mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeddings"],
        name="t5_encoder",
    )

    out = ort_out[0]
    expected = (1, seq_len, dim)
    assert out.shape == expected, f"T5 output {out.shape} != {expected}"
    return path


def test_clip_vision():
    """CLIP VisionTransformer -> ONNX round-trip."""
    from wan.modules.clip import VisionTransformer
    from wannx.converters.clip import _ONNXCLIPVision

    image_size = 32
    patch_size = 8
    dim = 64
    num_layers = 4  # need >= 2 for use_31_block slicing (transformer[:-1])

    vit = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        mlp_ratio=2,
        out_dim=dim,
        num_heads=4,
        num_layers=num_layers,
        pool_type="token",
        pre_norm=True,
        post_norm=False,
        activation="quick_gelu",
    )
    vit.eval()

    wrapper = _ONNXCLIPVision(vit)
    wrapper.eval()

    images = torch.randn(1, 3, image_size, image_size)

    ort_out, path = _export_and_validate(
        wrapper, (images,),
        input_names=["images"],
        output_names=["visual_features"],
        name="clip_vision",
    )

    out = ort_out[0]
    num_patches = (image_size // patch_size) ** 2
    # use_31_block returns before final layer & before head, shape [B, num_patches+1, dim]
    expected = (1, num_patches + 1, dim)
    assert out.shape == expected, f"CLIP output {out.shape} != {expected}"
    return path


def test_xlm_roberta():
    """XLM-RoBERTa -> ONNX round-trip."""
    from wan.modules.xlm_roberta import XLMRoberta
    from wannx.converters.xlm_roberta import _ONNXXLMRoberta

    dim = 64
    seq_len = 16

    roberta = XLMRoberta(
        vocab_size=256,
        max_seq_len=32,
        dim=dim,
        num_heads=4,
        num_layers=2,
    )
    roberta.eval()

    wrapper = _ONNXXLMRoberta(roberta)
    wrapper.eval()

    ids = torch.randint(2, 256, (1, seq_len))  # avoid pad_id=1

    ort_out, path = _export_and_validate(
        wrapper, (ids,),
        input_names=["input_ids"],
        output_names=["hidden_states"],
        name="xlm_roberta",
    )

    out = ort_out[0]
    expected = (1, seq_len, dim)
    assert out.shape == expected, f"XLM-R output {out.shape} != {expected}"
    return path


def test_vace():
    """VaceWanModel -> ONNX round-trip."""
    from wan.modules.vace_model import VaceWanModel
    from wannx.converters.vace import _ONNXVaceModel

    dim = 64
    num_heads = 4
    patch_size = (1, 2, 2)
    in_dim = 8
    out_dim = 8
    text_dim = 64
    text_len = 16

    model = VaceWanModel(
        vace_layers=[0, 1],
        vace_in_dim=in_dim,
        model_type="vace",
        patch_size=patch_size,
        text_len=text_len,
        in_dim=in_dim,
        dim=dim,
        ffn_dim=256,
        freq_dim=32,
        text_dim=text_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_layers=2,
    )
    model.eval()

    wrapper = _ONNXVaceModel(model)
    wrapper.eval()

    F_, H, W = 4, 8, 8
    x = torch.randn(1, in_dim, F_, H, W)
    t = torch.tensor([500.0])  # float32 — flow matching uses continuous timesteps
    ctx = torch.randn(1, text_len, text_dim)
    vace_ctx = torch.randn(1, in_dim, F_, H, W)

    ort_out, path = _export_and_validate(
        wrapper, (x, t, ctx, vace_ctx),
        input_names=["latent_input", "timestep", "text_embeddings", "vace_context"],
        output_names=["noise_prediction"],
        name="vace",
    )

    out = ort_out[0]
    # Output: [1, out_dim, F, H, W] — unpatchify reconstructs full resolution
    expected = (1, out_dim, F_, H, W)
    assert out.shape == expected, f"VACE output {out.shape} != {expected}"
    return path


def test_fp16_conversion_preserves_constant_nodes():
    """FP16 conversion should only rewrite initializers, not Constant attrs."""
    from wannx.fp16 import convert_to_fp16

    with tempfile.TemporaryDirectory(prefix="wannx_test_fp16_") as tmp:
        input_path = os.path.join(tmp, "input.onnx")
        output_path = os.path.join(tmp, "output_fp16.onnx")

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])

        weight = numpy_helper.from_array(
            np.arange(16, dtype=np.float32).reshape(4, 4),
            name="weight",
        )
        const_bias = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_bias"],
            value=numpy_helper.from_array(
                np.array([[1.0, -1.0, 0.5, -0.5]], dtype=np.float32),
                name="const_bias_value",
            ),
        )
        matmul = helper.make_node("MatMul", inputs=["x", "weight"], outputs=["mm"])
        add = helper.make_node("Add", inputs=["mm", "const_bias"], outputs=["y"])

        graph = helper.make_graph([const_bias, matmul, add], "fp16_graph", [x], [y], [weight])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
        model.ir_version = 8
        onnx.save_model(
            model,
            input_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="input.onnx.data",
            size_threshold=0,
        )

        convert_to_fp16(input_path, output_path)
        onnx.checker.check_model(output_path)

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"x": np.ones((1, 4), dtype=np.float32)})[0]
        assert out.shape == (1, 4), f"Unexpected output shape: {out.shape}"
        return output_path


# =========================================================================
# Runner
# =========================================================================

ALL_TESTS = [
    ("DiT", test_dit),
    ("VAE Encoder", test_vae_encoder),
    ("VAE Decoder", test_vae_decoder),
    ("VAE Cached Wrapper Fidelity", test_vae_wrappers_match_reference_cached_path),
    ("T5 Encoder", test_t5_encoder),
    ("CLIP Vision", test_clip_vision),
    ("XLM-RoBERTa", test_xlm_roberta),
    ("VACE", test_vace),
    ("FP16 Converter", test_fp16_conversion_preserves_constant_nodes),
]


def main():
    passed = 0
    failed = 0

    for name, test_fn in ALL_TESTS:
        try:
            path = test_fn()
            print(f"  PASS  {name}  ({path})")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            print()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
