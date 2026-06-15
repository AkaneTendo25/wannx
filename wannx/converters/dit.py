"""DiT (WanModel) -> ONNX converter.

Handles the main diffusion transformer used for denoising in WAN.

Limitations:
  - batch_size=1 only (WanModel uses list-based batching internally).
  - RoPE is patched to avoid complex number ops (torch.view_as_complex /
    torch.polar are not ONNX-exportable).
  - flash_attention is replaced with F.scaled_dot_product_attention.
"""

import contextlib
import io
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import onnx
from safetensors import safe_open

from .base import BaseConverter
from ._wan_import import ensure_wan_import_path

logger = logging.getLogger(__name__)


def _build_attention_translation_table(num_heads: int):
    """Map aten SDPA -> the ``com.microsoft.MultiHeadAttention`` contrib op (one
    fused, dynamic node with a flash/memory-efficient CUDA kernel) instead of the
    default decomposed MatMul->Softmax->MatMul that materializes the full
    [heads, S, S] score matrix and OOMs at long sequences.

    NOTE: the standard ONNX opset-23 ``Attention`` op has NO CUDA kernel in ORT
    1.24 (it silently runs on CPU). The contrib MultiHeadAttention op does, but
    onnxscript needs a schema for it (ORT doesn't register one into onnx.defs),
    and it wants [B, S, heads*head_dim] layout + a num_heads attribute.

    Returns (custom_translation_table, opset_version), or (None, None) if
    onnxscript is unavailable so the caller falls back to the decomposed export.
    """
    try:
        from typing import Optional

        from onnx import defs, TensorProto
        from onnxscript import opset17 as op
        from onnxscript import values
        from onnxscript.function_libs.torch_lib.registration import torch_op
        from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
        from onnxscript.onnx_types import TensorType
    except Exception as exc:  # pragma: no cover - depends on optional onnxscript
        logger.warning("Fused attention unavailable (%s); using decomposed attention.", exc)
        return None, None

    # Register a minimal MHA schema so onnxscript can build the node. ORT validates
    # against its own (real) schema and provides the CUDA kernel at load time.
    try:
        defs.get_schema("MultiHeadAttention", domain="com.microsoft")
    except Exception:
        FP = defs.OpSchema.FormalParameter
        AT = defs.OpSchema.AttrType
        schema = defs.OpSchema(
            "MultiHeadAttention", "com.microsoft", 1,
            inputs=[FP("query", "T"), FP("key", "T"), FP("value", "T")],
            outputs=[FP("output", "T")],
            attributes=[
                defs.OpSchema.Attribute("num_heads", AT.INT, "heads"),
                defs.OpSchema.Attribute("scale", AT.FLOAT, "scale", required=False),
            ],
            type_constraints=[("T", ["tensor(float)", "tensor(float16)"], "")],
        )
        defs.register_schema(schema)

    msft = values.Opset("com.microsoft", 1)
    nh = int(num_heads)

    @torch_op("aten::scaled_dot_product_attention", trace_only=True, private=True)
    def _sdpa_to_mha(
        query: TFloat,
        key: TFloat,
        value: TFloat,
        attn_mask: Optional[TensorType] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ) -> TFloat:
        # q/k/v arrive as [B, heads, S, head_dim]; MHA wants [B, S, heads*head_dim].
        # Run the attention as an explicit fp16 island so ORT uses its flash /
        # memory-efficient CUDA kernel (it is fp16-only) regardless of the model's
        # dtype, then cast back. This keeps the surrounding graph a single dtype
        # (no mixed fp32/fp16 elsewhere) while killing the O(S^2) score buffer.
        q3 = op.Cast(op.Reshape(op.Transpose(query, perm=[0, 2, 1, 3]), [0, 0, -1]), to=TensorProto.FLOAT16)
        k3 = op.Cast(op.Reshape(op.Transpose(key, perm=[0, 2, 1, 3]), [0, 0, -1]), to=TensorProto.FLOAT16)
        v3 = op.Cast(op.Reshape(op.Transpose(value, perm=[0, 2, 1, 3]), [0, 0, -1]), to=TensorProto.FLOAT16)
        if scale is None:
            y = msft.MultiHeadAttention(q3, k3, v3, num_heads=nh)
        else:
            y = msft.MultiHeadAttention(q3, k3, v3, num_heads=nh, scale=scale)
        # [B, S, heads*head_dim] -> [B, heads, S, head_dim], back to input dtype.
        y4 = op.Transpose(op.Reshape(y, [0, 0, nh, -1]), perm=[0, 2, 1, 3])
        return op.CastLike(y4, query)

    # opset 17: ORT 1.24 runs any opset, but Windows ORT (<=1.23.2) lacks CUDA
    # kernels for opset-23 standard ops, so a lower opset keeps it on-GPU there too.
    return {torch.ops.aten.scaled_dot_product_attention.default: _sdpa_to_mha}, 17


class _ONNXWanModel(nn.Module):
    """ONNX wrapper that runs WanModel as a pure tensor graph (batch=1)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.patch_size = tuple(int(v) for v in model.patch_size)
        self.out_dim = int(model.out_dim)
        self.freq_dim = int(model.freq_dim)
        self.dim = int(model.dim)

    @staticmethod
    def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
        assert dim % 2 == 0
        half = dim // 2
        position = position.float()
        device = position.device
        sinusoid = torch.outer(
            position,
            torch.pow(
                torch.tensor(10000.0, device=device, dtype=torch.float32),
                -torch.arange(half, device=device, dtype=torch.float32).div(float(half)),
            ),
        )
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

    def _unpatchify_dynamic(
        self,
        x: torch.Tensor,
        f: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Inverse of patchify for batch=1 with fixed-profile F/H/W."""
        b = x.shape[0]

        pt, ph, pw = self.patch_size
        c = self.out_dim
        x = x.reshape(b, f, h, w, pt, ph, pw, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(b, c, f * pt, h * ph, w * pw)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        Args:
            x:       [1, C_in, F, H, W]  latent video (batch=1 only)
            t:       [1]                  timestep
            context: [1, L, text_dim]     T5 text embeddings
        Returns:
            out:     [1, C_out, F', H', W'] noise prediction
        """
        # Patch embedding: [1, dim, Fp, Hp, Wp]
        xpe = self.model.patch_embedding(x)
        # Keep f/h/w as SYMBOLIC sizes (SymInts) and pass them straight through to
        # rope_apply as a tuple. We deliberately avoid materializing them into a
        # tensor (x.new_tensor([[f,h,w]]) or torch._shape_as_tensor): the former
        # constant-folds the export-time F/H/W into the graph (the old
        # `lifted_tensor_0`), freezing RoPE to one resolution; the latter is not
        # torch.export-friendly. SymInt arithmetic + torch.arange(symint) trace as
        # dynamic ops, so the single graph stays valid at any resolution/duration.
        f = xpe.shape[2]
        h = xpe.shape[3]
        w = xpe.shape[4]
        grid_sizes = (f, h, w)
        seq_lens = f * h * w

        # Flatten tokens: [1, seq, dim]
        x = xpe.flatten(2).transpose(1, 2)

        # Time embedding path.
        # sinusoidal_embedding uses float32 internally for precision;
        # cast result to match model dtype (fp32 or fp16).
        e = self.model.time_embedding(self._sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype))
        e0 = self.model.time_projection(e).unflatten(1, (6, self.dim))

        # Text embedding path.
        context = self.model.text_embedding(context)

        freqs = self.model.freqs
        if freqs.device != x.device:
            freqs = freqs.to(x.device)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=None,
        )
        for block in self.model.blocks:
            x = block(x, **kwargs)

        x = self.model.head(x, e)
        x = self._unpatchify_dynamic(x, f, h, w)
        return x.float()


class DiTConverter(BaseConverter):
    name = "dit"

    def _load_model_config(self, checkpoint_dir: str) -> Dict[str, Any]:
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _apply_local_attention_config(self, model: nn.Module, config: Dict[str, Any]) -> None:
        local_attn_block = int(config.get("local_attn_block_size", 0) or 0)
        if local_attn_block <= 0:
            return
        window = (local_attn_block, local_attn_block)
        model.window_size = window
        for block in model.blocks:
            block.window_size = window
            if hasattr(block, "self_attn"):
                block.self_attn.window_size = window
        logger.info("[dit] Using block-local attention window_size=%s", window)

    def _load_weights_from_single_file(self, model: nn.Module, path: str) -> None:
        """Load weights from a single unsharded safetensors file (e.g. 1.3B)."""
        model_tensors = model.state_dict(keep_vars=True)
        model_keys = set(model_tensors.keys())
        with safe_open(path, framework="pt", device="cpu") as shard:
            ckpt_keys = set(shard.keys())
            allowed_missing = {"freqs"}
            missing = sorted((model_keys - ckpt_keys) - allowed_missing)
            if missing:
                raise RuntimeError(f"Checkpoint is missing required model tensors: {missing[:20]}")
            copied = 0
            with torch.no_grad():
                for key in shard.keys():
                    target = model_tensors.get(key)
                    if target is None:
                        continue
                    tensor = shard.get_tensor(key)
                    if tensor.shape != target.shape:
                        raise RuntimeError(
                            f"Shape mismatch for {key}: checkpoint {tuple(tensor.shape)} "
                            f"!= model {tuple(target.shape)}"
                        )
                    if tensor.dtype != target.dtype:
                        tensor = tensor.to(dtype=target.dtype)
                    target.copy_(tensor)
                    copied += 1
        logger.info("[dit] Copied %d tensors from single-file checkpoint", copied)

    def _load_weights_from_shards(self, model: nn.Module, checkpoint_dir: str) -> None:
        index_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors.index.json")
        if not os.path.isfile(index_path):
            single = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
            if os.path.isfile(single):
                self._load_weights_from_single_file(model, single)
                return
            raise FileNotFoundError(
                f"No shard index or single safetensors found in {checkpoint_dir}"
            )
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        model_tensors = model.state_dict(keep_vars=True)
        model_keys = set(model_tensors.keys())
        ckpt_keys = set(weight_map.keys())

        # `freqs` is constructed analytically by the model code and is not stored
        # in the checkpoint shards.
        allowed_missing = {"freqs"}
        missing = sorted((model_keys - ckpt_keys) - allowed_missing)
        unexpected = sorted(ckpt_keys - model_keys)
        if missing:
            raise RuntimeError(f"Checkpoint is missing required model tensors: {missing[:20]}")
        if unexpected:
            raise RuntimeError(f"Checkpoint contains unexpected tensors: {unexpected[:20]}")

        shard_to_keys: Dict[str, List[str]] = defaultdict(list)
        for key, shard_name in weight_map.items():
            shard_to_keys[shard_name].append(key)

        copied = 0
        with torch.no_grad():
            for shard_name, keys in sorted(shard_to_keys.items()):
                shard_path = os.path.join(checkpoint_dir, shard_name)
                logger.info("[dit] Loading shard %s (%d tensors) ...", shard_name, len(keys))
                with safe_open(shard_path, framework="pt", device="cpu") as shard:
                    for key in keys:
                        target = model_tensors[key]
                        tensor = shard.get_tensor(key)
                        if tensor.shape != target.shape:
                            raise RuntimeError(
                                f"Shape mismatch for {key}: checkpoint {tuple(tensor.shape)} "
                                f"!= model {tuple(target.shape)}"
                            )
                        if tensor.dtype != target.dtype:
                            tensor = tensor.to(dtype=target.dtype)
                        target.copy_(tensor)
                        copied += 1
        logger.info("[dit] Copied %d tensors from checkpoint shards", copied)

    def _fake_mode_dummy_inputs(
        self,
        config: Dict[str, Any],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, ...]:
        c = int(config.get("latent_channels", 16))
        f_ = int(config.get("latent_frames", 4))
        h = int(config.get("latent_height", 32))
        w = int(config.get("latent_width", 32))
        text_len = int(config.get("text_len", 512))
        text_dim = int(config.get("text_dim", 4096))
        x = torch.randn(1, c, f_, h, w, dtype=dtype)
        t = torch.tensor([500.0], dtype=torch.float32)
        ctx = torch.randn(1, text_len, text_dim, dtype=dtype)
        return x, t, ctx

    def _compute_freqs_initializer(self, model_config: Dict[str, Any]) -> torch.Tensor:
        dim = int(model_config["dim"])
        num_heads = int(model_config["num_heads"])
        d = dim // num_heads

        def rope_params(max_seq_len: int, rope_dim: int, theta: int = 10000) -> torch.Tensor:
            assert rope_dim % 2 == 0
            freqs = torch.outer(
                torch.arange(max_seq_len),
                1.0 / torch.pow(
                    torch.tensor(theta, dtype=torch.float64),
                    torch.arange(0, rope_dim, 2, dtype=torch.float64).div(rope_dim),
                ),
            )
            return freqs.float()

        return torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

    def _fake_mode_extra_initializers(
        self,
        config: Dict[str, Any],
        model_config: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        # NOTE: grid_sizes is no longer constant-folded (forward derives it from a
        # Shape op), so the old `lifted_tensor_0` initializer is not emitted.
        return {
            "model.freqs": self._compute_freqs_initializer(model_config),
            "lifted_tensor_1": torch.tensor(10000.0, dtype=torch.float32),
        }

    def _export_fake_mode_graph(
        self,
        checkpoint_dir: str,
        output_dir: str,
        config: Dict[str, Any],
    ) -> str:
        ensure_wan_import_path(checkpoint_dir)
        from wan.modules.model import WanModel  # noqa: E402

        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, f"{self.name}.onnx")
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        sidecar_path = onnx_path + ".data"
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)

        model_config = self._load_model_config(checkpoint_dir)
        logger.info("[%s] Building fake-mode export graph ...", self.name)
        t0 = time.time()
        with torch.onnx.enable_fake_mode():
            model = WanModel.from_config(model_config)
            self._apply_local_attention_config(model, config)
            wrapped = _ONNXWanModel(model).eval()
            dummy = self._fake_mode_dummy_inputs(config, dtype=torch.float32)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                program = torch.onnx.export(
                    wrapped,
                    dummy,
                    input_names=self.input_names(),
                    output_names=self.output_names(),
                    dynamo=True,
                    optimize=False,
                    verify=False,
                    fallback=False,
                )
        logger.info("[%s] Fake-mode graph exported in %.1fs", self.name, time.time() - t0)

        index_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors.index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            weight_index = json.load(f)["weight_map"]

        shard_to_keys: Dict[str, List[str]] = defaultdict(list)
        for key, shard_name in weight_index.items():
            shard_to_keys[shard_name].append(key)

        expected_initializers = {f"model.{key}" for key in weight_index}
        extra_initializers = self._fake_mode_extra_initializers(config, model_config)
        unknown = sorted(
            set(program.model.graph.initializers.keys())
            - expected_initializers
            - set(extra_initializers.keys())
        )
        if unknown:
            raise RuntimeError(
                f"Unexpected fake-mode initializers that are not covered by checkpoint or extras: {unknown}"
            )

        logger.info("[%s] Applying shard weights into ONNX graph ...", self.name)
        for shard_name, keys in sorted(shard_to_keys.items()):
            shard_path = os.path.join(checkpoint_dir, shard_name)
            logger.info("[%s]   shard %s (%d tensors)", self.name, shard_name, len(keys))
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for key in keys:
                    tensor = shard.get_tensor(key)
                    if tensor.is_floating_point() and tensor.dtype != torch.float32:
                        tensor = tensor.to(torch.float32)
                    program.apply_weights({f"model.{key}": tensor})

        program.apply_weights(extra_initializers)

        logger.info("[%s] Saving fake-mode ONNX to %s ...", self.name, onnx_path)
        t0 = time.time()
        program.save(onnx_path, external_data=True)
        logger.info(
            "[%s] Saved fake-mode ONNX in %.1fs (%.1f MB + %.1f GB data)",
            self.name,
            time.time() - t0,
            os.path.getsize(onnx_path) / (1024 * 1024),
            os.path.getsize(sidecar_path) / (1024**3) if os.path.isfile(sidecar_path) else 0.0,
        )
        onnx.checker.check_model(onnx_path)
        return onnx_path

    def _maybe_fp16(self, fp32_path: str, output_dir: str, config: Dict[str, Any]) -> str:
        if not bool(config.get("fp16")):
            return fp32_path
        from ..fp16 import convert_to_fp16

        fp16_path = os.path.join(output_dir, f"{self.name}_fp16.onnx")
        if os.path.exists(fp16_path):
            os.remove(fp16_path)
        sidecar_path = fp16_path + ".data"
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
        logger.info("[%s] Converting ONNX to fp16 ...", self.name)
        convert_to_fp16(
            fp32_path,
            fp16_path,
            native=bool(config.get("native_fp16")),
            streaming=bool(config.get("native_fp16_streaming")),
        )
        return fp16_path

    def _build_dynamic_shapes(self, config: Dict[str, Any]):
        """dynamic_shapes positional tuple for torch.onnx.export(dynamo=True).

        Ordered to match _ONNXWanModel.forward(x, t, context). Latent height/width
        use 2*Dim so the stride-2 patch conv divides cleanly; frames pass through
        (patch_size temporal == 1). text_len up to 4096; RoPE table covers 1024
        positions per spatial/temporal axis.
        """
        from torch.export import Dim

        x_spec: Dict[int, Any] = {}
        if config.get("dynamic_frames"):
            x_spec[2] = Dim("frames", min=1, max=1024)
        if config.get("dynamic_resolution"):
            x_spec[3] = 2 * Dim("lat_h_half", min=1, max=1024)
            x_spec[4] = 2 * Dim("lat_w_half", min=1, max=1024)
        ctx_spec: Dict[int, Any] = {}
        if config.get("dynamic_text_len"):
            ctx_spec[1] = Dim("text_len", min=1, max=4096)
        return (x_spec or None, None, ctx_spec or None)

    def _export_dynamo_dynamic(
        self,
        checkpoint_dir: str,
        output_dir: str,
        config: Dict[str, Any],
    ) -> str:
        """Export the DiT to a SINGLE dynamic ONNX (any F/H/W/text-len) via the
        torch.export/dynamo path with real weights and dynamic_shapes."""
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, f"{self.name}.onnx")
        for p in (onnx_path, onnx_path + ".data"):
            if os.path.exists(p):
                os.remove(p)

        logger.info("[%s] Loading model (real weights) for dynamic export ...", self.name)
        t0 = time.time()
        model = self.load_model(checkpoint_dir, config)
        logger.info("[%s] Model loaded in %.1fs", self.name, time.time() - t0)
        wrapped = self.wrap_for_onnx(model, config).eval()
        dummy = self.dummy_inputs(config)
        dynamic_shapes = self._build_dynamic_shapes(config)
        logger.info("[%s] dynamic_shapes=%s", self.name, dynamic_shapes)

        export_kwargs = dict(
            input_names=self.input_names(),
            output_names=self.output_names(),
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            optimize=bool(config.get("optimize_graph", False)),
            verify=False,
            fallback=False,
        )
        if config.get("fused_attention", True):
            num_heads = int(self._load_model_config(checkpoint_dir)["num_heads"])
            table, opset = _build_attention_translation_table(num_heads)
            if table is not None:
                export_kwargs["custom_translation_table"] = table
                export_kwargs["opset_version"] = opset
                logger.info(
                    "[%s] Using fused com.microsoft.MultiHeadAttention (num_heads=%d, opset %d)",
                    self.name, num_heads, opset,
                )

        t0 = time.time()
        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                program = torch.onnx.export(wrapped, dummy, **export_kwargs)
        logger.info("[%s] Dynamo graph traced in %.1fs; saving ...", self.name, time.time() - t0)
        t0 = time.time()
        program.save(onnx_path, external_data=True)
        sidecar = onnx_path + ".data"
        logger.info(
            "[%s] Saved dynamic ONNX in %.1fs (%.1f MB + %.1f GB data)",
            self.name,
            time.time() - t0,
            os.path.getsize(onnx_path) / (1024 * 1024),
            os.path.getsize(sidecar) / (1024**3) if os.path.isfile(sidecar) else 0.0,
        )
        onnx.checker.check_model(onnx_path)
        return onnx_path

    def export(
        self,
        checkpoint_dir: str,
        output_dir: str,
        config: Dict[str, Any],
    ) -> str:
        dynamic = bool(
            config.get("dynamic_frames")
            or config.get("dynamic_resolution")
            or config.get("dynamic_text_len")
        )
        # Preferred path for "any resolution / any duration": one dynamic graph.
        if dynamic and not bool(config.get("fake_mode_export")):
            out_path = self._export_dynamo_dynamic(checkpoint_dir, output_dir, config)
            native = bool(config.get("native_fp16") or config.get("native_fp16_streaming"))
            if config.get("fp16") and native:
                # load_model already produced an fp16 model; the export is fp16.
                return out_path
            return self._maybe_fp16(out_path, output_dir, config)
        if bool(config.get("fake_mode_export")):
            fp32_path = self._export_fake_mode_graph(checkpoint_dir, output_dir, config)
            return self._maybe_fp16(fp32_path, output_dir, config)
        return super().export(checkpoint_dir, output_dir, config)

    def load_model(self, checkpoint_dir: str, config: Dict[str, Any]) -> nn.Module:
        ensure_wan_import_path(checkpoint_dir)
        from wan.modules.model import WanModel  # noqa: E402

        use_native_fp16 = bool(config.get("native_fp16") or config.get("native_fp16_streaming"))
        model_config = self._load_model_config(checkpoint_dir)
        model = WanModel.from_config(model_config)
        self._apply_local_attention_config(model, config)
        model.float().eval().requires_grad_(False)
        self._load_weights_from_shards(model, checkpoint_dir)
        if config.get("fp16") and use_native_fp16:
            model.half()

        # Estimate model size; fall back to CPU if it won't fit in VRAM.
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0]
            if free_vram > param_bytes * 1.5:
                device = torch.device("cuda")
            else:
                logger.warning(
                    "Model is %.1f GB but only %.1f GB VRAM free — using CPU",
                    param_bytes / 2**30, free_vram / 2**30,
                )
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

        self._device = device
        self._input_dtype = next(model.parameters()).dtype
        model.to(device)
        return model

    def wrap_for_onnx(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        return _ONNXWanModel(model)

    def dummy_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        C = config.get("latent_channels", 16)
        F_ = config.get("latent_frames", 4)
        H = config.get("latent_height", 32)
        W = config.get("latent_width", 32)
        text_len = config.get("text_len", 512)
        text_dim = config.get("text_dim", 4096)

        device = getattr(self, "_device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = getattr(self, "_input_dtype", torch.float32)
        x = torch.randn(1, C, F_, H, W, device=device, dtype=dtype)
        t = torch.tensor([500.0], device=device)  # float32 — flow matching uses continuous sigma*1000
        ctx = torch.randn(1, text_len, text_dim, device=device, dtype=dtype)
        return (x, t, ctx)

    def input_names(self) -> List[str]:
        return ["latent_input", "timestep", "text_embeddings"]

    def output_names(self) -> List[str]:
        return ["noise_prediction"]

    def dynamic_axes(self, config: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        axes: Dict[str, Dict[int, str]] = {}
        if config.get("dynamic_text_len", False):
            axes["text_embeddings"] = {1: "seq_len"}
        if config.get("dynamic_frames", False):
            axes["latent_input"] = {2: "latent_frames"}
            axes["noise_prediction"] = {2: "latent_frames_out"}
        if config.get("dynamic_resolution", False):
            axes.setdefault("latent_input", {})
            axes.setdefault("noise_prediction", {})
            axes["latent_input"][3] = "latent_height"
            axes["latent_input"][4] = "latent_width"
            axes["noise_prediction"][3] = "latent_height_out"
            axes["noise_prediction"][4] = "latent_width_out"
        return axes

    def export_options(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Constant folding is very expensive for large DiT graphs.
        return {"do_constant_folding": False}
