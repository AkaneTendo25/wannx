"""WanVAE -> ONNX converter.

Converts the video autoencoder encoder and decoder paths separately.

For fidelity, the wrappers follow WanVAE_'s reference cached
encode/decode implementations instead of calling the raw
Encoder3d/Decoder3d modules in a single uncached pass.  This keeps
temporal resampling semantics aligned with the original model.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base import BaseConverter
from ._wan_import import ensure_wan_import_path

logger = logging.getLogger(__name__)


class _ONNXVAEEncoder(nn.Module):
    """ONNX wrapper that follows WanVAE_.encode()."""

    def __init__(self, vae):
        super().__init__()
        # vae is a WanVAE instance (not nn.Module), vae.model is WanVAE_
        self.model = vae.model
        self.register_buffer("scale_mean", vae.mean.clone())
        self.register_buffer("scale_inv_std", (1.0 / vae.std).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, T, H, W] float video in ~[-1, 1]
        Returns:
            z: [B, z_dim, T', H', W'] normalised latent
        """
        scale = [self.scale_mean, self.scale_inv_std]
        return self.model.encode(x, scale)


class _ONNXVAEDecoder(nn.Module):
    """ONNX wrapper that follows WanVAE_.decode()."""

    def __init__(self, vae):
        super().__init__()
        self.model = vae.model
        self.register_buffer("scale_mean", vae.mean.clone())
        self.register_buffer("scale_inv_std", (1.0 / vae.std).clone())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """One-shot, NON-cached decode (no per-frame Python loop).

        WanVAE_.decode() iterates `for i in range(z.shape[2])` with a feature
        cache — that loop unrolls to a fixed length during tracing and bakes the
        temporal dimension, so the resulting ONNX is duration-locked. Decoding
        the whole latent block in a single pass (feat_cache=None) is fully
        convolutional, so the temporal axis stays symbolic and the graph
        generalizes to any duration. Output is 4×T_latent frames; the pipeline
        trims the small overshoot to the requested length.

        Args:
            z: [B, z_dim, T', H', W'] normalised latent
        Returns:
            x: [B, 3, T, H, W] reconstructed video clamped to [-1, 1]
        """
        mean = self.scale_mean.view(1, -1, 1, 1, 1)
        inv_std = self.scale_inv_std.view(1, -1, 1, 1, 1)
        zz = z / inv_std + mean
        x = self.model.conv2(zz)
        out = self.model.decoder(x)  # feat_cache=None -> non-cached, one-shot
        return out.clamp(-1, 1)


class VAEEncoderConverter(BaseConverter):
    name = "vae_encoder"

    def _find_vae_checkpoint(self, checkpoint_dir: str) -> str:
        for f in os.listdir(checkpoint_dir):
            if "vae" in f.lower() and f.endswith((".pth", ".pt", ".bin", ".safetensors")):
                return os.path.join(checkpoint_dir, f)
        raise FileNotFoundError(f"No VAE checkpoint found in {checkpoint_dir}")

    def load_model(self, checkpoint_dir: str, config: Dict[str, Any]) -> nn.Module:
        ensure_wan_import_path(checkpoint_dir)
        from wan.modules.vae import WanVAE  # noqa: E402

        vae_pth = self._find_vae_checkpoint(checkpoint_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = WanVAE(vae_pth=vae_pth, device=device)
        return vae  # Note: WanVAE is not nn.Module

    def wrap_for_onnx(self, model, config: Dict[str, Any]) -> nn.Module:
        return _ONNXVAEEncoder(model)

    def dummy_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        B = 1
        T = config.get("frames", 5)  # Must be 1 + 4*k for causal conv alignment
        H = config.get("height", 256)
        W = config.get("width", 256)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return (torch.randn(B, 3, T, H, W, device=device),)

    def input_names(self) -> List[str]:
        return ["input_video"]

    def output_names(self) -> List[str]:
        return ["latent"]

    def dynamic_axes(self, config: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        axes: Dict[str, Dict[int, str]] = {
            "input_video": {0: "batch"},
            "latent": {0: "batch"},
        }
        if config.get("dynamic_resolution", False):
            axes["input_video"][3] = "height"
            axes["input_video"][4] = "width"
            axes["latent"][3] = "latent_height"
            axes["latent"][4] = "latent_width"
        return axes


class VAEDecoderConverter(BaseConverter):
    name = "vae_decoder"

    def _find_vae_checkpoint(self, checkpoint_dir: str) -> str:
        for f in os.listdir(checkpoint_dir):
            if "vae" in f.lower() and f.endswith((".pth", ".pt", ".bin", ".safetensors")):
                return os.path.join(checkpoint_dir, f)
        raise FileNotFoundError(f"No VAE checkpoint found in {checkpoint_dir}")

    def load_model(self, checkpoint_dir: str, config: Dict[str, Any]) -> nn.Module:
        ensure_wan_import_path(checkpoint_dir)
        from wan.modules.vae import WanVAE  # noqa: E402

        vae_pth = self._find_vae_checkpoint(checkpoint_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vae = WanVAE(vae_pth=vae_pth, device=device)
        return vae

    def wrap_for_onnx(self, model, config: Dict[str, Any]) -> nn.Module:
        return _ONNXVAEDecoder(model)

    def dummy_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        B = 1
        z_dim = config.get("z_dim", 16)
        # 5 latent frames is a safe trace shape (the causal conv stack rejects
        # T<3); the temporal axis is made symbolic via dynamic_shapes anyway.
        T = config.get("latent_frames", 5)
        H = config.get("latent_height", 32)
        W = config.get("latent_width", 32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return (torch.randn(B, z_dim, T, H, W, device=device),)

    def input_names(self) -> List[str]:
        return ["latent"]

    def output_names(self) -> List[str]:
        return ["output_video"]

    def dynamic_axes(self, config: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        axes: Dict[str, Dict[int, str]] = {
            "latent": {0: "batch"},
            "output_video": {0: "batch"},
        }
        if config.get("dynamic_frames", False):
            # Temporal axis: required for "any duration" decode. Without this the
            # decoder is locked to the dummy latent-frame count (e.g. 4).
            axes["latent"][2] = "latent_frames"
            axes["output_video"][2] = "frames"
        if config.get("dynamic_resolution", False):
            axes["latent"][3] = "latent_height"
            axes["latent"][4] = "latent_width"
            axes["output_video"][3] = "height"
            axes["output_video"][4] = "width"
        return axes

    def _build_dynamic_shapes(self, config: Dict[str, Any]):
        """Positional dynamic_shapes for torch.onnx.export(dynamo=True).

        Marks the latent temporal/spatial axes symbolic so one graph decodes any
        duration/resolution. The legacy tracer cannot do this for the VAE (the
        conv stack bakes the temporal size), so dynamic VAE export REQUIRES the
        dynamo path.
        """
        from torch.export import Dim

        z_spec: Dict[int, Any] = {}
        if config.get("dynamic_frames"):
            z_spec[2] = Dim("latent_frames", min=1, max=1024)
        if config.get("dynamic_resolution"):
            z_spec[3] = Dim("latent_height", min=1, max=1024)
            z_spec[4] = Dim("latent_width", min=1, max=1024)
        return (z_spec or None,)

    def export(
        self,
        checkpoint_dir: str,
        output_dir: str,
        config: Dict[str, Any],
    ) -> str:
        dynamic = bool(
            config.get("dynamic_frames") or config.get("dynamic_resolution")
        )
        if dynamic and not bool(config.get("fake_mode_export")):
            return self._export_dynamo_dynamic(checkpoint_dir, output_dir, config)
        return super().export(checkpoint_dir, output_dir, config)

    def _export_dynamo_dynamic(
        self,
        checkpoint_dir: str,
        output_dir: str,
        config: Dict[str, Any],
    ) -> str:
        """Single dynamic ONNX (any duration/resolution) via torch.export/dynamo."""
        import contextlib
        import io
        import time

        import onnx

        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, f"{self.name}.onnx")
        for p in (onnx_path, onnx_path + ".data"):
            if os.path.exists(p):
                os.remove(p)

        logger.info("[%s] Loading model for dynamic export ...", self.name)
        model = self.load_model(checkpoint_dir, config)
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
        t0 = time.time()
        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                program = torch.onnx.export(wrapped, dummy, **export_kwargs)
        logger.info("[%s] Dynamo graph traced in %.1fs; saving ...", self.name, time.time() - t0)
        program.save(onnx_path, external_data=True)
        onnx.checker.check_model(onnx_path)
        logger.info("[%s] Saved dynamic ONNX -> %s", self.name, onnx_path)
        return onnx_path
