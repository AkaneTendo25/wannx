"""VaceWanModel -> ONNX converter.

VaceWanModel.forward() signature:
    forward(x, t, vace_context, context, seq_len, vace_context_scale=1.0,
            clip_fea=None, y=None)

Where x and vace_context are List[Tensor] [C,F,H,W], context is List[Tensor]
[L,C].

Same limitations as DiT:
  - batch_size=1 only
  - spatial dims baked in during tracing
  - RoPE patched to avoid complex ops
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base import BaseConverter
from ._wan_import import ensure_wan_import_path

logger = logging.getLogger(__name__)


class _ONNXVaceModel(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        vace_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:             [1, C_in, F, H, W]
            t:             [1]
            context:       [1, L, text_dim]
            vace_context:  [1, C_in, F, H, W]
        Returns:
            out:           [1, C_out, F', H', W']
        """
        x_list = [x[0]]
        ctx_list = [context[0]]
        vace_list = [vace_context[0]]

        ps = self.model.patch_size
        f_p = x.shape[2] // ps[0]
        h_p = x.shape[3] // ps[1]
        w_p = x.shape[4] // ps[2]
        seq_len = f_p * h_p * w_p

        # VaceWanModel.forward signature:
        # (x, t, vace_context, context, seq_len, vace_context_scale, clip_fea, y)
        out_list = self.model(x_list, t, vace_list, ctx_list, seq_len)
        return out_list[0].unsqueeze(0)


class VACEConverter(BaseConverter):
    name = "vace"

    def load_model(self, checkpoint_dir: str, config: Dict[str, Any]) -> nn.Module:
        ensure_wan_import_path(checkpoint_dir)
        from wan.modules.vace_model import VaceWanModel  # noqa: E402

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VaceWanModel.from_pretrained(checkpoint_dir)
        model.eval().requires_grad_(False)
        model.to(device)
        return model

    def wrap_for_onnx(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        return _ONNXVaceModel(model)

    def dummy_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        C = config.get("latent_channels", 16)
        F_ = config.get("latent_frames", 4)
        H = config.get("latent_height", 32)
        W = config.get("latent_width", 32)
        text_len = config.get("text_len", 512)
        text_dim = config.get("text_dim", 4096)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(1, C, F_, H, W, device=device)
        t = torch.tensor([500.0], device=device)  # float32 — flow matching uses continuous sigma*1000
        ctx = torch.randn(1, text_len, text_dim, device=device)
        vace_ctx = torch.randn(1, C, F_, H, W, device=device)
        return (x, t, ctx, vace_ctx)

    def input_names(self) -> List[str]:
        return ["latent_input", "timestep", "text_embeddings", "vace_context"]

    def output_names(self) -> List[str]:
        return ["noise_prediction"]

    def dynamic_axes(self, config: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        return {
            "text_embeddings": {1: "seq_len"},
            "noise_prediction": {},
        }
