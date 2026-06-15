"""CLIP vision encoder -> ONNX converter.

Converts the VisionTransformer from CLIPModel.  The VisionTransformer's
internal SelfAttention calls flash_attention(version=2) which is replaced
globally by the patcher.

CLIPModel.visual() does preprocessing (resize, normalize) before calling
self.model.visual(videos, use_31_block=True).  For ONNX we export just
the VisionTransformer.forward() with use_31_block=True (returns features
from layer 31 of 32, as used during WAN inference).

Real attribute names (VisionTransformer):
  - patch_embedding  (nn.Conv2d)
  - cls_embedding    (nn.Parameter)
  - pos_embedding    (nn.Parameter)
  - pre_norm         (LayerNorm or None)
  - transformer      (nn.Sequential of AttentionBlocks)
  - post_norm        (LayerNorm)
  - head             (nn.Parameter or nn.Linear or AttentionPool)

Real AttentionBlock:
  - norm1, norm2     (LayerNorm)
  - attn             (SelfAttention with .to_qkv fused, calls flash_attention)
  - mlp              (nn.Sequential or SwiGLU)
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base import BaseConverter
from ._wan_import import ensure_wan_import_path

logger = logging.getLogger(__name__)


class _ONNXCLIPVision(nn.Module):
    """Wraps VisionTransformer for clean tensor I/O.

    Calls VisionTransformer.forward(x, use_31_block=True) which is what
    CLIPModel.visual() uses during WAN inference.  flash_attention is
    already patched globally before this runs.
    """

    def __init__(self, vision_transformer: nn.Module):
        super().__init__()
        self.vit = vision_transformer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224] normalised (CLIP mean/std)
        Returns:
            features: [B, num_patches+1, vision_dim]  (from block 31)
        """
        return self.vit(images, use_31_block=True)


class CLIPConverter(BaseConverter):
    name = "clip"

    def load_model(self, checkpoint_dir: str, config: Dict[str, Any]):
        ensure_wan_import_path(checkpoint_dir)
        from wan.modules.clip import CLIPModel  # noqa: E402

        clip_pth = None
        tokenizer_path = None
        for f in os.listdir(checkpoint_dir):
            fl = f.lower()
            if "clip" in fl and f.endswith((".pth", ".pt", ".bin", ".safetensors")):
                clip_pth = os.path.join(checkpoint_dir, f)
            if "xlm" in fl and os.path.isdir(os.path.join(checkpoint_dir, f)):
                tokenizer_path = os.path.join(checkpoint_dir, f)

        if clip_pth is None:
            raise FileNotFoundError(f"No CLIP checkpoint found in {checkpoint_dir}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip = CLIPModel(
            dtype=torch.float32,
            device=device,
            checkpoint_path=clip_pth,
            tokenizer_path=tokenizer_path or "xlm-roberta-large",
        )
        return clip

    def wrap_for_onnx(self, model, config: Dict[str, Any]) -> nn.Module:
        # model is CLIPModel; model.model is XLMRobertaCLIP;
        # model.model.visual is VisionTransformer
        vit = model.model.visual
        vit.eval()
        return _ONNXCLIPVision(vit)

    def dummy_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        B = 1
        H = config.get("clip_image_size", 224)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return (torch.randn(B, 3, H, H, device=device),)

    def input_names(self) -> List[str]:
        return ["images"]

    def output_names(self) -> List[str]:
        return ["visual_features"]

    def dynamic_axes(self, config: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        return {
            "images": {0: "batch"},
            "visual_features": {0: "batch"},
        }
