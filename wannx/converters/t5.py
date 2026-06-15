"""T5 Encoder -> ONNX converter.

Converts the T5EncoderModel text encoder used by WAN for prompt conditioning.
The T5 implementation uses einsum-based attention which is ONNX-compatible.

Note: T5EncoderModel.__init__ defaults device to torch.cuda.current_device()
which crashes on CPU.  We explicitly pass device to avoid this.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base import BaseConverter
from ._wan_import import ensure_wan_import_path

logger = logging.getLogger(__name__)


class _ONNXT5Encoder(nn.Module):
    """ONNX wrapper for the T5 encoder.

    The original T5EncoderModel.__call__ accepts strings and runs the
    tokenizer internally.  For ONNX we accept pre-tokenized tensors.
    The inner self.model is a T5Encoder with forward(ids, mask).
    """

    def __init__(self, t5_encoder_model):
        super().__init__()
        # t5_encoder_model.model is a T5Encoder (nn.Module)
        self.encoder = t5_encoder_model.model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, L] int64
            attention_mask: [B, L] int64 (1=real, 0=pad)
        Returns:
            hidden_states:  [B, L, 4096]
        """
        return self.encoder(input_ids, mask=attention_mask)


class T5Converter(BaseConverter):
    name = "t5_encoder"

    def load_model(self, checkpoint_dir: str, config: Dict[str, Any]):
        ensure_wan_import_path(checkpoint_dir)
        from wan.modules.t5 import T5EncoderModel  # noqa: E402

        # Locate T5 checkpoint
        t5_pth = None
        tokenizer_path = None
        for f in os.listdir(checkpoint_dir):
            fl = f.lower()
            if "t5" in fl and f.endswith((".pth", ".pt", ".bin", ".safetensors")):
                t5_pth = os.path.join(checkpoint_dir, f)
            if os.path.isdir(os.path.join(checkpoint_dir, f)) and "tokenizer" in fl:
                tokenizer_path = os.path.join(checkpoint_dir, f)

        if t5_pth is None:
            raise FileNotFoundError(f"No T5 checkpoint found in {checkpoint_dir}")

        tokenizer_name = config.get("tokenizer", tokenizer_path or "google/umt5-xxl")

        # Explicitly pick device to avoid torch.cuda.current_device() crash
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_len = config.get("text_len", 512)

        enc = T5EncoderModel(
            text_len=text_len,
            dtype=torch.float32,
            device=device,
            checkpoint_path=t5_pth,
            tokenizer_path=tokenizer_name,
            shard_fn=None,
        )
        return enc

    def wrap_for_onnx(self, model, config: Dict[str, Any]) -> nn.Module:
        wrapper = _ONNXT5Encoder(model)
        wrapper.eval()
        return wrapper

    def dummy_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        B = 1
        # Export with a shorter trace length for speed; seq_len remains dynamic.
        export_len = config.get("export_text_len")
        if export_len is None:
            export_len = min(config.get("text_len", 512), 64)
        L = int(export_len)
        L = max(1, L)
        vocab = config.get("vocab_size", 256384)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ids = torch.randint(0, vocab, (B, L), device=device)
        mask = torch.ones(B, L, dtype=torch.long, device=device)
        return (ids, mask)

    def input_names(self) -> List[str]:
        return ["input_ids", "attention_mask"]

    def output_names(self) -> List[str]:
        return ["text_embeddings"]

    def dynamic_axes(self, config: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "text_embeddings": {0: "batch", 1: "seq_len"},
        }

    def export_options(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Constant folding on UMT5-XXL is very expensive and can stall export.
        return {"do_constant_folding": False}
