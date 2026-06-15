"""XLM-RoBERTa text encoder -> ONNX converter.

Converts the standalone XLM-RoBERTa model used as the text backbone
inside CLIP.  This module already uses F.scaled_dot_product_attention
so it is directly ONNX-exportable.

XLMRoberta.forward(ids) -> [B, L, 1024]
Internally builds a padding mask from pad_id (=1).
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base import BaseConverter
from ._wan_import import ensure_wan_import_path

logger = logging.getLogger(__name__)


class _ONNXXLMRoberta(nn.Module):

    def __init__(self, roberta: nn.Module):
        super().__init__()
        self.roberta = roberta

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] int64 token ids
        Returns:
            hidden_states: [B, L, 1024]
        """
        return self.roberta(input_ids)


class XLMRobertaConverter(BaseConverter):
    name = "xlm_roberta"

    def load_model(self, checkpoint_dir: str, config: Dict[str, Any]) -> nn.Module:
        ensure_wan_import_path(checkpoint_dir)
        from wan.modules.xlm_roberta import XLMRoberta  # noqa: E402

        device = "cuda" if torch.cuda.is_available() else "cpu"

        roberta = XLMRoberta(
            vocab_size=config.get("vocab_size", 250002),
            max_seq_len=config.get("max_seq_len", 514),
            dim=config.get("dim", 1024),
            num_heads=config.get("num_heads", 16),
            num_layers=config.get("num_layers", 24),
        ).to(device)

        # Try to load weights from a checkpoint
        ckpt_path = config.get("xlm_roberta_checkpoint")
        if ckpt_path and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            roberta.load_state_dict(state, strict=False)
            logger.info("Loaded XLM-RoBERTa weights from %s", ckpt_path)
        else:
            logger.warning(
                "No XLM-RoBERTa checkpoint provided - exporting with random weights. "
                "Provide --xlm-roberta-checkpoint or extract weights from CLIP checkpoint."
            )

        roberta.eval()
        return roberta

    def wrap_for_onnx(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        return _ONNXXLMRoberta(model)

    def dummy_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        B = 1
        L = config.get("max_seq_len", 514)
        vocab = config.get("vocab_size", 250002)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ids = torch.randint(2, vocab, (B, L), device=device)  # avoid pad_id=1
        return (ids,)

    def input_names(self) -> List[str]:
        return ["input_ids"]

    def output_names(self) -> List[str]:
        return ["hidden_states"]

    def dynamic_axes(self, config: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "seq_len"},
        }
