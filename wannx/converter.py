"""Orchestrator: resolves module names to converters and runs them."""

import logging
from typing import Any, Dict, List

from .converters.dit import DiTConverter
from .converters.vae import VAEEncoderConverter, VAEDecoderConverter
from .converters.t5 import T5Converter
from .converters.clip import CLIPConverter
from .converters.xlm_roberta import XLMRobertaConverter
from .converters.vace import VACEConverter
from .converters.base import BaseConverter

logger = logging.getLogger(__name__)

# Canonical registry  (name -> converter class)
REGISTRY: Dict[str, type] = {
    "dit":          DiTConverter,
    "vae_encoder":  VAEEncoderConverter,
    "vae_decoder":  VAEDecoderConverter,
    "t5":           T5Converter,
    "clip":         CLIPConverter,
    "xlm_roberta":  XLMRobertaConverter,
    "vace":         VACEConverter,
}

ALL_MODULES = list(REGISTRY.keys())


def list_modules() -> List[str]:
    return ALL_MODULES


def get_converter(name: str) -> BaseConverter:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown module '{name}'. Available: {', '.join(ALL_MODULES)}"
        )
    return cls()


def convert_modules(
    modules: List[str],
    checkpoint_dir: str,
    output_dir: str,
    config: Dict[str, Any],
) -> Dict[str, str]:
    """Convert one or more modules.  Returns {module_name: onnx_path}."""
    results: Dict[str, str] = {}
    for name in modules:
        converter = get_converter(name)
        logger.info("=" * 60)
        logger.info("Converting module: %s", name)
        logger.info("=" * 60)
        onnx_path = converter.export(checkpoint_dir, output_dir, config)
        results[name] = onnx_path
    return results
