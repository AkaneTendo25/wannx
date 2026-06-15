"""Base converter interface for all WAN module converters."""

import os
import logging
import time
import shutil
import copy
import io
import contextlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import onnx
from onnx import external_data_helper as onnx_ext

logger = logging.getLogger(__name__)

OPSET_VERSION = 17


def _write_model_proto_preserve_external_data(model_proto: onnx.ModelProto, path: str) -> None:
    """Write ModelProto without losing external-data offset/length metadata.

    `onnx.save/save_model(..., save_as_external_data=False)` rewrites the graph
    but drops `length` fields for externally stored tensors in our large-model
    path. ORT tolerates that, but TensorRT's ONNX parser does not. Writing the
    protobuf bytes directly preserves the exact metadata that we already set via
    `set_external_data(...)`.
    """
    with open(path, "wb") as f:
        f.write(model_proto.SerializeToString())


class BaseConverter(ABC):
    """Abstract base for all module converters."""

    name: str = "base"

    @abstractmethod
    def load_model(self, checkpoint_dir: str, config: Dict[str, Any]) -> nn.Module:
        """Load the original PyTorch model from checkpoint directory."""

    @abstractmethod
    def wrap_for_onnx(self, model: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """Return an ONNX-safe wrapper around the model (no flash_attn, etc.)."""

    @abstractmethod
    def dummy_inputs(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        """Create dummy tensors matching the model's forward() signature."""

    @abstractmethod
    def input_names(self) -> List[str]:
        ...

    @abstractmethod
    def output_names(self) -> List[str]:
        ...

    @abstractmethod
    def dynamic_axes(self, config: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
        ...

    def export_options(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optional per-converter torch.onnx.export kwargs override."""
        return {}

    # ------------------------------------------------------------------ #
    # ONNX export with 2GB workaround                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _onnx_export(model, dummy, path, kwargs):
        """Run torch.onnx.export, bypassing the 2GB protobuf shape-inference
        crash that affects models >2GB (UMT5-XXL, WanModel-14B, etc.).

        The legacy TorchScript exporter calls
        ``_C._jit_pass_onnx_graph_shape_type_inference`` which serialises
        the *entire* parameter dict into a single protobuf message for
        in-memory shape inference.  For models larger than 2 GiB this
        always raises ``RuntimeError``.

        Strategy: attempt a normal export first.  If it fails with the
        known 2-GiB message, retry with that C++ pass replaced by a
        no-op.  ONNX-level shape inference is run afterwards via the
        ``onnx`` library to fill in the missing type/shape annotations.
        """
        dynamic_axes = kwargs.get("dynamic_axes")
        can_try_dynamo = not dynamic_axes
        if can_try_dynamo:
            dynamo_kwargs = copy.deepcopy(kwargs)
            dynamo_kwargs.pop("dynamic_axes", None)
            dynamo_kwargs.update(
                dynamo=True,
                optimize=False,
                verify=False,
                fallback=False,
            )
            try:
                logger.info("Trying torch.onnx.export with dynamo=True")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    torch.onnx.export(model, dummy, path, **dynamo_kwargs)
                return
            except Exception as exc:
                logger.warning(
                    "dynamo=True ONNX export failed; falling back to legacy exporter: %s",
                    exc,
                )

        try:
            torch.onnx.export(model, dummy, path, **kwargs)
        except RuntimeError as exc:
            if "2GiB" not in str(exc) and "2GB" not in str(exc):
                raise
            logger.warning(
                "Model exceeds 2 GiB protobuf limit — retrying with "
                "shape-inference pass disabled."
            )
            import torch._C as _C

            orig = _C._jit_pass_onnx_graph_shape_type_inference
            try:
                _C._jit_pass_onnx_graph_shape_type_inference = (
                    lambda *a, **kw: None
                )
                torch.onnx.export(model, dummy, path, **kwargs)
            finally:
                _C._jit_pass_onnx_graph_shape_type_inference = orig

    # ------------------------------------------------------------------ #
    # Shared export logic                                                  #
    # ------------------------------------------------------------------ #
    def export(
        self,
        checkpoint_dir: str,
        output_dir: str,
        config: Dict[str, Any],
    ) -> str:
        """Full pipeline: load -> wrap -> dummy -> export -> validate.

        Returns the path to the written .onnx file.
        """
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, f"{self.name}.onnx")
        sidecar_name = f"{self.name}.onnx.data"
        sidecar_path = os.path.join(output_dir, sidecar_name)
        tmp_export_dir = os.path.join(output_dir, f".__tmp_{self.name}_export")
        tmp_onnx_path = os.path.join(tmp_export_dir, f"{self.name}.onnx")
        if os.path.isdir(tmp_export_dir):
            shutil.rmtree(tmp_export_dir, ignore_errors=True)
        os.makedirs(tmp_export_dir, exist_ok=True)

        logger.info("[%s] Loading model from %s ...", self.name, checkpoint_dir)
        t0 = time.time()
        model = self.load_model(checkpoint_dir, config)
        logger.info("[%s] Model loaded in %.1fs", self.name, time.time() - t0)

        logger.info("[%s] Wrapping model for ONNX ...", self.name)
        wrapped = self.wrap_for_onnx(model, config)
        wrapped.eval()

        logger.info("[%s] Building dummy inputs ...", self.name)
        dummy = self.dummy_inputs(config)

        logger.info("[%s] Exporting to %s (opset %d) ...", self.name, onnx_path, OPSET_VERSION)
        t0 = time.time()
        export_kwargs = {
            "export_params": True,
            "opset_version": OPSET_VERSION,
            "do_constant_folding": True,
            "input_names": self.input_names(),
            "output_names": self.output_names(),
            "dynamic_axes": self.dynamic_axes(config),
            "external_data": True,
        }
        export_kwargs.update(self.export_options(config))
        try:
            self._onnx_export(wrapped, dummy, tmp_onnx_path, export_kwargs)
            model_proto = onnx.load_model(tmp_onnx_path, load_external_data=False)
            base_dir = os.path.dirname(tmp_onnx_path)

            external_tensors = [
                tensor for tensor in onnx_ext._get_all_tensors(model_proto)
                if onnx_ext.uses_external_data(tensor)
            ]
            unique_locations = []
            seen_locations = set()
            for tensor in external_tensors:
                info = onnx_ext.ExternalDataInfo(tensor)
                if info.location and info.location not in seen_locations:
                    seen_locations.add(info.location)
                    unique_locations.append(info.location)

            fast_sidecar_copy = False
            if len(unique_locations) == 1:
                src_location = unique_locations[0]
                src_path = (
                    src_location if os.path.isabs(src_location)
                    else os.path.join(base_dir, src_location)
                )
                if os.path.isfile(src_path):
                    if os.path.abspath(src_path) != os.path.abspath(sidecar_path):
                        shutil.copyfile(src_path, sidecar_path)
                    for tensor in external_tensors:
                        info = onnx_ext.ExternalDataInfo(tensor)
                        if not tensor.HasField("raw_data"):
                            tensor.raw_data = b""
                        onnx_ext.set_external_data(
                            tensor,
                            location=sidecar_name,
                            offset=info.offset,
                            length=info.length,
                        )
                    fast_sidecar_copy = True

            if not fast_sidecar_copy:
                # Consolidate all per-tensor external data files into one sidecar.
                # torch.onnx.export with external_data=True may create one file per
                # tensor; we merge them into a single <name>.onnx.data file and
                # update each tensor's proto metadata with the correct offset.
                offset = 0
                with open(sidecar_path, "wb") as sidecar_f:
                    for tensor in external_tensors:
                        info = onnx_ext.ExternalDataInfo(tensor)
                        if not info.location:
                            continue
                        src_path = (
                            info.location if os.path.isabs(info.location)
                            else os.path.join(base_dir, info.location)
                        )
                        if not os.path.isfile(src_path):
                            logger.warning(
                                "External data file missing: %s", src_path
                            )
                            continue

                        # Read this tensor's binary data from its individual file
                        with open(src_path, "rb") as src_f:
                            if info.offset is not None:
                                src_f.seek(info.offset)
                            if info.length is not None and info.length > 0:
                                data = src_f.read(info.length)
                            else:
                                data = src_f.read()

                        # Append to the consolidated sidecar
                        sidecar_f.write(data)

                        # Rewrite tensor metadata to point to the merged file.
                        # Ensure raw_data exists — set_external_data requires it,
                        # but the 2GB workaround path may leave it unset.
                        if not tensor.HasField("raw_data"):
                            tensor.raw_data = b""
                        onnx_ext.set_external_data(
                            tensor,
                            location=sidecar_name,
                            offset=offset,
                            length=len(data),
                        )
                        offset += len(data)

            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            _write_model_proto_preserve_external_data(model_proto, onnx_path)
        finally:
            shutil.rmtree(tmp_export_dir, ignore_errors=True)
        export_secs = time.time() - t0
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(
            "[%s] Export done in %.1fs  (%.1f MB)",
            self.name,
            export_secs,
            size_mb,
        )

        logger.info("[%s] Checking ONNX model ...", self.name)
        onnx.checker.check_model(onnx_path)
        logger.info("[%s] ONNX model is valid.", self.name)

        return onnx_path
