"""ONNX inference pipeline for WAN 2.1 Text-to-Video.

Pipeline: T5 encode -> denoise loop (DiT + CFG) -> VAE decode -> MP4

All inference math uses numpy — no torch dependency at runtime.
"""

import logging
import base64
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort

logger = logging.getLogger(__name__)
_ORT_DLLS_PRELOADED = False
_ORT_DLL_DIR_HANDLES = []
_ORT_PRELOADED_DLLS = []

# Official Wan shared default negative prompt from upstream shared_config.py.
WAN_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def resolve_negative_prompt(negative_prompt: Optional[str]) -> str:
    """Mirror upstream Wan behavior for classifier-free guidance.

    Upstream Wan does not use an empty string as the unconditional prompt:
    when no negative prompt is supplied, it falls back to the shared default
    negative prompt from `wan/configs/shared_config.py`.
    """
    if negative_prompt is None:
        return WAN_DEFAULT_NEGATIVE_PROMPT
    if not str(negative_prompt).strip():
        return WAN_DEFAULT_NEGATIVE_PROMPT
    return str(negative_prompt)


def pad_text_embeddings(embeddings: np.ndarray, text_len: int) -> np.ndarray:
    """Pad or trim T5 embeddings to the fixed Wan text length."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    seq_len = int(embeddings.shape[0])
    if seq_len >= int(text_len):
        return embeddings[: int(text_len)]
    padded = np.zeros((int(text_len), int(embeddings.shape[1])), dtype=np.float32)
    padded[:seq_len] = embeddings
    return padded


# ====================================================================== #
#  Flow-Matching Euler Scheduler (pure numpy)                              #
# ====================================================================== #

class FlowMatchEulerScheduler:
    """WAN-compatible FlowMatch Euler scheduler wrapper.

    We delegate timestep/sigma construction and stepping to diffusers'
    reference implementation instead of maintaining a parallel numpy-only
    approximation. The scheduler overhead is negligible relative to DiT
    inference, and using the reference path avoids silent schedule drift.
    """

    def __init__(
        self,
        num_steps: int = 50,
        shift: float = 3.0,
        num_train_timesteps: int = 1000,
    ):
        import torch
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        self._torch = torch
        self.num_steps = num_steps
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
        self._scheduler.set_timesteps(num_steps, device="cpu")
        self.timesteps = self._scheduler.timesteps.detach().cpu().numpy().astype(np.float32)
        self.sigmas = self._scheduler.sigmas.detach().cpu().numpy().astype(np.float32)

    def scale_noise(self, latents: np.ndarray) -> np.ndarray:
        """Scale initial noise using the reference scheduler implementation."""
        torch = self._torch
        noise = torch.from_numpy(np.ascontiguousarray(latents.astype(np.float32, copy=False)))
        scaled = self._scheduler.scale_noise(noise, self._scheduler.timesteps[:1], noise)
        return scaled.detach().cpu().numpy()

    def step(self, model_output: np.ndarray, sample: np.ndarray,
             step_index: int) -> np.ndarray:
        """One Euler step using diffusers' FlowMatchEulerDiscreteScheduler."""
        torch = self._torch
        prev = self._scheduler.step(
            torch.from_numpy(np.ascontiguousarray(model_output.astype(np.float32, copy=False))),
            self._scheduler.timesteps[step_index],
            torch.from_numpy(np.ascontiguousarray(sample.astype(np.float32, copy=False))),
            return_dict=False,
        )[0]
        return prev.detach().cpu().numpy()


class FlowMatchUniPCScheduler:
    """WAN-compatible UniPC scheduler wrapper using upstream Wan solver."""

    def __init__(
        self,
        num_steps: int = 50,
        shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        import torch
        from .fm_solvers_unipc import FlowUniPCMultistepScheduler

        self._torch = torch
        self.num_steps = num_steps
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self._scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        self._scheduler.set_timesteps(num_steps, device="cpu", shift=shift)
        self.timesteps = self._scheduler.timesteps.detach().cpu().numpy().astype(np.float32)

    def scale_noise(self, latents: np.ndarray) -> np.ndarray:
        # WAN's reference path starts directly from standard normal noise.
        return latents

    def step(self, model_output: np.ndarray, sample: np.ndarray, step_index: int) -> np.ndarray:
        torch = self._torch
        timestep = self._scheduler.timesteps[step_index]
        prev = self._scheduler.step(
            torch.from_numpy(np.ascontiguousarray(model_output.astype(np.float32, copy=False))),
            timestep,
            torch.from_numpy(np.ascontiguousarray(sample.astype(np.float32, copy=False))),
            return_dict=False,
        )[0]
        return prev.detach().cpu().numpy()


class FlowMatchDPMSolverScheduler:
    """WAN-compatible DPM++ scheduler wrapper using upstream Wan solver."""

    def __init__(
        self,
        num_steps: int = 50,
        shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        from .fm_solvers import (
            FlowDPMSolverMultistepScheduler,
            get_sampling_sigmas,
            retrieve_timesteps,
        )

        self.num_steps = num_steps
        self.shift = shift
        self.num_train_timesteps = num_train_timesteps
        self._scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(num_steps, shift)
        timesteps, _ = retrieve_timesteps(
            self._scheduler,
            device="cpu",
            sigmas=sampling_sigmas,
        )
        self.timesteps = timesteps.detach().cpu().numpy().astype(np.float32)

    def scale_noise(self, latents: np.ndarray) -> np.ndarray:
        return latents

    def step(self, model_output: np.ndarray, sample: np.ndarray, step_index: int) -> np.ndarray:
        import torch

        timestep = self._scheduler.timesteps[step_index]
        prev = self._scheduler.step(
            torch.from_numpy(np.ascontiguousarray(model_output.astype(np.float32, copy=False))),
            timestep,
            torch.from_numpy(np.ascontiguousarray(sample.astype(np.float32, copy=False))),
            return_dict=False,
        )[0]
        return prev.detach().cpu().numpy()


def build_scheduler(name: str, num_steps: int, shift: float):
    scheduler_name = str(name).lower()
    if scheduler_name == "euler":
        return FlowMatchEulerScheduler(num_steps=num_steps, shift=shift)
    if scheduler_name == "unipc":
        return FlowMatchUniPCScheduler(num_steps=num_steps, shift=shift)
    if scheduler_name in ("dpm++", "dpmpp"):
        return FlowMatchDPMSolverScheduler(num_steps=num_steps, shift=shift)
    raise ValueError(f"Unsupported scheduler '{name}'. Expected 'unipc', 'dpm++', or 'euler'.")


# ====================================================================== #
#  T5 Tokenizer wrapper                                                     #
# ====================================================================== #

class T5Tokenizer:
    """Thin wrapper around HuggingFace AutoTokenizer for T5.

    Returns numpy int64 arrays for direct ORT consumption.
    """

    def __init__(self, tokenizer_path: str, text_len: int = 512):
        self.text_len = text_len
        self._fast = None
        self.tokenizer = None
        # Prefer the torch-free `tokenizers` Rust lib (tokenizer.json) so the ONNX
        # inference path never imports transformers/torch. IDs match AutoTokenizer
        # exactly (incl. the </s> post-processor).
        tj = os.path.join(tokenizer_path, "tokenizer.json") if os.path.isdir(tokenizer_path) else None
        if tj and os.path.isfile(tj):
            try:
                from tokenizers import Tokenizer
                self._fast = Tokenizer.from_file(tj)
                self._pad_id = 0
            except Exception:
                self._fast = None
        if self._fast is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenize text and return (input_ids, attention_mask) as int64 numpy."""
        if self._fast is not None:
            ids = self._fast.encode(text).ids[: self.text_len]
            mask = [1] * len(ids)
            if len(ids) < self.text_len:
                pad = self.text_len - len(ids)
                ids = ids + [self._pad_id] * pad
                mask = mask + [0] * pad
            return (np.array([ids], dtype=np.int64), np.array([mask], dtype=np.int64))
        enc = self.tokenizer(
            text,
            max_length=self.text_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)          # [1, L]
        attention_mask = enc["attention_mask"].astype(np.int64)  # [1, L]
        return input_ids, attention_mask


# ====================================================================== #
#  ONNX Runtime session wrapper                                            #
# ====================================================================== #

class OnnxModelSession:
    """Lazy-loadable ORT session wrapper with load/unload for memory management."""

    _GRAPH_OPT_LEVELS = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "layout": ort.GraphOptimizationLevel.ORT_ENABLE_LAYOUT,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    def __init__(
        self,
        onnx_path: str,
        device: str = "cuda",
        provider: str = "auto",
        graph_opt_level: str = "basic",
        disable_cpu_fallback: bool = False,
        execution_mode: str = "sequential",
        disable_mem_pattern: bool = False,
        disable_mem_reuse: bool = False,
        disable_cpu_mem_arena: bool = False,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        enable_profiling: bool = False,
        trt_fp16: bool = False,
        trt_engine_cache_dir: Optional[str] = None,
        trt_max_workspace_size: Optional[int] = None,
        trt_builder_optimization_level: Optional[int] = None,
        trt_auxiliary_streams: Optional[int] = None,
        trt_layer_norm_fp32_fallback: bool = False,
        cuda_mem_limit: Optional[int] = None,
    ):
        self.onnx_path = onnx_path
        self.device = device
        self.provider = provider
        self.graph_opt_level = str(graph_opt_level).lower()
        self.disable_cpu_fallback = bool(disable_cpu_fallback)
        self.execution_mode = str(execution_mode).lower()
        self.disable_mem_pattern = bool(disable_mem_pattern)
        self.disable_mem_reuse = bool(disable_mem_reuse)
        self.disable_cpu_mem_arena = bool(disable_cpu_mem_arena)
        self.intra_op_threads = int(intra_op_threads)
        self.inter_op_threads = int(inter_op_threads)
        self.enable_profiling = bool(enable_profiling)
        self.trt_fp16 = bool(trt_fp16)
        self.trt_engine_cache_dir = trt_engine_cache_dir
        self.trt_max_workspace_size = (
            int(trt_max_workspace_size) if trt_max_workspace_size is not None else None
        )
        self.trt_builder_optimization_level = (
            int(trt_builder_optimization_level) if trt_builder_optimization_level is not None else None
        )
        self.trt_auxiliary_streams = (
            int(trt_auxiliary_streams) if trt_auxiliary_streams is not None else None
        )
        self.trt_layer_norm_fp32_fallback = bool(trt_layer_norm_fp32_fallback)
        self.cuda_mem_limit = int(cuda_mem_limit) if cuda_mem_limit is not None else None
        self._session: Optional[ort.InferenceSession] = None

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def load(self):
        global _ORT_DLLS_PRELOADED
        if self._session is not None:
            return
        if not _ORT_DLLS_PRELOADED and os.name == "nt" and self.device == "cuda":
            try:
                _ensure_windows_dll_paths()
                ort.preload_dlls(directory="")
                _ORT_DLLS_PRELOADED = True
                logger.info("Preloaded ONNX Runtime CUDA/cuDNN DLLs from available runtimes")
            except Exception as exc:
                logger.warning("Failed to preload ONNX Runtime CUDA/cuDNN DLLs: %s", exc)
        providers, provider_options = self._get_providers()
        sess_opts = self._build_session_options()
        logger.info("Loading ONNX session: %s  (providers=%s)",
                     os.path.basename(self.onnx_path), providers)
        t0 = time.time()
        self._session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options,
        )
        if self.disable_cpu_fallback:
            self._session.disable_fallback()
        logger.info("  active providers=%s", self._session.get_providers())
        logger.info("  loaded in %.1fs", time.time() - t0)

    def unload(self):
        if self._session is not None:
            del self._session
            self._session = None
            import gc
            gc.collect()
            logger.info("Unloaded ONNX session: %s",
                         os.path.basename(self.onnx_path))

    def _build_session_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = self._GRAPH_OPT_LEVELS.get(
            self.graph_opt_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        if self.execution_mode == "parallel":
            opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = not self.disable_mem_pattern
        opts.enable_mem_reuse = not self.disable_mem_reuse
        opts.enable_cpu_mem_arena = not self.disable_cpu_mem_arena
        if self.intra_op_threads > 0:
            opts.intra_op_num_threads = self.intra_op_threads
        if self.inter_op_threads > 0:
            opts.inter_op_num_threads = self.inter_op_threads
        if self.enable_profiling:
            opts.enable_profiling = True
        return opts

    def _get_providers(self) -> Tuple[List[str], List[Dict[str, str]]]:
        available = ort.get_available_providers()
        provider = str(self.provider).lower()
        use_cuda = self.device == "cuda" and "CUDAExecutionProvider" in available
        use_trt = use_cuda and "TensorrtExecutionProvider" in available

        providers: List[str] = []
        provider_options: List[Dict[str, str]] = []
        # kSameAsRequested allocates exactly what each step needs instead of the
        # default power-of-two growth, which otherwise over-allocates the CUDA
        # arena on a cold large forward (e.g. 81f 14B: 39 GB -> ~21 GB). This is
        # what makes the 4-bit 14B reliably fit a 24 GB GPU at full length.
        cuda_opts: Dict[str, str] = {"arena_extend_strategy": "kSameAsRequested"}
        if self.cuda_mem_limit is not None and self.cuda_mem_limit > 0:
            cuda_opts["gpu_mem_limit"] = str(self.cuda_mem_limit)

        def add(name: str, options: Optional[Dict[str, str]] = None):
            providers.append(name)
            provider_options.append(options or {})

        if provider == "cpu":
            add("CPUExecutionProvider")
        elif provider == "tensorrt":
            if use_trt:
                trt_opts: Dict[str, str] = {}
                if self.trt_engine_cache_dir:
                    os.makedirs(self.trt_engine_cache_dir, exist_ok=True)
                    trt_opts["trt_engine_cache_enable"] = "True"
                    trt_opts["trt_engine_cache_path"] = self.trt_engine_cache_dir
                if self.trt_fp16:
                    trt_opts["trt_fp16_enable"] = "True"
                if self.trt_max_workspace_size is not None and self.trt_max_workspace_size > 0:
                    trt_opts["trt_max_workspace_size"] = str(self.trt_max_workspace_size)
                if self.trt_builder_optimization_level is not None and self.trt_builder_optimization_level >= 0:
                    trt_opts["trt_builder_optimization_level"] = str(self.trt_builder_optimization_level)
                if self.trt_auxiliary_streams is not None and self.trt_auxiliary_streams >= 0:
                    trt_opts["trt_auxiliary_streams"] = str(self.trt_auxiliary_streams)
                if self.trt_layer_norm_fp32_fallback:
                    trt_opts["trt_layer_norm_fp32_fallback"] = "True"
                add("TensorrtExecutionProvider", trt_opts)
                add("CUDAExecutionProvider", cuda_opts)
            elif use_cuda:
                add("CUDAExecutionProvider", cuda_opts)
            else:
                add("CPUExecutionProvider")
        elif provider == "cuda":
            if use_cuda:
                add("CUDAExecutionProvider", cuda_opts)
            else:
                add("CPUExecutionProvider")
        else:
            if use_cuda:
                add("CUDAExecutionProvider", cuda_opts)
            else:
                add("CPUExecutionProvider")

        if not self.disable_cpu_fallback and "CPUExecutionProvider" not in providers:
            add("CPUExecutionProvider")
        return providers, provider_options

    def get_input_names(self) -> List[str]:
        self.load()
        return [inp.name for inp in self._session.get_inputs()]

    def get_input_shapes(self) -> Dict[str, list]:
        self.load()
        return {inp.name: inp.shape for inp in self._session.get_inputs()}

    _ORT_TYPE_TO_NP = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }

    def __call__(self, **feeds: np.ndarray) -> List[np.ndarray]:
        """Run inference. Auto-casts feed dtypes to match model inputs."""
        if self._session is None:
            self.load()
        # Auto-cast feeds to expected dtypes (handles fp16 models transparently)
        input_meta = {inp.name: inp.type for inp in self._session.get_inputs()}
        for name, arr in feeds.items():
            expected_type = input_meta.get(name)
            if expected_type:
                expected_np = self._ORT_TYPE_TO_NP.get(expected_type)
                if expected_np is not None and arr.dtype != expected_np:
                    feeds[name] = arr.astype(expected_np)
        return self._session.run(None, feeds)


class TorchT5Encoder:
    """Lazy torch-backed WAN T5 encoder for practical prompt embedding."""

    def __init__(
        self,
        checkpoint_dir: str,
        tokenizer_path: str,
        text_len: int = 512,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer_path = tokenizer_path
        self.text_len = int(text_len)
        self.device = device
        # umt5-xxl checkpoint is bf16 -> ~11 GB (fp32 would be ~22 GB). T5 runs in
        # its own (sub)process before the DiT, so 11 GB never coexists with the DiT.
        self.dtype = dtype
        self._t5 = None

    def _find_t5_checkpoint(self) -> str:
        for name in os.listdir(self.checkpoint_dir):
            lower = name.lower()
            if "t5" in lower and name.endswith((".pth", ".pt", ".bin", ".safetensors")):
                return os.path.join(self.checkpoint_dir, name)
        raise FileNotFoundError(f"No T5 checkpoint found in {self.checkpoint_dir}")

    def load(self):
        if self._t5 is not None:
            return
        import torch
        from .converters._wan_import import ensure_wan_import_path

        ensure_wan_import_path(self.checkpoint_dir)
        from wan.modules.t5 import T5EncoderModel

        t5_pth = self._find_t5_checkpoint()
        logger.info("Loading torch T5 encoder from %s", os.path.basename(t5_pth))
        t0 = time.time()
        self._t5 = T5EncoderModel(
            text_len=self.text_len,
            dtype=getattr(torch, self.dtype),
            device=torch.device(self.device),
            checkpoint_path=t5_pth,
            tokenizer_path=self.tokenizer_path,
            shard_fn=None,
        )
        logger.info("  loaded in %.1fs", time.time() - t0)

    def unload(self):
        if self._t5 is None:
            return
        del self._t5
        self._t5 = None
        logger.info("Unloaded torch T5 encoder")

    def encode_many(self, texts: List[str]) -> List[np.ndarray]:
        self.load()
        import torch

        with torch.no_grad():
            embeds = self._t5(texts, self.device)
        padded_embeds: List[np.ndarray] = []
        for emb in embeds:
            # .float() first: numpy has no bfloat16, and the DiT context is fp32.
            emb_np = pad_text_embeddings(
                emb.detach().float().cpu().numpy().astype(np.float32, copy=False),
                self.text_len,
            )
            padded_embeds.append(emb_np[None, ...])
        return padded_embeds

    def encode(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]


class IsolatedTorchT5Encoder:
    """Run torch T5 text encoding in a short-lived subprocess.

    This keeps the main ONNX inference process free of torch allocator state,
    which materially reduces DiT session load memory for large models.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        tokenizer_path: str,
        text_len: int = 512,
        device: str = "cuda",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer_path = tokenizer_path
        self.text_len = int(text_len)
        self.device = device

    def load(self):
        return

    def unload(self):
        return

    def encode_many(self, texts: List[str]) -> List[np.ndarray]:
        repo_root = str(Path(__file__).resolve().parents[1])
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        script = r"""
import json
import os
import sys
import numpy as np
from wannx.inference import TorchT5Encoder

request_path, output_path = sys.argv[1], sys.argv[2]
with open(request_path, 'r', encoding='utf-8') as f:
    request = json.load(f)

encoder = TorchT5Encoder(
    checkpoint_dir=request['checkpoint_dir'],
    tokenizer_path=request['tokenizer_path'],
    text_len=int(request['text_len']),
    device=request['device'],
)
embeds = encoder.encode_many(list(request['texts']))
np.savez(output_path, **{f'arr_{i}': arr for i, arr in enumerate(embeds)})
"""
        with tempfile.TemporaryDirectory(prefix="wannx_t5_") as tmpdir:
            request_path = os.path.join(tmpdir, "request.json")
            output_path = os.path.join(tmpdir, "embeds.npz")
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "checkpoint_dir": self.checkpoint_dir,
                        "tokenizer_path": self.tokenizer_path,
                        "text_len": self.text_len,
                        "device": self.device,
                        "texts": list(texts),
                    },
                    f,
                )
            result = subprocess.run(
                [sys.executable, "-c", script, request_path, output_path],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Isolated T5 subprocess failed.\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            with np.load(output_path) as data:
                return [data[f"arr_{i}"].copy() for i in range(len(texts))]

    def encode(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]


class TorchVAEDecoder:
    """Lazy torch-backed WanVAE decoder for fidelity-critical final decode."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self._vae = None

    def _find_vae_checkpoint(self) -> str:
        for name in os.listdir(self.checkpoint_dir):
            if "vae" in name.lower() and name.endswith((".pth", ".pt", ".bin", ".safetensors")):
                return os.path.join(self.checkpoint_dir, name)
        raise FileNotFoundError(f"No VAE checkpoint found in {self.checkpoint_dir}")

    def load(self):
        if self._vae is not None:
            return
        from .converters._wan_import import ensure_wan_import_path

        ensure_wan_import_path(self.checkpoint_dir)
        from wan.modules.vae import WanVAE

        vae_pth = self._find_vae_checkpoint()
        logger.info("Loading torch WanVAE decoder from %s", os.path.basename(vae_pth))
        t0 = time.time()
        self._vae = WanVAE(vae_pth=vae_pth, device=self.device)
        logger.info("  loaded in %.1fs", time.time() - t0)

    def unload(self):
        if self._vae is None:
            return
        del self._vae
        self._vae = None
        logger.info("Unloaded torch WanVAE decoder")

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        self.load()
        import torch

        latent_tensor = torch.from_numpy(
            np.ascontiguousarray(latents.astype(np.float32, copy=False))
        ).to(self.device)
        with torch.no_grad():
            return self._vae.decode([latent_tensor[0]])[0].unsqueeze(0).cpu().numpy()


def _ensure_windows_dll_paths():
    """Add CUDA/cuDNN DLL directories on Windows before ORT provider init."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates: List[Path] = []
    cuda_home = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_home.is_dir():
        for version_dir in sorted(cuda_home.iterdir(), reverse=True):
            bin_dir = version_dir / "bin"
            if bin_dir.is_dir():
                candidates.append(bin_dir)

    site_package_roots: List[Path] = []
    venv_site_packages = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages"
    if venv_site_packages.is_dir():
        site_package_roots.append(venv_site_packages)

    import site

    for p in site.getsitepackages():
        sp = Path(p)
        if sp.is_dir():
            site_package_roots.append(sp)

    user_site = Path(site.getusersitepackages())
    if user_site.is_dir():
        site_package_roots.append(user_site)

    seen_roots = set()
    for site_packages in site_package_roots:
        try:
            resolved = site_packages.resolve()
        except OSError:
            resolved = site_packages
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)

        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            for subdir in sorted(nvidia_root.iterdir()):
                bin_dir = subdir / "bin"
                if bin_dir.is_dir():
                    candidates.append(bin_dir)

        tensorrt_libs = site_packages / "tensorrt_libs"
        if tensorrt_libs.is_dir():
            candidates.append(tensorrt_libs)

    seen_candidates = set()
    path_entries: List[str] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen_candidates:
            continue
        seen_candidates.add(resolved)
        try:
            _ORT_DLL_DIR_HANDLES.append(os.add_dll_directory(str(path)))
            path_entries.append(str(path))
        except OSError:
            continue

    if path_entries:
        current_path = os.environ.get("PATH", "")
        current_parts = current_path.split(os.pathsep) if current_path else []
        prefix = [p for p in path_entries if p not in current_parts]
        if prefix:
            os.environ["PATH"] = os.pathsep.join(prefix + current_parts)

    # TensorRT EP on Windows can still fail to resolve dependencies even after
    # add_dll_directory. Explicit preloading avoids that loader ambiguity.
    tensorrt_dir = Path(site.getusersitepackages()) / "tensorrt_libs"
    if tensorrt_dir.is_dir():
        for dll_name in ("nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll"):
            dll_path = tensorrt_dir / dll_name
            if not dll_path.is_file():
                continue
            try:
                _ORT_PRELOADED_DLLS.append(ctypes.WinDLL(str(dll_path)))
            except OSError:
                continue


# ====================================================================== #
#  Helper functions                                                         #
# ====================================================================== #

ONNX_FILE_NAMES = {
    "t5": "t5_encoder.onnx",
    "dit": "dit.onnx",
    "vae_decoder": "vae_decoder.onnx",
}

ONNX_FILE_NAMES_INT8 = {
    "t5": "t5_encoder_int8.onnx",
    "dit": "dit_int8.onnx",
    "vae_decoder": "vae_decoder.onnx",  # VAE stays fp32
}

ONNX_FILE_NAMES_FP16 = {
    "t5": "t5_encoder_fp16.onnx",
    "dit": "dit_fp16.onnx",
    "vae_decoder": "vae_decoder.onnx",  # VAE stays fp32
}

PROFILE_ROOT = "profiles"
PROFILE_RE = re.compile(r"^f(?P<f>\d+)_h(?P<h>\d+)_w(?P<w>\d+)$")
PROFILE_SHARED_MODEL_ROLES = {"t5"}
PROFILE_SPECIFIC_MODEL_ROLES = {"dit", "vae_decoder"}


def _candidate_names(name_or_names) -> List[str]:
    if not name_or_names:
        return []
    if isinstance(name_or_names, (list, tuple)):
        return list(name_or_names)
    return [name_or_names]


def _profile_name(num_frames: int, height: int, width: int) -> str:
    return f"f{int(num_frames)}_h{int(height)}_w{int(width)}"


def _profile_path(root_dir: str, num_frames: int, height: int, width: int) -> str:
    return os.path.join(root_dir, PROFILE_ROOT, _profile_name(num_frames, height, width))


def resolve_onnx_layout(
    onnx_dir: str,
    num_frames: int,
    height: int,
    width: int,
    *,
    prefer_profile_layout: bool = False,
) -> Tuple[str, Optional[str], bool]:
    """Resolve root/profile dirs for legacy and multi-profile ONNX layouts."""
    root_dir = os.path.abspath(onnx_dir)
    base = os.path.basename(root_dir)
    parent = os.path.basename(os.path.dirname(root_dir))
    if parent == PROFILE_ROOT and PROFILE_RE.match(base):
        profile_dir = root_dir
        root_dir = os.path.dirname(os.path.dirname(root_dir))
        return root_dir, profile_dir, True

    profile_root = os.path.join(root_dir, PROFILE_ROOT)
    has_profile_layout = (
        os.path.isdir(profile_root) or
        os.path.isfile(os.path.join(root_dir, "profiles.json")) or
        bool(prefer_profile_layout)
    )
    if not has_profile_layout:
        return root_dir, None, False
    return root_dir, _profile_path(root_dir, num_frames, height, width), True


def _fp16_variant_candidates(role: str, text_len: Optional[int]) -> List[str]:
    base = ONNX_FILE_NAMES_FP16.get(role)
    names: List[str] = []
    if role == "dit":
        if text_len:
            names.append(f"dit_fp16_seq{int(text_len)}.onnx")
        if text_len in (None, 512):
            names.append("dit_fp16_seq512.onnx")
    names.extend(_candidate_names(base))
    return names


def collect_onnx_models_from_layout(
    root_dir: str,
    profile_dir: Optional[str],
    *,
    prefer_int8: bool = False,
    prefer_fp16: bool = False,
    text_len: Optional[int] = None,
) -> Dict[str, str]:
    root_models = find_onnx_models(
        root_dir,
        prefer_int8=prefer_int8,
        prefer_fp16=prefer_fp16,
        text_len=text_len,
    )
    if not profile_dir:
        return root_models

    merged = dict(root_models)
    if os.path.isdir(profile_dir):
        profile_models = find_onnx_models(
            profile_dir,
            prefer_int8=prefer_int8,
            prefer_fp16=prefer_fp16,
            text_len=text_len,
        )
        for role in PROFILE_SPECIFIC_MODEL_ROLES:
            if role in profile_models:
                merged[role] = profile_models[role]
    return merged


def _write_fixed_text_len_alias(
    onnx_path: str,
    text_len: int,
    *,
    input_name: str = "text_embeddings",
) -> Optional[str]:
    """Write a fixed-text-length alias ONNX header that reuses the same sidecar."""
    if not os.path.isfile(onnx_path):
        return None

    def _iter_external_locations(model_proto: onnx.ModelProto) -> List[str]:
        locations: List[str] = []

        def _add_tensor(tensor: onnx.TensorProto):
            if not tensor.external_data:
                return
            for entry in tensor.external_data:
                if entry.key == "location" and entry.value:
                    locations.append(entry.value)
                    break

        def _walk_graph(graph: onnx.GraphProto):
            for tensor in graph.initializer:
                _add_tensor(tensor)
            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.TENSOR:
                        _add_tensor(attr.t)
                    elif attr.type == onnx.AttributeProto.GRAPH:
                        _walk_graph(attr.g)
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        for subgraph in attr.graphs:
                            _walk_graph(subgraph)

        _walk_graph(model_proto.graph)
        return sorted(set(locations))

    model = onnx.load(onnx_path, load_external_data=False)
    changed = False
    for value_info in model.graph.input:
        if value_info.name != input_name:
            continue
        tensor_type = value_info.type.tensor_type
        if len(tensor_type.shape.dim) < 2:
            continue
        seq_dim = tensor_type.shape.dim[1]
        current = int(seq_dim.dim_value) if seq_dim.HasField("dim_value") else None
        if current == int(text_len):
            return onnx_path
        seq_dim.ClearField("dim_param")
        seq_dim.dim_value = int(text_len)
        changed = True
    if not changed:
        return onnx_path
    stem, ext = os.path.splitext(onnx_path)
    alias_path = f"{stem}_seq{int(text_len)}{ext}"
    src_dir = os.path.dirname(onnx_path)
    dst_dir = os.path.dirname(alias_path)
    with open(alias_path, "wb") as f:
        f.write(model.SerializeToString())
    for rel_location in _iter_external_locations(model):
        src = os.path.join(src_dir, rel_location)
        dst = os.path.join(dst_dir, rel_location)
        if not os.path.isfile(src) or os.path.isfile(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return alias_path


def find_onnx_models(
    onnx_dir: str,
    prefer_int8: bool = False,
    prefer_fp16: bool = False,
    text_len: Optional[int] = None,
) -> Dict[str, str]:
    """Locate required ONNX files in directory. Returns {role: path}.

    Priority: int8 > fp16 > fp32 (when both are requested, int8 wins).
    Falls back to fp32 if preferred variant is missing.
    """
    found = {}
    if prefer_int8:
        priority = [ONNX_FILE_NAMES_INT8, ONNX_FILE_NAMES_FP16, ONNX_FILE_NAMES]
    elif prefer_fp16:
        priority = [ONNX_FILE_NAMES_FP16, ONNX_FILE_NAMES]
    else:
        priority = [ONNX_FILE_NAMES]

    for role in ONNX_FILE_NAMES:
        for names in priority:
            if prefer_fp16 and names is ONNX_FILE_NAMES_FP16:
                candidates = _fp16_variant_candidates(role, text_len)
            else:
                candidates = _candidate_names(names.get(role))
            for filename in candidates:
                path = os.path.join(onnx_dir, filename)
                if os.path.isfile(path):
                    found[role] = path
                    break
            if role in found:
                break
        # Final fallback to base name
        if role not in found:
            path = os.path.join(onnx_dir, ONNX_FILE_NAMES[role])
            if os.path.isfile(path):
                found[role] = path
    return found


def find_tokenizer(checkpoint_dir: str) -> Optional[str]:
    """Find the T5 tokenizer directory inside a WAN checkpoint.

    Looks for google/umt5-xxl or similar tokenizer dirs.
    """
    candidates = [
        os.path.join(checkpoint_dir, "google", "umt5-xxl"),
        os.path.join(checkpoint_dir, "tokenizer"),
    ]
    if os.path.isdir(checkpoint_dir):
        # Scan top-level for dirs containing "tokenizer"
        for name in os.listdir(checkpoint_dir):
            full = os.path.join(checkpoint_dir, name)
            if os.path.isdir(full) and "tokenizer" in name.lower():
                candidates.append(full)
        # Scan google/ subdir
        google_dir = os.path.join(checkpoint_dir, "google")
        if os.path.isdir(google_dir):
            for sub in os.listdir(google_dir):
                candidates.append(os.path.join(google_dir, sub))
    for c in candidates:
        if os.path.isdir(c) and (
            os.path.isfile(os.path.join(c, "tokenizer_config.json")) or
            os.path.isfile(os.path.join(c, "spiece.model"))
        ):
            return c
    return None


def _read_onnx_input_shape(onnx_path: str, input_name: str) -> Optional[List[Union[int, str, None]]]:
    """Read an ONNX input shape without loading external tensor data."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    for value in model.graph.input:
        if value.name != input_name:
            continue
        dims: List[Union[int, str, None]] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        return dims
    return None


def _onnx_input_shape_is_compatible(
    onnx_path: str,
    input_name: str,
    expected_shape: Tuple[int, ...],
) -> bool:
    """Return True when all fixed ONNX dimensions match the requested shape."""
    actual_shape = _read_onnx_input_shape(onnx_path, input_name)
    if actual_shape is None or len(actual_shape) != len(expected_shape):
        return False
    for actual_dim, expected_dim in zip(actual_shape, expected_shape):
        if isinstance(actual_dim, int) and actual_dim > 0 and actual_dim != expected_dim:
            return False
    return True


def _assert_onnx_input_shape_compatible(
    onnx_path: str,
    input_name: str,
    expected_shape: Tuple[int, ...],
    role: str,
):
    """Raise a clear error before inference when a model is shape-locked."""
    actual_shape = _read_onnx_input_shape(onnx_path, input_name)
    if actual_shape is None:
        raise ValueError(
            f"{role} input '{input_name}' was not found in {onnx_path}"
        )
    if not _onnx_input_shape_is_compatible(onnx_path, input_name, expected_shape):
        raise ValueError(
            f"{role} ONNX input '{input_name}' in {onnx_path} expects {actual_shape}, "
            f"which is incompatible with requested shape {list(expected_shape)}. "
            "Re-export this model with dynamic dimensions or matching latent/video sizes."
        )


def save_video(frames: np.ndarray, output_path: str, fps: int = 16):
    """Save [T, H, W, 3] uint8 frames to MP4 or fallback to PNGs.

    Args:
        frames: uint8 array of shape [T, H, W, 3]
        output_path: path ending in .mp4
        fps: frames per second
    """
    logger.info("Saving video: %s  (%d frames, %d fps)",
                 output_path, len(frames), fps)

    # 1) Preferred: imageio v3 + PyAV (in-process, no external ffmpeg binary).
    try:
        import imageio.v3 as iio

        iio.imwrite(output_path, frames, fps=fps, codec="libx264", plugin="pyav")
        logger.info("Video saved (pyav): %s (%.1f MB)",
                     output_path, os.path.getsize(output_path) / (1024**2))
        return
    except Exception as e:
        # ImportError here usually just means PyAV ('av') isn't installed — fall
        # through to the imageio-ffmpeg writer, which bundles its own ffmpeg.
        logger.info("pyav writer unavailable (%s), trying imageio-ffmpeg", e)

    # 2) Fallback: imageio-ffmpeg (ships a static ffmpeg; 'av' not required).
    try:
        import imageio

        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264", format="FFMPEG",
            macro_block_size=1,
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        logger.info("Video saved (imageio-ffmpeg): %s (%.1f MB)",
                     output_path, os.path.getsize(output_path) / (1024**2))
        return
    except Exception as ffmpeg_exc:
        logger.warning("FFMPEG video encoding failed (%s), saving PNGs instead", ffmpeg_exc)

    # 3) Last resort: individual PNGs.
    _save_pngs(frames, output_path)


def _save_pngs(frames: np.ndarray, output_path: str):
    """Fallback: save frames as individual PNG files using PIL."""
    from PIL import Image
    png_dir = output_path.rsplit(".", 1)[0] + "_frames"
    os.makedirs(png_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        png_path = os.path.join(png_dir, f"frame_{i:04d}.png")
        Image.fromarray(frame).save(png_path)
    logger.info("Saved %d PNGs to %s", len(frames), png_dir)


def latents_to_video(raw_output: np.ndarray) -> np.ndarray:
    """Convert VAE decoder output [1,3,T,H,W] float -> [T,H,W,3] uint8."""
    # raw_output is in [-1, 1]
    video = raw_output[0]              # [3, T, H, W]
    video = np.transpose(video, (1, 2, 3, 0))  # [T, H, W, 3]
    video = np.clip((video + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return video


def onnx_vae_decode_tiled(decode, latents, tile_frames=8, tile_overlap=2):
    """Decode latents with an ONNX VAE in memory-bounded temporal tiles.

    ``decode`` is a callable ``latent[1,16,T,H,W] -> video[1,3,4T,8H,8W]`` (an
    ONNX VAE session call). The temporal axis is decoded in chunks of
    ``tile_frames`` latent frames, each padded by ``tile_overlap`` neighbour
    frames of context, and overlapping outputs are cross-faded with a linear
    ramp so chunk borders are continuous. GPU memory then scales with tile size,
    not clip length. Each latent frame maps to 4 video frames, so the output
    length equals a one-shot decode (4 * T_latent).
    """
    T = int(latents.shape[2])
    chunk = max(1, int(tile_frames))
    ov = max(0, int(tile_overlap))
    scale = 4
    if T <= chunk or ov == 0:
        if T <= chunk:
            return decode(latents)
    out_t = scale * T
    acc = None
    wsum = np.zeros(out_t, dtype=np.float32)
    ramp = scale * ov
    i = 0
    while i < T:
        j = min(i + chunk, T)
        lo = max(0, i - ov)
        hi = min(T, j + ov)
        dec = decode(latents[:, :, lo:hi]).astype(np.float32)
        length = dec.shape[2]
        w = np.ones(length, dtype=np.float32)
        if lo > 0 and ramp > 0:
            w[:ramp] = np.linspace(1.0 / (ramp + 1), 1.0, ramp, endpoint=False)
        if hi < T and ramp > 0:
            w[-ramp:] = np.linspace(1.0, 1.0 / (ramp + 1), ramp, endpoint=False)
        if acc is None:
            acc = np.zeros((dec.shape[0], dec.shape[1], out_t, dec.shape[3], dec.shape[4]),
                           dtype=np.float32)
        s = scale * lo
        acc[:, :, s:s + length] += dec * w[None, None, :, None, None]
        wsum[s:s + length] += w
        logger.info("  VAE tile latent[%d:%d] (context %d:%d)", i, j, lo, hi)
        i = j
    acc /= np.maximum(wsum, 1e-6)[None, None, :, None, None]
    return acc


def onnx_vae_decode_streaming(init_session, step_session, latents):
    """Decode latents with the faithful STREAMING ONNX VAE (recurrent).

    Two graphs from tools/export_stream_vae.py reproduce the reference
    WanVAE.decode bit-for-bit, one latent frame at a time (tiny tensors, no
    cuDNN size limit, fits 24 GB at any duration/resolution):

    - init graph: latent 0 (the reference None-cache first-latent path) -> 1
      decoded frame + the initial caches. Latent 0 is NOT temporally doubled,
      matching the reference (and ComfyUI); this is what makes frame 0 correct.
    - step graph: latent i>=1 + caches -> 4 decoded frames + updated caches.

    Output is exactly 1 + 4*(T_latent - 1) frames — no overshoot, no trimming.

    init_session / step_session: onnxruntime.InferenceSession for the two graphs.
    """
    def cidx(name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else None

    step_in = [i.name for i in step_session.get_inputs()]
    step_out = [o.name for o in step_session.get_outputs()]
    init_out = [o.name for o in init_session.get_outputs()]
    step_vid = step_out[0]
    init_vid = init_out[0]
    step_cache_pos = {cidx(nm): p for p, nm in enumerate(step_out) if nm != step_vid}
    init_cache_pos = {cidx(nm): p for p, nm in enumerate(init_out) if nm != init_vid}
    n_caches = len(step_cache_pos)

    T = int(latents.shape[2])
    # --- latent 0: init graph ---
    res = init_session.run(None, {"latent_frame":
                                  np.ascontiguousarray(latents[:, :, 0:1].astype(np.float32))})
    outs = [res[init_out.index(init_vid)]]
    caches = [None] * n_caches
    for k, p in init_cache_pos.items():
        caches[k] = res[p]
    logger.info("  VAE init frame 1/%d", T)
    # --- latents 1..T-1: step graph ---
    for i in range(1, T):
        feed = {}
        for nm in step_in:
            if nm == "latent_frame":
                feed[nm] = np.ascontiguousarray(latents[:, :, i:i + 1].astype(np.float32))
            else:
                feed[nm] = caches[cidx(nm)]
        res = step_session.run(None, feed)
        outs.append(res[step_out.index(step_vid)])
        for k, p in step_cache_pos.items():
            caches[k] = res[p]
        logger.info("  VAE step frame %d/%d", i + 1, T)
    return np.concatenate(outs, axis=2)


def auto_convert_if_needed(
    checkpoint_dir: str,
    onnx_dir: str,
    modules: List[str],
    config: Dict,
):
    """Convert missing ONNX models from checkpoints.

    Re-converts modules that are missing or shape-incompatible.
    """
    from .patcher import apply_patches
    apply_patches()
    from .converter import convert_modules

    existing = find_onnx_models(
        onnx_dir,
        prefer_int8=bool(config.get("int8", False)),
        prefer_fp16=bool(config.get("fp16", False)),
        text_len=int(config.get("text_len", 0) or 0) or None,
    )
    to_convert = [m for m in modules if m not in existing]

    latent_shape = (
        1,
        int(config.get("latent_channels", 16)),
        int(config.get("latent_frames", 4)),
        int(config.get("latent_height", 32)),
        int(config.get("latent_width", 32)),
    )
    expected_inputs = {
        "dit": ("latent_input", latent_shape),
        "vae_decoder": ("latent", latent_shape),
    }
    for module, (input_name, expected_shape) in expected_inputs.items():
        if module not in modules:
            continue
        path = existing.get(module)
        if path and not _onnx_input_shape_is_compatible(path, input_name, expected_shape):
            logger.info(
                "Re-exporting %s because %s is incompatible with requested shape %s",
                module,
                os.path.basename(path),
                list(expected_shape),
            )
            to_convert.append(module)

    if not to_convert:
        logger.info("All required ONNX models found in %s", onnx_dir)
        return

    ordered = []
    seen = set()
    for module in to_convert:
        if module not in seen:
            ordered.append(module)
            seen.add(module)

    results: Dict[str, str] = {}
    if ordered:
        logger.info("Converting modules: %s", ", ".join(ordered))
        results = convert_modules(
            modules=ordered,
            checkpoint_dir=checkpoint_dir,
            output_dir=onnx_dir,
            config=config,
        )
    if config.get("fp16"):
        from .fp16 import convert_to_fp16

        text_len = int(config.get("text_len", 0) or 0)
        for module in modules:
            if module not in ("dit", "t5"):
                continue
            path = results.get(module) or existing.get(module)
            if not path:
                continue
            if path.endswith("_fp16.onnx"):
                if (
                    module == "dit"
                    and not bool(config.get("dynamic_text_len", False))
                    and text_len > 0
                ):
                    _write_fixed_text_len_alias(path, text_len)
                continue
            sidecar = path + ".data"
            if not os.path.isfile(sidecar):
                continue
            fp16_path = path.replace(".onnx", "_fp16.onnx")
            if not os.path.isfile(fp16_path):
                convert_to_fp16(
                    path,
                    fp16_path,
                    native=bool(config.get("native_fp16")),
                    streaming=bool(config.get("native_fp16_streaming")),
                )
            if (
                module == "dit"
                and not bool(config.get("dynamic_text_len", False))
                and text_len > 0
            ):
                _write_fixed_text_len_alias(fp16_path, text_len)


# ====================================================================== #
#  T2V Pipeline                                                             #
# ====================================================================== #

class T2VPipeline:
    """Text-to-Video inference pipeline using ONNX Runtime.

    Runs: T5 encode -> DiT denoise loop (with CFG) -> VAE decode -> video
    """

    def __init__(
        self,
        t5_path: str,
        dit_path: str,
        vae_decoder_path: str,
        tokenizer_path: str,
        checkpoint_dir: Optional[str] = None,
        device: str = "cuda",
        text_len: int = 512,
        keep_models_loaded: bool = False,
        ort_provider: str = "auto",
        ort_graph_opt_level: str = "basic",
        ort_disable_cpu_fallback: bool = False,
        ort_execution_mode: str = "sequential",
        ort_disable_mem_pattern: bool = False,
        ort_disable_mem_reuse: bool = False,
        ort_disable_cpu_mem_arena: bool = False,
        ort_intra_threads: int = 0,
        ort_inter_threads: int = 0,
        ort_enable_profiling: bool = False,
        ort_trt_fp16: bool = False,
        ort_trt_engine_cache_dir: Optional[str] = None,
        ort_trt_max_workspace_size: Optional[int] = None,
        ort_trt_builder_optimization_level: Optional[int] = None,
        ort_trt_auxiliary_streams: Optional[int] = None,
        ort_trt_layer_norm_fp32_fallback: bool = False,
        ort_cuda_mem_limit: Optional[int] = None,
        ort_reload_dit_session_per_call: bool = False,
        ort_isolate_dit_call: bool = False,
        scheduler: str = "unipc",
        text_backend: str = "onnx",
        vae_backend: str = "onnx",
        vae_tile_frames: int = 8,
        vae_tile_overlap: int = 2,
    ):
        sess_kwargs = dict(
            device=device,
            provider=ort_provider,
            graph_opt_level=ort_graph_opt_level,
            disable_cpu_fallback=ort_disable_cpu_fallback,
            execution_mode=ort_execution_mode,
            disable_mem_pattern=ort_disable_mem_pattern,
            disable_mem_reuse=ort_disable_mem_reuse,
            disable_cpu_mem_arena=ort_disable_cpu_mem_arena,
            intra_op_threads=ort_intra_threads,
            inter_op_threads=ort_inter_threads,
            enable_profiling=ort_enable_profiling,
            trt_fp16=ort_trt_fp16,
            trt_engine_cache_dir=ort_trt_engine_cache_dir,
            trt_max_workspace_size=ort_trt_max_workspace_size,
            trt_builder_optimization_level=ort_trt_builder_optimization_level,
            trt_auxiliary_streams=ort_trt_auxiliary_streams,
            trt_layer_norm_fp32_fallback=ort_trt_layer_norm_fp32_fallback,
            cuda_mem_limit=ort_cuda_mem_limit,
        )
        text_backend = str(text_backend).lower()
        if text_backend == "auto":
            text_backend = "torch-isolated" if checkpoint_dir else "onnx"
        if text_backend not in ("onnx", "torch", "torch-isolated"):
            raise ValueError(
                f"Unsupported text_backend '{text_backend}'. Expected 'auto', 'onnx', 'torch', or 'torch-isolated'."
            )
        if text_backend in ("torch", "torch-isolated") and not checkpoint_dir:
            raise ValueError("text_backend='torch' or 'torch-isolated' requires checkpoint_dir")
        if text_backend == "torch":
            self.t5 = TorchT5Encoder(
                checkpoint_dir=checkpoint_dir,
                tokenizer_path=tokenizer_path,
                text_len=text_len,
                device=device,
            )
            self.tokenizer = None
        elif text_backend == "torch-isolated":
            self.t5 = IsolatedTorchT5Encoder(
                checkpoint_dir=checkpoint_dir,
                tokenizer_path=tokenizer_path,
                text_len=text_len,
                device=device,
            )
            self.tokenizer = None
        else:
            self.t5 = OnnxModelSession(t5_path, **sess_kwargs)
            self.tokenizer = T5Tokenizer(tokenizer_path, text_len=text_len)
        self.dit = OnnxModelSession(dit_path, **sess_kwargs)
        backend = str(vae_backend).lower()
        if backend == "auto":
            backend = "torch" if checkpoint_dir else "onnx"
        if backend not in ("onnx", "torch"):
            raise ValueError(f"Unsupported vae_backend '{vae_backend}'. Expected 'auto', 'onnx', or 'torch'.")
        if backend == "torch" and not checkpoint_dir:
            raise ValueError("vae_backend='torch' requires checkpoint_dir")
        if backend == "torch":
            self.vae_decoder = TorchVAEDecoder(checkpoint_dir=checkpoint_dir, device=device)
        else:
            self.vae_decoder = OnnxModelSession(vae_decoder_path, **sess_kwargs)
        self.keep_models_loaded = keep_models_loaded
        self.device = device
        self.scheduler_name = str(scheduler).lower()
        self.text_backend = text_backend
        self.vae_backend = backend
        self.vae_tile_frames = int(vae_tile_frames)
        self.vae_tile_overlap = int(vae_tile_overlap)
        self.ort_reload_dit_session_per_call = bool(ort_reload_dit_session_per_call)
        self.ort_isolate_dit_call = bool(ort_isolate_dit_call)
        self._sess_kwargs = dict(sess_kwargs)
        self._dit_path = dit_path

    def _onnx_vae_decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latents with the ONNX VAE, temporally tiled to bound memory.

        The dynamic ONNX VAE decodes a whole latent block in one pass, which
        materializes every frame's activations at once — too much memory for long
        full-resolution clips on a 24 GB GPU. Instead we decode the temporal axis
        in chunks of ``vae_tile_frames`` latent frames, each padded by
        ``vae_tile_overlap`` neighbour latent frames of context (the non-cached
        graph has no cross-chunk cache). GPU memory is then bounded by the tile
        size, not the clip length.

        A hard crop at chunk borders leaves a visible seam because tiles with
        limited context disagree there, so overlapping output frames are instead
        cross-faded with a linear ramp (accumulate weighted decodes, divide by the
        weight). Each latent frame maps to exactly 4 video frames, so the output
        length matches a one-shot decode (4 * T_latent).
        """
        return onnx_vae_decode_tiled(
            lambda l: self.vae_decoder(latent=l)[0],
            latents,
            tile_frames=self.vae_tile_frames,
            tile_overlap=self.vae_tile_overlap,
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        num_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 5.0,  # WAN T2V-14B reference value (3.0 is the i2v setting)
        seed: int = -1,
        callback=None,
        dit_temporal_chunk_latent: int = 0,
        dit_temporal_overlap_latent: int = 0,
    ) -> np.ndarray:
        """Run full T2V generation.

        Args:
            prompt: text description of the video to generate
            negative_prompt: negative conditioning text
            num_frames: number of output video frames (should be 1+4k)
            height: output video height in pixels
            width: output video width in pixels
            num_steps: number of denoising steps
            guidance_scale: classifier-free guidance scale
            shift: flow-matching shift parameter
            seed: random seed (-1 for random)
            callback: optional fn(step, total, latents) called each step

        Returns:
            frames: [T, H, W, 3] uint8 numpy array
        """
        # Latent shape
        latent_f = (num_frames - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8
        latent_shape = (1, 16, latent_f, latent_h, latent_w)
        logger.info("Latent shape: %s", latent_shape)
        _assert_onnx_input_shape_compatible(
            self.dit.onnx_path,
            "latent_input",
            latent_shape,
            role="DiT",
        )
        if self.vae_backend == "onnx":
            _assert_onnx_input_shape_compatible(
                self.vae_decoder.onnx_path,
                "latent",
                latent_shape,
                role="VAE decoder",
            )

        if guidance_scale > 1.0:
            resolved_negative_prompt = resolve_negative_prompt(negative_prompt)
            if not str(negative_prompt).strip():
                logger.info("Using official Wan default negative prompt for CFG")
        else:
            resolved_negative_prompt = ""

        # Seed
        rng = np.random.default_rng(seed if seed >= 0 else None)

        # --- 1. Encode text ---
        logger.info("Encoding prompt with T5 (%s backend)...", self.text_backend)
        if self.text_backend == "onnx":
            self.t5.load()
            cond_embeds = self._encode_text(prompt)
            uncond_embeds = self._encode_text(resolved_negative_prompt) if guidance_scale > 1.0 else None
            if not self.keep_models_loaded:
                self.t5.unload()
        else:
            texts = [prompt]
            if guidance_scale > 1.0:
                texts.append(resolved_negative_prompt)
            embeddings = self.t5.encode_many(texts)
            cond_embeds = embeddings[0]
            uncond_embeds = embeddings[1] if guidance_scale > 1.0 else None
            if not self.keep_models_loaded:
                self.t5.unload()

        # --- 2. Init scheduler + noise ---
        scheduler = build_scheduler(self.scheduler_name, num_steps=num_steps, shift=shift)
        latents = rng.standard_normal(latent_shape).astype(np.float32)
        latents = scheduler.scale_noise(latents)

        # --- 3. Denoise loop ---
        logger.info("Denoising: %d steps, guidance_scale=%.1f",
                     num_steps, guidance_scale)
        self.dit.load()
        t_start = time.time()
        dit_call_count = 0

        def predict_noise(current_latents, current_timestep, current_text_embeddings):
            nonlocal dit_call_count
            if self.ort_isolate_dit_call:
                return self._predict_noise_isolated(
                    current_latents,
                    current_timestep,
                    current_text_embeddings,
                    temporal_chunk_latent=dit_temporal_chunk_latent,
                    temporal_overlap_latent=dit_temporal_overlap_latent,
                )
            if self.ort_reload_dit_session_per_call and dit_call_count > 0:
                self.dit.unload()
                self.dit.load()
            dit_call_count += 1
            return self._predict_noise_tiled(
                current_latents,
                current_timestep,
                current_text_embeddings,
                temporal_chunk_latent=dit_temporal_chunk_latent,
                temporal_overlap_latent=dit_temporal_overlap_latent,
            )

        for i in range(num_steps):
            t_step = time.time()
            timestep = scheduler.timesteps[i:i+1].astype(np.float32)  # [1]

            # Conditional prediction
            noise_pred_cond = predict_noise(latents, timestep, cond_embeds)

            if uncond_embeds is not None:
                # Unconditional prediction (CFG)
                noise_pred_uncond = predict_noise(latents, timestep, uncond_embeds)
                # CFG combination
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond

            # Euler step
            latents = scheduler.step(noise_pred, latents, i)

            dt = time.time() - t_step
            logger.info("  Step %d/%d  (%.1fs)", i + 1, num_steps, dt)
            if callback:
                callback(i + 1, num_steps, latents)

        dt_total = time.time() - t_start
        logger.info("Denoising done in %.1fs (%.2fs/step)",
                     dt_total, dt_total / num_steps)
        if not self.keep_models_loaded:
            self.dit.unload()

        # --- 4. VAE decode ---
        logger.info("Decoding latents with VAE (%s backend)...", self.vae_backend)
        if self.vae_backend == "torch":
            raw_video = self.vae_decoder(latents)
        else:
            self.vae_decoder.load()
            raw_video = self._onnx_vae_decode(latents)  # [1,3,T,H,W]
        if not self.keep_models_loaded:
            self.vae_decoder.unload()

        # --- 5. Convert to uint8 ---
        frames = latents_to_video(raw_video)
        # Trim to requested frame count. The ONNX VAE decode (streaming and
        # non-cached) lacks the reference decoder's first-latent special case
        # (the real WanVAE.decode emits 1 frame for latent 0 and 4 for each
        # subsequent latent; our graph emits 4 for every latent), so it
        # overshoots by exactly 3 frames *at the front* — those are latent 0
        # decoded with a cold (zero) temporal cache. Drop the leading overshoot,
        # not the tail, so frame 0 is a settled frame aligned with the reference.
        if frames.shape[0] > num_frames:
            logger.info("Trimming %d -> %d frames (dropping leading VAE warmup)",
                         frames.shape[0], num_frames)
            frames = frames[frames.shape[0] - num_frames:]
        logger.info("Output video: %s", frames.shape)
        return frames

    def _predict_noise_tiled(
        self,
        latents: np.ndarray,
        timestep: np.ndarray,
        text_embeddings: np.ndarray,
        *,
        temporal_chunk_latent: int = 0,
        temporal_overlap_latent: int = 0,
    ) -> np.ndarray:
        total_f = int(latents.shape[2])
        chunk_f = int(temporal_chunk_latent or 0)
        if chunk_f <= 0 or chunk_f >= total_f:
            return self.dit(
                latent_input=latents,
                timestep=timestep,
                text_embeddings=text_embeddings,
            )[0]

        overlap = max(0, int(temporal_overlap_latent or 0))
        overlap = min(overlap, chunk_f - 1)
        stride = max(1, chunk_f - overlap)

        logger.info(
            "Using DiT temporal tiling: total_latent_f=%d, chunk=%d, overlap=%d",
            total_f,
            chunk_f,
            overlap,
        )

        acc = np.zeros_like(latents, dtype=np.float32)
        counts = np.zeros((1, 1, total_f, 1, 1), dtype=np.float32)

        start = 0
        while start < total_f:
            end = min(start + chunk_f, total_f)
            pred = self.dit(
                latent_input=latents[:, :, start:end, :, :],
                timestep=timestep,
                text_embeddings=text_embeddings,
            )[0].astype(np.float32, copy=False)
            acc[:, :, start:end, :, :] += pred
            counts[:, :, start:end, :, :] += 1.0
            if end >= total_f:
                break
            start += stride

        return acc / np.maximum(counts, 1.0)

    def _predict_noise_isolated(
        self,
        latents: np.ndarray,
        timestep: np.ndarray,
        text_embeddings: np.ndarray,
        *,
        temporal_chunk_latent: int = 0,
        temporal_overlap_latent: int = 0,
    ) -> np.ndarray:
        repo_root = str(Path(__file__).resolve().parents[1])
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        script = r"""
import base64
import json
import numpy as np
import sys
from wannx.inference import OnnxModelSession, T2VPipeline

request_path, output_path = sys.argv[1], sys.argv[2]
with open(request_path, 'r', encoding='utf-8') as f:
    request = json.load(f)

def _decode_array(payload):
    raw = base64.b64decode(payload['data'])
    arr = np.frombuffer(raw, dtype=np.dtype(payload['dtype']))
    return arr.reshape(payload['shape']).copy()

latents = _decode_array(request['latents'])
timestep = _decode_array(request['timestep'])
text_embeddings = _decode_array(request['text_embeddings'])

sess = OnnxModelSession(request['dit_path'], **request['sess_kwargs'])
sess.load()
try:
    out = T2VPipeline._predict_noise_tiled(
        type('Obj', (), {'dit': sess})(),
        latents,
        timestep,
        text_embeddings,
        temporal_chunk_latent=int(request['temporal_chunk_latent']),
        temporal_overlap_latent=int(request['temporal_overlap_latent']),
    )
finally:
    sess.unload()

np.save(output_path, out)
"""

        def _encode_array(arr: np.ndarray) -> dict:
            contiguous = np.ascontiguousarray(arr)
            return {
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
                "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
            }

        with tempfile.TemporaryDirectory(prefix="wannx_dit_") as tmpdir:
            request_path = os.path.join(tmpdir, "request.json")
            output_path = os.path.join(tmpdir, "pred.npy")
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dit_path": self._dit_path,
                        "sess_kwargs": self._sess_kwargs,
                        "temporal_chunk_latent": int(temporal_chunk_latent or 0),
                        "temporal_overlap_latent": int(temporal_overlap_latent or 0),
                        "latents": _encode_array(latents),
                        "timestep": _encode_array(timestep),
                        "text_embeddings": _encode_array(text_embeddings),
                    },
                    f,
                )
            result = subprocess.run(
                [sys.executable, "-c", script, request_path, output_path],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Isolated DiT subprocess failed.\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            return np.load(output_path).copy()

    def _encode_text(self, text: str) -> np.ndarray:
        """Tokenize and encode text through T5 ONNX."""
        if self.text_backend == "torch":
            return self.t5.encode(text)

        input_ids, attention_mask = self.tokenizer(text)
        embeddings = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]  # [1, L, 4096]
        seq_len = int(attention_mask[0].sum())
        seq_len = max(seq_len, 1)
        trimmed = embeddings[:, :seq_len, :]
        if seq_len != embeddings.shape[1]:
            padded = np.zeros_like(embeddings)
            padded[:, :seq_len, :] = trimmed
            embeddings = padded
        return embeddings
