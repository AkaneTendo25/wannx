"""wannx CLI entry point.

Usage examples:

    # Convert all modules
    wannx convert --checkpoint-dir ./models/wan2.1/wan --output-dir ./onnx_out --modules all

    # Convert specific modules
    wannx convert -c ./models/wan2.1/wan -o ./onnx_out -m dit vae_encoder vae_decoder t5

    # List available modules
    wannx list

    # Validate an exported ONNX model
    wannx validate ./onnx_out/dit.onnx

    # Validate all ONNX files in a directory
    wannx validate ./onnx_out/

    # Run T2V inference from checkpoints (auto-converts to ONNX)
    wannx infer --prompt "a cat..." --checkpoint-dir /path/to/Wan2.1-T2V-14B -o output.mp4

    # Run T2V inference from pre-exported ONNX models
    wannx infer --prompt "a cat..." --onnx-dir ./onnx_out --tokenizer ./tokenizer -o output.mp4
"""

import argparse
import logging
import os
import sys
import json
from typing import Any, Dict, List

from . import __version__


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)
    # Quiet noisy libs
    for lib in ("onnxruntime", "transformers", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def _load_config_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")
    return data


def _command_config(cfg: Dict[str, Any], command: str) -> Dict[str, Any]:
    # Support either top-level keys or per-command sections:
    #   {"checkpoint_dir": "..."} OR {"convert": {...}, "infer": {...}}
    section = cfg.get(command)
    if isinstance(section, dict):
        return section
    return cfg


def _checkpoint_hint(checkpoint_dir: str | None) -> str:
    if not checkpoint_dir:
        return ""
    return os.path.basename(os.path.normpath(checkpoint_dir)).lower()


def _default_guidance_scale(checkpoint_dir: str | None) -> float:
    hint = _checkpoint_hint(checkpoint_dir)
    if "1.3b" in hint or "1_3b" in hint:
        # Upstream Wan2.1-T2V-1.3B recommendation.
        return 6.0
    return 5.0


def _default_flow_shift(width: int, height: int, checkpoint_dir: str | None = None) -> float:
    hint = _checkpoint_hint(checkpoint_dir)
    if "1.3b" in hint or "1_3b" in hint:
        # Upstream Wan2.1-T2V-1.3B recommendation.
        return 8.0
    # Wan2.1-T2V-14B reference: shift=5.0 at all resolutions. (The 3.0-at-480p
    # rule from upstream applies ONLY to i2v, not t2v — using it here was the
    # cause of washed-out/low-quality t2v output.)
    return 5.0


def _warn_checkpoint_resolution(checkpoint_dir: str | None, width: int, height: int):
    hint = _checkpoint_hint(checkpoint_dir)
    if "1.3b" in hint or "1_3b" in hint:
        if (width, height) != (832, 480):
            logging.getLogger(__name__).warning(
                "Wan2.1-T2V-1.3B is tuned for 832x480. Requested %dx%d may produce degraded output.",
                width,
                height,
            )


# ====================================================================== #
#  Sub-commands                                                            #
# ====================================================================== #

def cmd_list(args):
    from .converter import list_modules
    print("Available WAN modules for ONNX conversion:\n")
    for name in list_modules():
        print(f"  - {name}")
    print("\nUse 'all' to convert every module at once.")


def cmd_convert(args):
    # Apply runtime patches before importing any WAN code
    from .patcher import apply_patches
    apply_patches()

    from .converter import convert_modules, list_modules

    file_cfg: Dict[str, Any] = {}
    if args.config:
        try:
            file_cfg = _command_config(_load_config_file(args.config), "convert")
        except Exception as exc:
            print(f"ERROR: Failed to load config '{args.config}': {exc}")
            sys.exit(1)

    def pick(name: str, default=None):
        v = getattr(args, name, None)
        if v is not None:
            return v
        return file_cfg.get(name, default)

    modules = pick("modules", ["all"])
    if isinstance(modules, str):
        modules = [modules]
    if not modules:
        modules = ["all"]

    if "all" in modules:
        modules = list_modules()

    checkpoint_dir = pick("checkpoint_dir")
    output_dir = pick("output_dir")
    if not checkpoint_dir or not output_dir:
        print("ERROR: Provide checkpoint/output via CLI or --config")
        sys.exit(1)

    text_len = int(pick("text_len", 512))
    export_text_len = pick("export_text_len", min(text_len, 64))
    if export_text_len is not None:
        export_text_len = int(export_text_len)

    config = {
        "batch_size": pick("batch_size", 1),
        "dynamic_resolution": pick("dynamic_resolution", False),
        "dynamic_frames": pick("dynamic_frames", False),
        "dynamic_text_len": pick("dynamic_text_len", False),
        "text_len": text_len,
        "export_text_len": export_text_len,
        "text_dim": pick("text_dim", 4096),
        "latent_channels": pick("latent_channels", 16),
        "latent_frames": pick("latent_frames", 4),
        "latent_height": pick("latent_height", 32),
        "latent_width": pick("latent_width", 32),
        "frames": pick("frames", 8),
        "height": pick("height", 256),
        "width": pick("width", 256),
        "xlm_roberta_checkpoint": pick("xlm_roberta_checkpoint", None),
        "local_attn_block_size": pick("local_attn_block_size", 0),
        "fp16": pick("fp16", False),
        "fake_mode_export": pick("fake_mode_export", False),
        "native_fp16": pick("native_fp16", False),
        "native_fp16_streaming": pick("native_fp16_streaming", False),
        "optimize_graph": pick("optimize_graph", False),
    }

    print(f"wannx v{__version__}")
    print(f"Checkpoint dir : {checkpoint_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Modules        : {', '.join(modules)}")
    print()

    results = convert_modules(
        modules=modules,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        config=config,
    )

    print("\n" + "=" * 60)
    print("Conversion results:")
    print("=" * 60)
    for name, path in results.items():
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name:20s}  ->  {path}  ({size:.1f} MB)")

    # Post-process: convert to FP16 if requested
    if config.get("fp16"):
        from .fp16 import convert_to_fp16
        print("\n--- FP16 post-processing ---")
        for name, path in results.items():
            if path.endswith("_fp16.onnx"):
                print(f"  Skipping {name} (already exported directly as FP16)")
                continue
            fp16_path = path.replace(".onnx", "_fp16.onnx")
            sidecar = path + ".data"
            if not os.path.isfile(sidecar):
                print(f"  Skipping {name} (no external data, already small)")
                continue
            if config.get("native_fp16"):
                mode = "native fp16 (streaming)" if config.get("native_fp16_streaming") else "native fp16"
            else:
                mode = "storage-only fp16"
            print(f"  Converting {name} to {mode}...")
            convert_to_fp16(
                path,
                fp16_path,
                native=bool(config.get("native_fp16")),
                streaming=bool(config.get("native_fp16_streaming")),
            )
            fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
            fp16_data = fp16_path + ".data"
            data_size = os.path.getsize(fp16_data) / (1024**3) if os.path.isfile(fp16_data) else 0
            print(f"  {name:20s}  ->  {fp16_path}  ({fp16_size:.1f} MB + {data_size:.1f} GB data)")


def cmd_fp16(args):
    """Convert existing ONNX model(s) to FP16."""
    from .fp16 import convert_to_fp16

    input_path = args.input
    output_path = args.output
    if not output_path:
        output_path = input_path.replace(".onnx", "_fp16.onnx")
        if output_path == input_path:
            output_path = input_path + ".fp16.onnx"

    print(f"wannx v{__version__} — FP16 conversion")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    if args.native:
        mode = "native-fp16 graph (streaming)" if args.streaming else "native-fp16 graph"
    else:
        mode = "storage-only fp16"
    print(f"  Mode   : {mode}")
    print()

    convert_to_fp16(
        input_path,
        output_path,
        native=bool(args.native),
        streaming=bool(args.streaming),
    )

    print(f"\nDone! Output: {output_path}")


def cmd_infer(args):
    from .inference import (
        T2VPipeline, auto_convert_if_needed, collect_onnx_models_from_layout,
        find_tokenizer, save_video, ONNX_FILE_NAMES, resolve_onnx_layout,
        PROFILE_SHARED_MODEL_ROLES, PROFILE_SPECIFIC_MODEL_ROLES,
    )

    file_cfg: Dict[str, Any] = {}
    if args.config:
        try:
            file_cfg = _command_config(_load_config_file(args.config), "infer")
        except Exception as exc:
            print(f"ERROR: Failed to load config '{args.config}': {exc}")
            sys.exit(1)

    def pick(name: str, default=None):
        v = getattr(args, name, None)
        if v is not None:
            return v
        return file_cfg.get(name, default)

    prompt = pick("prompt")
    if not prompt:
        print("ERROR: Provide prompt via --prompt or --config")
        sys.exit(1)

    onnx_dir = pick("onnx_dir", None)
    checkpoint_dir = pick("checkpoint_dir", None)

    negative_prompt = pick("negative_prompt", "")
    output = pick("output", "output.mp4")
    num_frames = int(pick("num_frames", 81))
    height = int(pick("height", 480))
    width = int(pick("width", 832))
    num_steps = int(pick("num_steps", 50))
    guidance_scale = float(pick("guidance_scale", _default_guidance_scale(checkpoint_dir)))
    shift = float(pick("shift", _default_flow_shift(width, height, checkpoint_dir)))
    seed = int(pick("seed", -1))
    fps = int(pick("fps", 16))
    text_len = int(pick("text_len", 512))
    export_text_len = int(pick("export_text_len", min(text_len, 64)))
    device = pick("device", "cuda")
    scheduler = str(pick("scheduler", "unipc")).lower()
    text_backend = str(pick("text_backend", "auto")).lower()
    vae_backend = str(pick("vae_backend", "auto")).lower()
    vae_tile_frames = int(pick("vae_tile_frames", 8))
    vae_tile_overlap = int(pick("vae_tile_overlap", 2))
    def as_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)

    keep_models_loaded = as_bool(pick("keep_models_loaded", False), default=False)
    ort_provider = str(pick("ort_provider", "auto")).lower()
    ort_graph_opt_level = str(pick("ort_graph_opt_level", "basic")).lower()
    ort_disable_cpu_fallback = as_bool(pick("ort_disable_cpu_fallback", False), default=False)
    ort_execution_mode = str(pick("ort_execution_mode", "sequential")).lower()
    ort_disable_mem_pattern = as_bool(pick("ort_disable_mem_pattern", False), default=False)
    ort_disable_mem_reuse = as_bool(pick("ort_disable_mem_reuse", False), default=False)
    ort_disable_cpu_mem_arena = as_bool(pick("ort_disable_cpu_mem_arena", False), default=False)
    ort_isolate_dit_call = as_bool(pick("ort_isolate_dit_call", False), default=False)
    ort_reload_dit_session_per_call = as_bool(
        pick("ort_reload_dit_session_per_call", False), default=False
    )
    ort_intra_threads = int(pick("ort_intra_threads", 0))
    ort_inter_threads = int(pick("ort_inter_threads", 0))
    ort_enable_profiling = as_bool(pick("ort_enable_profiling", False), default=False)
    ort_trt_fp16 = as_bool(pick("ort_trt_fp16", False), default=False)
    ort_trt_engine_cache_dir = pick("ort_trt_engine_cache_dir", None)
    ort_trt_max_workspace_size = pick("ort_trt_max_workspace_size", None)
    if ort_trt_max_workspace_size is not None:
        ort_trt_max_workspace_size = int(ort_trt_max_workspace_size)
    ort_trt_builder_optimization_level = pick("ort_trt_builder_optimization_level", None)
    if ort_trt_builder_optimization_level is not None:
        ort_trt_builder_optimization_level = int(ort_trt_builder_optimization_level)
    ort_trt_auxiliary_streams = pick("ort_trt_auxiliary_streams", None)
    if ort_trt_auxiliary_streams is not None:
        ort_trt_auxiliary_streams = int(ort_trt_auxiliary_streams)
    ort_trt_layer_norm_fp32_fallback = as_bool(
        pick("ort_trt_layer_norm_fp32_fallback", False), default=False
    )
    ort_cuda_mem_limit = pick("ort_cuda_mem_limit", None)
    if ort_cuda_mem_limit is not None:
        ort_cuda_mem_limit = int(ort_cuda_mem_limit)
    dit_temporal_chunk_latent = pick("dit_temporal_chunk_latent", None)
    if dit_temporal_chunk_latent is not None:
        dit_temporal_chunk_latent = int(dit_temporal_chunk_latent)
    else:
        dit_temporal_chunk_latent = 0
    dit_temporal_overlap_latent = pick("dit_temporal_overlap_latent", None)
    if dit_temporal_overlap_latent is not None:
        dit_temporal_overlap_latent = int(dit_temporal_overlap_latent)
    else:
        dit_temporal_overlap_latent = 0

    if device not in ("cuda", "cpu"):
        print(f"ERROR: Invalid device '{device}'. Expected 'cuda' or 'cpu'.")
        sys.exit(1)
    if ort_provider not in ("auto", "cuda", "tensorrt", "cpu"):
        print("ERROR: --ort-provider must be one of: auto, cuda, tensorrt, cpu")
        sys.exit(1)
    if ort_graph_opt_level not in ("disable", "basic", "extended", "layout", "all"):
        print("ERROR: --ort-graph-opt-level must be one of: disable, basic, extended, layout, all")
        sys.exit(1)
    if ort_execution_mode not in ("sequential", "parallel"):
        print("ERROR: --ort-execution-mode must be one of: sequential, parallel")
        sys.exit(1)
    if scheduler not in ("unipc", "euler", "dpm++", "dpmpp"):
        print("ERROR: --scheduler must be one of: unipc, dpm++, dpmpp, euler")
        sys.exit(1)
    if text_backend not in ("auto", "onnx", "torch", "torch-isolated"):
        print("ERROR: --text-backend must be one of: auto, onnx, torch, torch-isolated")
        sys.exit(1)
    if vae_backend not in ("auto", "onnx", "torch"):
        print("ERROR: --vae-backend must be one of: auto, onnx, torch")
        sys.exit(1)

    print(f"wannx v{__version__} — T2V inference")
    print()

    # --- Resolve ONNX model paths --- #
    if onnx_dir is None and checkpoint_dir is None:
        print("ERROR: Provide --onnx-dir (pre-exported) or --checkpoint-dir (auto-convert)")
        sys.exit(1)
    _warn_checkpoint_resolution(checkpoint_dir, width, height)
    effective_text_backend = "torch-isolated" if text_backend == "auto" and checkpoint_dir else text_backend
    if effective_text_backend == "auto":
        effective_text_backend = "onnx"
    if effective_text_backend in ("torch", "torch-isolated") and checkpoint_dir is None:
        print("ERROR: --text-backend=torch or torch-isolated requires --checkpoint-dir")
        sys.exit(1)
    effective_vae_backend = "torch" if vae_backend == "auto" and checkpoint_dir else vae_backend
    if effective_vae_backend == "auto":
        effective_vae_backend = "onnx"
    if effective_vae_backend == "torch" and checkpoint_dir is None:
        print("ERROR: --vae-backend=torch requires --checkpoint-dir")
        sys.exit(1)

    if onnx_dir is None:
        onnx_dir = os.path.join(os.path.dirname(output), "onnx_cache")
        os.makedirs(onnx_dir, exist_ok=True)

    # Compute latent dims for conversion config
    latent_f = (num_frames - 1) // 4 + 1
    latent_h = height // 8
    latent_w = width // 8
    use_int8 = as_bool(pick("int8", False), default=False)
    use_fp16 = as_bool(pick("fp16", False), default=False)
    required_model_roles = ["dit"]
    if effective_text_backend == "onnx":
        required_model_roles.append("t5")
    if effective_vae_backend == "onnx":
        required_model_roles.append("vae_decoder")

    prefer_profile_layout = checkpoint_dir is not None
    onnx_root_dir, onnx_profile_dir, use_profile_layout = resolve_onnx_layout(
        onnx_dir,
        num_frames,
        height,
        width,
        prefer_profile_layout=prefer_profile_layout,
    )

    # Auto-convert if needed
    if checkpoint_dir:
        convert_config = {
            "batch_size": 1,
            "dynamic_resolution": False,
            "dynamic_frames": False,
            "dynamic_text_len": False,
            "text_len": text_len,
            "export_text_len": export_text_len,
            "text_dim": 4096,
            "latent_channels": 16,
            "latent_frames": latent_f,
            "latent_height": latent_h,
            "latent_width": latent_w,
            "frames": num_frames,
            "height": height,
            "width": width,
            "local_attn_block_size": int(pick("local_attn_block_size", 0) or 0),
            "fp16": use_fp16,
            "fake_mode_export": as_bool(pick("fake_mode_export", False), default=False),
            "native_fp16": False,
            "native_fp16_streaming": False,
        }
        if use_profile_layout:
            shared_required = [
                role for role in required_model_roles if role in PROFILE_SHARED_MODEL_ROLES
            ]
            profile_required = [
                role for role in required_model_roles if role in PROFILE_SPECIFIC_MODEL_ROLES
            ]
            if shared_required:
                auto_convert_if_needed(checkpoint_dir, onnx_root_dir, shared_required, convert_config)
            if profile_required:
                os.makedirs(onnx_profile_dir, exist_ok=True)
                auto_convert_if_needed(checkpoint_dir, onnx_profile_dir, profile_required, convert_config)
        else:
            auto_convert_if_needed(checkpoint_dir, onnx_root_dir, required_model_roles, convert_config)

    # Verify all ONNX files exist
    models = collect_onnx_models_from_layout(
        onnx_root_dir,
        onnx_profile_dir if use_profile_layout else None,
        prefer_int8=use_int8,
        prefer_fp16=use_fp16,
        text_len=text_len,
    )
    missing = [r for r in required_model_roles if r not in models]
    if missing:
        if use_profile_layout:
            print(f"ERROR: Missing ONNX models for profile {num_frames}f@{width}x{height}")
            print(f"  Root    : {onnx_root_dir}")
            print(f"  Profile : {onnx_profile_dir}")
        else:
            print(f"ERROR: Missing ONNX models in {onnx_root_dir}: {', '.join(missing)}")
        print("  Run 'wannx convert' first or provide --checkpoint-dir for auto-conversion.")
        sys.exit(1)

    # --- Resolve tokenizer --- #
    tokenizer_path = pick("tokenizer", None)
    if tokenizer_path is None and checkpoint_dir:
        tokenizer_path = find_tokenizer(checkpoint_dir)
    if tokenizer_path is None:
        print("ERROR: Cannot find tokenizer. Provide --tokenizer path.")
        sys.exit(1)

    print(f"ONNX root  : {onnx_root_dir}")
    if use_profile_layout:
        print(f"ONNX prof  : {onnx_profile_dir}")
    print(f"Tokenizer  : {tokenizer_path}")
    print(f"Resolution : {width}x{height}, {num_frames} frames")
    print(f"Steps      : {num_steps}, guidance={guidance_scale}, shift={shift}, scheduler={scheduler}")
    print(f"T5 encode  : {effective_text_backend}")
    print(f"VAE decode : {effective_vae_backend}")
    print(f"ORT        : provider={ort_provider}, graph_opt={ort_graph_opt_level}, cpu_fallback={'off' if ort_disable_cpu_fallback else 'on'}")
    print(f"Output     : {output}")
    print()

    # --- Build pipeline and run --- #
    pipeline = T2VPipeline(
        t5_path=models.get("t5", os.path.join(onnx_dir, ONNX_FILE_NAMES["t5"])),
        dit_path=models["dit"],
        vae_decoder_path=models.get("vae_decoder", os.path.join(onnx_dir, ONNX_FILE_NAMES["vae_decoder"])),
        tokenizer_path=tokenizer_path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        text_len=text_len,
        keep_models_loaded=keep_models_loaded,
        ort_provider=ort_provider,
        ort_graph_opt_level=ort_graph_opt_level,
        ort_disable_cpu_fallback=ort_disable_cpu_fallback,
        ort_execution_mode=ort_execution_mode,
        ort_disable_mem_pattern=ort_disable_mem_pattern,
        ort_disable_mem_reuse=ort_disable_mem_reuse,
        ort_disable_cpu_mem_arena=ort_disable_cpu_mem_arena,
        ort_isolate_dit_call=ort_isolate_dit_call,
        ort_reload_dit_session_per_call=ort_reload_dit_session_per_call,
        ort_intra_threads=ort_intra_threads,
        ort_inter_threads=ort_inter_threads,
        ort_enable_profiling=ort_enable_profiling,
        ort_trt_fp16=ort_trt_fp16,
        ort_trt_engine_cache_dir=ort_trt_engine_cache_dir,
        ort_trt_max_workspace_size=ort_trt_max_workspace_size,
        ort_trt_builder_optimization_level=ort_trt_builder_optimization_level,
        ort_trt_auxiliary_streams=ort_trt_auxiliary_streams,
        ort_trt_layer_norm_fp32_fallback=ort_trt_layer_norm_fp32_fallback,
        ort_cuda_mem_limit=ort_cuda_mem_limit,
        scheduler=scheduler,
        text_backend=effective_text_backend,
        vae_backend=effective_vae_backend,
        vae_tile_frames=vae_tile_frames,
        vae_tile_overlap=vae_tile_overlap,
    )

    frames = pipeline.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        seed=seed,
        dit_temporal_chunk_latent=dit_temporal_chunk_latent,
        dit_temporal_overlap_latent=dit_temporal_overlap_latent,
    )

    # --- Save output --- #
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    save_video(frames, output, fps=fps)
    print(f"\nDone! Output: {output}")


def cmd_validate(args):
    from .validator import validate

    targets: List[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".onnx"):
                    targets.append(os.path.join(p, f))
        else:
            targets.append(p)

    if not targets:
        print("No .onnx files found.")
        return

    all_ok = True
    for path in targets:
        print(f"\nValidating: {path}")
        result = validate(path)

        status = "PASS" if result["valid"] else "FAIL"
        print(f"  Status       : {status}")
        print(f"  Size         : {result['size_mb']:.1f} MB")
        print(f"  Structure OK : {result['structure_ok']}")
        print(f"  Inference OK : {result['inference_ok']}")

        if result.get("inputs"):
            print("  Inputs:")
            for inp in result["inputs"]:
                print(f"    {inp['name']:30s}  {inp['shape']}  {inp['type']}")
        if result.get("outputs"):
            print("  Outputs:")
            for out in result["outputs"]:
                print(f"    {out['name']:30s}  {out['shape']}  {out['type']}")
        if result.get("avg_inference_ms"):
            print(f"  Avg inference: {result['avg_inference_ms']:.1f} ms")

        if result["errors"]:
            for err in result["errors"]:
                print(f"  ERROR: {err}")
            all_ok = False

    if args.json:
        print(json.dumps([validate(t) for t in targets], indent=2, default=str))

    sys.exit(0 if all_ok else 1)


# ====================================================================== #
#  Argument parser                                                         #
# ====================================================================== #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wannx",
        description="Convert WAN video generation model modules to ONNX.",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- list -------------------------------------------------------- #
    sub.add_parser("list", help="List available modules")

    # ---- convert ----------------------------------------------------- #
    p_conv = sub.add_parser("convert", help="Convert module(s) to ONNX")
    p_conv.add_argument(
        "--config", default=None,
        help="JSON config file (supports top-level keys or a 'convert' section)",
    )
    p_conv.add_argument(
        "-c", "--checkpoint-dir", default=None,
        help="Path to WAN model checkpoint directory (e.g. models/wan2.1/wan)",
    )
    p_conv.add_argument(
        "-o", "--output-dir", default=None,
        help="Directory to write .onnx files",
    )
    p_conv.add_argument(
        "-m", "--modules", nargs="+", default=None,
        help="Module(s) to convert (dit, vae_encoder, vae_decoder, t5, clip, xlm_roberta, vace) or 'all'",
    )
    p_conv.add_argument("--batch-size", type=int, default=None)
    p_conv.add_argument("--dynamic-resolution", action="store_true", default=None)
    p_conv.add_argument("--dynamic-frames", action="store_true", default=None)
    p_conv.add_argument("--dynamic-text-len", action="store_true", default=None,
                        help="Keep DiT text embedding length dynamic (default: fixed to text_len for TRT-friendly exports)")
    p_conv.add_argument("--text-len", type=int, default=None)
    p_conv.add_argument("--export-text-len", type=int, default=None,
                        help="Trace length for T5 ONNX export")
    p_conv.add_argument("--text-dim", type=int, default=None)
    p_conv.add_argument("--latent-channels", type=int, default=None)
    p_conv.add_argument("--latent-frames", type=int, default=None)
    p_conv.add_argument("--latent-height", type=int, default=None)
    p_conv.add_argument("--latent-width", type=int, default=None)
    p_conv.add_argument("--frames", type=int, default=None)
    p_conv.add_argument("--height", type=int, default=None)
    p_conv.add_argument("--width", type=int, default=None)
    p_conv.add_argument("--local-attn-block-size", type=int, default=None,
                        help="Use block-local self-attention with this token block size for DiT export")
    p_conv.add_argument("--fp16", action="store_true", default=None,
                        help="Export model in FP16 (halves weight size, reduces VRAM)")
    p_conv.add_argument("--fake-mode-export", action="store_true", default=None,
                        help="Export DiT through torch.onnx fake mode and inject shard weights directly")
    p_conv.add_argument("--native-fp16", action="store_true", default=None,
                        help="Use real fp16 graph conversion instead of storage-only fp16+Cast mode")
    p_conv.add_argument("--native-fp16-streaming", action="store_true", default=None,
                        help="Use streaming native-fp16 conversion for large external-data models")
    p_conv.add_argument("--optimize-graph", action="store_true", default=None,
                        help="Run the onnxscript optimizer during dynamo export (folds constant "
                             "Transposes into initializers so weights can be 4-bit quantized)")
    p_conv.add_argument(
        "--xlm-roberta-checkpoint", default=None,
        help="Optional path to XLM-RoBERTa checkpoint weights for xlm_roberta export",
    )

    # ---- infer ------------------------------------------------------- #
    p_inf = sub.add_parser("infer", help="Run T2V inference with ONNX models")
    p_inf.add_argument(
        "--config", default=None,
        help="JSON config file (supports top-level keys or an 'infer' section)",
    )
    p_inf.add_argument(
        "--prompt", default=None,
        help="Text description of the video to generate",
    )
    p_inf.add_argument(
        "--negative-prompt", default=None,
        help="Negative prompt for classifier-free guidance (default: official Wan shared negative prompt)",
    )
    p_inf.add_argument(
        "--onnx-dir", default=None,
        help="Directory with pre-exported ONNX models (t5_encoder.onnx, dit.onnx, vae_decoder.onnx)",
    )
    p_inf.add_argument(
        "-c", "--checkpoint-dir", default=None,
        help="WAN checkpoint directory (auto-converts to ONNX if needed)",
    )
    p_inf.add_argument(
        "-o", "--output", default=None,
        help="Output video path (default: output.mp4)",
    )
    p_inf.add_argument(
        "--tokenizer", default=None,
        help="Path to T5 tokenizer directory (auto-detected from checkpoint-dir if omitted)",
    )
    p_inf.add_argument("--num-frames", type=int, default=None,
                       help="Number of video frames (should be 1+4k, default: 81)")
    p_inf.add_argument("--height", type=int, default=None,
                       help="Video height in pixels (default: 480)")
    p_inf.add_argument("--width", type=int, default=None,
                       help="Video width in pixels (default: 832)")
    p_inf.add_argument("--num-steps", type=int, default=None,
                       help="Number of denoising steps (default: 50)")
    p_inf.add_argument("--guidance-scale", type=float, default=None,
                       help="CFG guidance scale (default: 5.0)")
    p_inf.add_argument("--shift", type=float, default=None,
                       help="Flow-matching shift parameter (default: auto — 5.0 for T2V-14B, 8.0 for 1.3B)")
    p_inf.add_argument("--scheduler", default=None,
                       help="Sampling scheduler: unipc|dpm++|dpmpp|euler (default: unipc)")
    p_inf.add_argument("--text-backend", default=None,
                       help="Prompt encoder backend: auto|onnx|torch|torch-isolated (default: auto)")
    p_inf.add_argument("--vae-tile-frames", type=int, default=None,
                       help="ONNX VAE temporal tile size in latent frames (default 8). "
                            "Smaller = less VRAM. Only affects --vae-backend onnx.")
    p_inf.add_argument("--vae-tile-overlap", type=int, default=None,
                       help="ONNX VAE temporal tile overlap in latent frames (default 1), "
                            "cropped after decode to keep chunk borders seamless.")
    p_inf.add_argument("--vae-backend", default=None,
                       help="VAE decode backend: auto|onnx|torch (default: auto)")
    p_inf.add_argument("--seed", type=int, default=None,
                       help="Random seed (-1 for random, default: -1)")
    p_inf.add_argument("--fps", type=int, default=None,
                       help="Output video FPS (default: 16)")
    p_inf.add_argument("--text-len", type=int, default=None,
                       help="Max text token length (default: 512)")
    p_inf.add_argument("--export-text-len", type=int, default=None,
                       help="T5 trace length used only for auto-conversion (default: min(text_len,64))")
    p_inf.add_argument("--device", default=None,
                       help="ORT execution device (default: cuda)")
    p_inf.add_argument("--keep-models-loaded", action="store_true", default=None,
                       help="Keep all ONNX sessions in memory (needs ~60GB+)")
    p_inf.add_argument("--ort-provider", default=None,
                       help="Execution provider: auto|cuda|tensorrt|cpu (default: auto)")
    p_inf.add_argument("--ort-graph-opt-level", default=None,
                       help="ORT graph optimization: disable|basic|extended|layout|all (default: all)")
    p_inf.add_argument("--ort-disable-cpu-fallback", action="store_true", default=None,
                       help="Disable CPU fallback to detect unsupported GPU ops")
    p_inf.add_argument("--ort-execution-mode", default=None,
                       help="ORT execution mode: sequential|parallel (default: sequential)")
    p_inf.add_argument("--ort-disable-mem-pattern", action="store_true", default=None,
                       help="Disable ORT memory pattern optimization")
    p_inf.add_argument("--ort-disable-mem-reuse", action="store_true", default=None,
                       help="Disable ORT memory reuse optimization")
    p_inf.add_argument("--ort-disable-cpu-mem-arena", action="store_true", default=None,
                       help="Disable ORT CPU memory arena")
    p_inf.add_argument("--ort-isolate-dit-call", action="store_true", default=None,
                       help="Run each DiT ORT call in a fresh subprocess")
    p_inf.add_argument("--ort-reload-dit-session-per-call", action="store_true", default=None,
                       help="Unload and reload the DiT ORT session before each inference call")
    p_inf.add_argument("--ort-intra-threads", type=int, default=None,
                       help="ORT intra-op threads (CPU tuning; default: runtime)")
    p_inf.add_argument("--ort-inter-threads", type=int, default=None,
                       help="ORT inter-op threads (CPU tuning; default: runtime)")
    p_inf.add_argument("--ort-enable-profiling", action="store_true", default=None,
                       help="Enable ORT profiling traces")
    p_inf.add_argument("--ort-trt-fp16", action="store_true", default=None,
                       help="Enable TensorRT FP16 (when --ort-provider tensorrt)")
    p_inf.add_argument("--ort-trt-engine-cache-dir", default=None,
                       help="TensorRT engine cache directory")
    p_inf.add_argument("--ort-trt-max-workspace-size", type=int, default=None,
                       help="TensorRT max workspace size in bytes")
    p_inf.add_argument("--ort-trt-builder-optimization-level", type=int, default=None,
                       help="TensorRT builder optimization level (0-5)")
    p_inf.add_argument("--ort-trt-auxiliary-streams", type=int, default=None,
                       help="TensorRT auxiliary streams count")
    p_inf.add_argument("--ort-trt-layer-norm-fp32-fallback", action="store_true", default=None,
                       help="Force TensorRT layer norm reductions to FP32")
    p_inf.add_argument("--ort-cuda-mem-limit", type=int, default=None,
                       help="CUDA EP GPU memory limit in bytes")
    p_inf.add_argument("--int8", action="store_true", default=None,
                       help="Prefer INT8 quantized models (dit_int8.onnx, t5_encoder_int8.onnx)")
    p_inf.add_argument("--fp16", action="store_true", default=None,
                       help="Prefer FP16 models (dit_fp16.onnx, t5_encoder_fp16.onnx)")
    p_inf.add_argument("--fake-mode-export", action="store_true", default=None,
                       help="When auto-converting DiT, use fake-mode fp16 export instead of fp32+post-conversion")
    p_inf.add_argument("--local-attn-block-size", type=int, default=None,
                       help="Auto-convert DiT with block-local self-attention of this token block size")
    p_inf.add_argument("--dit-temporal-chunk-latent", type=int, default=None,
                       help="Run DiT over temporal latent chunks of this size and average overlaps")
    p_inf.add_argument("--dit-temporal-overlap-latent", type=int, default=None,
                       help="Temporal latent overlap to use with --dit-temporal-chunk-latent")

    # ---- fp16 ------------------------------------------------------- #
    p_fp16 = sub.add_parser("fp16", help="Convert ONNX model weights to FP16")
    p_fp16.add_argument(
        "input",
        help="Input ONNX model path (e.g. dit.onnx)",
    )
    p_fp16.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: <input>_fp16.onnx)",
    )
    p_fp16.add_argument(
        "--native", action="store_true", default=None,
        help="Use real fp16 graph conversion instead of storage-only fp16+Cast mode",
    )
    p_fp16.add_argument(
        "--streaming", action="store_true", default=None,
        help="Use streaming native-fp16 conversion for large external-data models",
    )

    # ---- validate ---------------------------------------------------- #
    p_val = sub.add_parser("validate", help="Validate exported ONNX model(s)")
    p_val.add_argument(
        "paths", nargs="+",
        help="Path(s) to .onnx file(s) or directories containing them",
    )
    p_val.add_argument(
        "--json", action="store_true", help="Print results as JSON",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    handlers = {
        "list": cmd_list,
        "convert": cmd_convert,
        "fp16": cmd_fp16,
        "validate": cmd_validate,
        "infer": cmd_infer,
    }
    handlers[args.command](args)
