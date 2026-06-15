"""Simple interactive CLI for conversion and inference with progress bars.

This CLI supports two deployment styles:
1) Legacy single-profile exports (one dit/vae pair in output_dir)
2) Multi-profile exports under output_dir/profiles/f{frames}_h{height}_w{width}
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

PROFILE_ROOT = "profiles"
PROFILE_RE = re.compile(r"^f(?P<f>\d+)_h(?P<h>\d+)_w(?P<w>\d+)$")


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")
    return data


def _section(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    sec = cfg.get(name)
    if isinstance(sec, dict):
        return sec
    return cfg


def _ask_str(label: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{label}{suffix}: ").strip()
    if val:
        return val
    return "" if default is None else str(default)


def _ask_int(label: str, default: int) -> int:
    while True:
        val = _ask_str(label, str(default))
        try:
            return int(val)
        except ValueError:
            print(f"Invalid integer: {val}")


def _ask_float(label: str, default: float) -> float:
    while True:
        val = _ask_str(label, str(default))
        try:
            return float(val)
        except ValueError:
            print(f"Invalid float: {val}")


def _latent_dims(num_frames: int, height: int, width: int) -> Tuple[int, int, int]:
    return ((num_frames - 1) // 4 + 1, height // 8, width // 8)


def _profile_name(num_frames: int, height: int, width: int) -> str:
    return f"f{num_frames}_h{height}_w{width}"


def _profile_path(root_dir: str, num_frames: int, height: int, width: int) -> str:
    return os.path.join(root_dir, PROFILE_ROOT, _profile_name(num_frames, height, width))


def _parse_profiles(cfg: Dict[str, Any]) -> List[Dict[str, int]]:
    raw_profiles = cfg.get("profiles")
    if not raw_profiles:
        return [{
            "frames": int(cfg.get("frames", 5)),
            "height": int(cfg.get("height", 512)),
            "width": int(cfg.get("width", 512)),
        }]

    profiles: List[Dict[str, int]] = []
    if not isinstance(raw_profiles, list):
        raise ValueError("'profiles' must be a list of objects")
    for i, p in enumerate(raw_profiles):
        if not isinstance(p, dict):
            raise ValueError(f"profiles[{i}] must be an object")
        frames = int(p.get("frames", cfg.get("frames", 5)))
        height = int(p.get("height", cfg.get("height", 512)))
        width = int(p.get("width", cfg.get("width", 512)))
        profiles.append({"frames": frames, "height": height, "width": width})
    return profiles


def _discover_profiles(root_dir: str) -> List[Dict[str, int]]:
    prof_root = os.path.join(root_dir, PROFILE_ROOT)
    found: List[Dict[str, int]] = []
    if not os.path.isdir(prof_root):
        return found

    for name in sorted(os.listdir(prof_root)):
        m = PROFILE_RE.match(name)
        if not m:
            continue
        frames = int(m.group("f"))
        height = int(m.group("h"))
        width = int(m.group("w"))
        sub = os.path.join(prof_root, name)
        dit = os.path.join(sub, "dit.onnx")
        vae = os.path.join(sub, "vae_decoder.onnx")
        if os.path.isfile(dit) and os.path.isfile(vae):
            found.append({"frames": frames, "height": height, "width": width})
    return found


def _write_profile_manifest(root_dir: str, profiles: List[Dict[str, int]]) -> None:
    path = os.path.join(root_dir, "profiles.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"profiles": profiles}, f, indent=2)


def _legacy_module_list(cfg: Dict[str, Any]) -> List[str]:
    from .converter import list_modules

    modules = cfg.get("modules", ["all"])
    if isinstance(modules, str):
        modules = [modules]
    if not modules:
        modules = ["all"]
    if "all" in modules:
        modules = list_modules()
    return list(modules)


def _convert(config_path: str) -> None:
    from .patcher import apply_patches
    apply_patches()
    from .converter import get_converter

    cfg = _section(_load_config(config_path), "convert")
    checkpoint_dir = cfg.get("checkpoint_dir")
    output_dir = cfg.get("output_dir")
    if not checkpoint_dir or not output_dir:
        raise ValueError("convert config must define checkpoint_dir and output_dir")
    os.makedirs(output_dir, exist_ok=True)

    # Legacy mode: modules-only config, single profile in output_dir.
    if "profiles" not in cfg and "shared_modules" not in cfg and "profiled_modules" not in cfg:
        modules = _legacy_module_list(cfg)
        print(f"Checkpoint: {checkpoint_dir}")
        print(f"Output dir : {output_dir}")
        print(f"Modules    : {', '.join(modules)}")
        results: Dict[str, str] = {}
        with tqdm(total=len(modules), desc="Converting", unit="module") as pbar:
            for module_name in modules:
                pbar.set_postfix_str(module_name)
                converter = get_converter(module_name)
                onnx_path = converter.export(
                    checkpoint_dir=checkpoint_dir,
                    output_dir=output_dir,
                    config=cfg,
                )
                results[module_name] = onnx_path
                pbar.update(1)
        print("\nDone. Exported:")
        for name, path in results.items():
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {name:12s} {path} ({size_mb:.1f} MB)")
        return

    profiles = _parse_profiles(cfg)
    shared_modules = cfg.get("shared_modules", ["t5"])
    profiled_modules = cfg.get("profiled_modules", ["dit", "vae_decoder"])
    if isinstance(shared_modules, str):
        shared_modules = [shared_modules]
    if isinstance(profiled_modules, str):
        profiled_modules = [profiled_modules]

    total_jobs = len(shared_modules) + len(profiled_modules) * len(profiles)
    print(f"Checkpoint       : {checkpoint_dir}")
    print(f"Output root      : {output_dir}")
    print(f"Shared modules   : {', '.join(shared_modules) if shared_modules else '(none)'}")
    print(f"Profiled modules : {', '.join(profiled_modules)}")
    print("Profiles:")
    for p in profiles:
        print(f"  - frames={p['frames']}, size={p['width']}x{p['height']}")

    with tqdm(total=total_jobs, desc="Converting", unit="task") as pbar:
        # Export shared modules once at root.
        for module_name in shared_modules:
            pbar.set_postfix_str(f"shared:{module_name}")
            converter = get_converter(module_name)
            converter.export(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                config=cfg,
            )
            pbar.update(1)

        # Export profile-specific modules into profile subfolders.
        for p in profiles:
            frames = int(p["frames"])
            height = int(p["height"])
            width = int(p["width"])
            latent_f, latent_h, latent_w = _latent_dims(frames, height, width)
            prof_cfg = dict(cfg)
            prof_cfg.update({
                "frames": frames,
                "height": height,
                "width": width,
                "latent_frames": int(p.get("latent_frames", latent_f)),
                "latent_height": int(p.get("latent_height", latent_h)),
                "latent_width": int(p.get("latent_width", latent_w)),
            })
            prof_out = _profile_path(output_dir, frames, height, width)
            os.makedirs(prof_out, exist_ok=True)
            for module_name in profiled_modules:
                pbar.set_postfix_str(f"{module_name}:{frames}f")
                converter = get_converter(module_name)
                converter.export(
                    checkpoint_dir=checkpoint_dir,
                    output_dir=prof_out,
                    config=prof_cfg,
                )
                pbar.update(1)

    _write_profile_manifest(output_dir, profiles)
    print("\nDone. Multi-profile export complete.")
    print(f"Manifest: {os.path.join(output_dir, 'profiles.json')}")


def _infer(config_path: str) -> None:
    from .inference import T2VPipeline, find_tokenizer, save_video
    from .patcher import apply_patches
    from .converter import get_converter

    all_cfg = _load_config(config_path)
    cfg = _section(all_cfg, "infer")
    conv_cfg = _section(all_cfg, "convert")

    checkpoint_dir = cfg.get("checkpoint_dir", conv_cfg.get("checkpoint_dir"))
    onnx_root = cfg.get("onnx_dir", conv_cfg.get("output_dir"))
    tokenizer_path = cfg.get("tokenizer")

    if not onnx_root and not checkpoint_dir:
        raise ValueError("infer config must define onnx_dir/output_dir or checkpoint_dir")
    if not onnx_root:
        onnx_root = os.path.join(os.getcwd(), "onnx_cache")
    os.makedirs(onnx_root, exist_ok=True)

    prompt_default = cfg.get("prompt", "")
    prompt = _ask_str("Prompt", prompt_default)
    while not prompt:
        prompt = _ask_str("Prompt (required)")
    negative_prompt = _ask_str("Negative prompt", cfg.get("negative_prompt", ""))
    num_steps = _ask_int("Steps", int(cfg.get("num_steps", 30)))
    num_frames = _ask_int("Frames", int(cfg.get("num_frames", 5)))
    height = _ask_int("Height", int(cfg.get("height", 512)))
    width = _ask_int("Width", int(cfg.get("width", 512)))
    guidance = _ask_float("Guidance scale", float(cfg.get("guidance_scale", 5.0)))
    shift = _ask_float("Shift", float(cfg.get("shift", 3.0)))
    seed = _ask_int("Seed (-1 random)", int(cfg.get("seed", -1)))
    output = _ask_str("Output video", cfg.get("output", os.path.join(os.getcwd(), "output.mp4")))

    text_len = int(cfg.get("text_len", conv_cfg.get("text_len", 512)))
    export_text_len = int(cfg.get("export_text_len", conv_cfg.get("export_text_len", min(text_len, 64))))
    device = str(cfg.get("device", "cuda"))
    fps = int(cfg.get("fps", 16))

    ort_provider = str(cfg.get("ort_provider", "auto"))
    ort_graph_opt_level = str(cfg.get("ort_graph_opt_level", "basic"))
    ort_disable_cpu_fallback = bool(cfg.get("ort_disable_cpu_fallback", False))
    ort_intra_threads = int(cfg.get("ort_intra_threads", 0))
    ort_inter_threads = int(cfg.get("ort_inter_threads", 0))
    ort_enable_profiling = bool(cfg.get("ort_enable_profiling", False))
    ort_trt_fp16 = bool(cfg.get("ort_trt_fp16", False))
    ort_trt_engine_cache_dir = cfg.get("ort_trt_engine_cache_dir")
    text_backend = str(cfg.get("text_backend", "auto"))
    vae_backend = str(cfg.get("vae_backend", "onnx"))

    t5_path = os.path.join(onnx_root, "t5_encoder.onnx")
    prof_dir = _profile_path(onnx_root, num_frames, height, width)
    dit_path = os.path.join(prof_dir, "dit.onnx")
    vae_path = os.path.join(prof_dir, "vae_decoder.onnx")

    # Backward compatible fallback to single-profile root files.
    if not os.path.isfile(dit_path):
        fallback_dit = os.path.join(onnx_root, "dit.onnx")
        fallback_vae = os.path.join(onnx_root, "vae_decoder.onnx")
        if os.path.isfile(fallback_dit) and os.path.isfile(fallback_vae):
            dit_path = fallback_dit
            vae_path = fallback_vae
            prof_dir = onnx_root

    # Auto-build missing profile if checkpoint is available.
    if checkpoint_dir and (not os.path.isfile(t5_path) or not os.path.isfile(dit_path) or not os.path.isfile(vae_path)):
        apply_patches()
        os.makedirs(prof_dir, exist_ok=True)
        latent_f, latent_h, latent_w = _latent_dims(num_frames, height, width)
        export_cfg = dict(conv_cfg)
        export_cfg.update({
            "text_len": text_len,
            "export_text_len": export_text_len,
            "text_dim": int(conv_cfg.get("text_dim", 4096)),
            "latent_channels": int(conv_cfg.get("latent_channels", 16)),
            "latent_frames": int(conv_cfg.get("latent_frames", latent_f)),
            "latent_height": int(conv_cfg.get("latent_height", latent_h)),
            "latent_width": int(conv_cfg.get("latent_width", latent_w)),
            "frames": int(conv_cfg.get("frames", num_frames)),
            "height": int(conv_cfg.get("height", height)),
            "width": int(conv_cfg.get("width", width)),
            "dynamic_resolution": bool(conv_cfg.get("dynamic_resolution", False)),
            "dynamic_frames": bool(conv_cfg.get("dynamic_frames", False)),
            "batch_size": int(conv_cfg.get("batch_size", 1)),
        })
        if not os.path.isfile(t5_path):
            get_converter("t5").export(checkpoint_dir, onnx_root, export_cfg)
        if not os.path.isfile(dit_path):
            get_converter("dit").export(checkpoint_dir, prof_dir, export_cfg)
        if not os.path.isfile(vae_path):
            get_converter("vae_decoder").export(checkpoint_dir, prof_dir, export_cfg)

    if not os.path.isfile(t5_path) or not os.path.isfile(dit_path) or not os.path.isfile(vae_path):
        avail = _discover_profiles(onnx_root)
        avail_msg = ", ".join(
            f"{p['frames']}f@{p['width']}x{p['height']}" for p in avail
        ) or "(none)"
        raise FileNotFoundError(
            "Missing required ONNX files.\n"
            f"  t5={os.path.isfile(t5_path)} dit={os.path.isfile(dit_path)} vae={os.path.isfile(vae_path)}\n"
            f"  requested profile: {num_frames}f@{width}x{height}\n"
            f"  available profiles: {avail_msg}"
        )

    if tokenizer_path is None and checkpoint_dir:
        tokenizer_path = find_tokenizer(checkpoint_dir)
    if tokenizer_path is None:
        raise ValueError("Tokenizer not found. Set infer.tokenizer or checkpoint_dir.")

    print(f"Using profile: {prof_dir}")
    pipeline = T2VPipeline(
        t5_path=t5_path,
        dit_path=dit_path,
        vae_decoder_path=vae_path,
        tokenizer_path=tokenizer_path,
        checkpoint_dir=checkpoint_dir,
        device=device,
        text_len=text_len,
        keep_models_loaded=bool(cfg.get("keep_models_loaded", False)),
        ort_provider=ort_provider,
        ort_graph_opt_level=ort_graph_opt_level,
        ort_disable_cpu_fallback=ort_disable_cpu_fallback,
        ort_intra_threads=ort_intra_threads,
        ort_inter_threads=ort_inter_threads,
        ort_enable_profiling=ort_enable_profiling,
        ort_trt_fp16=ort_trt_fp16,
        ort_trt_engine_cache_dir=ort_trt_engine_cache_dir,
        text_backend=text_backend,
        vae_backend=vae_backend,
    )

    with tqdm(total=num_steps, desc="Denoising", unit="step") as pbar:
        def callback(step: int, total: int, _latents) -> None:
            pbar.n = step
            pbar.refresh()

        frames = pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_steps=num_steps,
            guidance_scale=guidance,
            shift=shift,
            seed=seed,
            callback=callback,
        )

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    save_video(frames, output, fps=fps)
    print(f"\nDone: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wannx-simple",
        description="Simple interactive CLI for ONNX convert/infer with progress bars",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON config path (supports top-level or convert/infer sections)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("convert", help="Convert modules from config")
    sub.add_parser("infer", help="Run interactive inference using config defaults")
    args = parser.parse_args()

    if args.command == "convert":
        _convert(args.config)
    else:
        _infer(args.config)


if __name__ == "__main__":
    main()
