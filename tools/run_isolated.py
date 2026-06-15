"""Strict-24GB isolated T2V runner.

Runs the three heavy stages (T5 encode -> DiT denoise -> VAE decode) each in its
OWN subprocess, so the GPU is fully released between stages (ORT does not return
its CUDA arena on session delete within a single process). End-to-end peak then
equals the largest single stage (~21 GB for the 4-bit 14B at 81f) instead of the
sum, which is what makes the full 14B pipeline fit a 24 GB GPU.

Usage:
    python tools/run_isolated.py --checkpoint-dir <ckpt> --dit <q4 dit.onnx> \
        --out out.mp4 --prompt "..." --num-frames 81 --height 480 --width 832 --steps 20
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _latent_shape(num_frames, h, w):
    return (1, 16, (num_frames - 1) // 4 + 1, h // 8, w // 8)


def stage_t5(cfg, work):
    from wannx.inference import resolve_negative_prompt

    texts = [cfg["prompt"]]
    if cfg["guidance"] > 1.0:
        texts.append(resolve_negative_prompt(cfg["negative_prompt"]))

    if cfg.get("t5_onnx"):
        # Pure-ONNX T5 (no checkpoint needed).
        from wannx.inference import OnnxModelSession, T5Tokenizer
        tok = T5Tokenizer(cfg["tokenizer"], text_len=cfg["text_len"])
        sess = OnnxModelSession(cfg["t5_onnx"], device="cuda", provider="cuda",
                                graph_opt_level="basic")
        sess.load()
        embs = []
        for t in texts:
            ids, mask = tok(t)
            emb = sess(input_ids=ids, attention_mask=mask)[0].astype(np.float32)
            # Zero embeddings past the real tokens (the DiT expects padded
            # positions to be zero, as torch T5 + pad_text_embeddings produce).
            seq_len = max(int(mask[0].sum()), 1)
            if seq_len < emb.shape[1]:
                emb[:, seq_len:, :] = 0.0
            embs.append(emb)
        sess.unload()
        for name, emb in zip(["cond", "uncond"], embs):
            np.save(os.path.join(work, name + ".npy"), emb)  # [1, text_len, 4096]
        return

    # Torch T5 from the checkpoint, bf16 (~11 GB vs 22 GB fp32).
    import torch
    from wannx.patcher import apply_patches
    apply_patches()
    from wannx.converters._wan_import import ensure_wan_import_path
    ensure_wan_import_path(cfg["checkpoint_dir"])
    from wan.modules.t5 import T5EncoderModel
    from wannx.inference import pad_text_embeddings

    t5_pth = next(
        os.path.join(cfg["checkpoint_dir"], f)
        for f in os.listdir(cfg["checkpoint_dir"])
        if "t5" in f.lower() and f.endswith((".pth", ".pt", ".safetensors"))
    )
    enc = T5EncoderModel(
        text_len=cfg["text_len"], dtype=torch.bfloat16, device=torch.device("cuda"),
        checkpoint_path=t5_pth, tokenizer_path=cfg["tokenizer"], shard_fn=None,
    )
    with torch.no_grad():
        embs = enc(texts, "cuda")
    for name, emb in zip(["cond", "uncond"], embs):
        arr = pad_text_embeddings(emb.float().cpu().numpy().astype(np.float32), cfg["text_len"])
        np.save(os.path.join(work, name + ".npy"), arr[None, ...])


def stage_dit(cfg, work):
    from wannx.inference import OnnxModelSession

    if cfg.get("torch_free"):
        # Pure ONNX Runtime: numpy UniPC scheduler, no torch, no WAN patches.
        from wannx.np_scheduler import NumpyUniPCScheduler
        sch = NumpyUniPCScheduler(shift=cfg["shift"])
        sch.set_timesteps(cfg["steps"], shift=cfg["shift"])
    else:
        from wannx.patcher import apply_patches
        apply_patches()
        from wannx.inference import build_scheduler
        sch = build_scheduler(cfg["scheduler"], num_steps=cfg["steps"], shift=cfg["shift"])

    cond = np.load(os.path.join(work, "cond.npy"))
    uncond = np.load(os.path.join(work, "uncond.npy")) if cfg["guidance"] > 1.0 else None
    rng = np.random.default_rng(cfg["seed"] if cfg["seed"] >= 0 else None)
    lat = rng.standard_normal(tuple(cfg["latent_shape"])).astype(np.float32)
    lat = sch.scale_noise(lat)

    # disable_mem_pattern: ORT's memory-pattern planner pre-allocates a large
    # buffer across the multi-step CFG loop (81f denoise: 21->34 GB). Disabling it
    # keeps the denoise at ~21.5 GB so the full 14B fits a 24 GB GPU.
    dit = OnnxModelSession(
        cfg["dit"], device="cuda", provider="cuda",
        graph_opt_level="basic", disable_mem_pattern=True,
    )
    dit.load()

    def pred(l, t, e):
        return dit(latent_input=l, timestep=t, text_embeddings=e)[0]

    for i in range(cfg["steps"]):
        t = sch.timesteps[i:i + 1].astype(np.float32)
        nc = pred(lat, t, cond)
        if uncond is not None:
            nu = pred(lat, t, uncond)
            npred = nu + cfg["guidance"] * (nc - nu)
        else:
            npred = nc
        lat = sch.step(npred, lat, i)
        print(f"  dit step {i + 1}/{cfg['steps']}", flush=True)
    np.save(os.path.join(work, "latents.npy"), lat)


def stage_vae(cfg, work):
    # apply_patches is only needed for the torch VAE (WAN torch modules); the
    # ONNX VAE paths run purely on ONNX Runtime + numpy.
    if not (cfg.get("vae_stream") or cfg.get("vae_onnx")):
        from wannx.patcher import apply_patches
        apply_patches()
    from wannx.inference import latents_to_video, save_video

    lat = np.load(os.path.join(work, "latents.npy"))
    if cfg.get("vae_stream"):
        # Streaming (recurrent) ONNX VAE: one latent frame at a time with
        # per-layer caches. Faithful to the reference WanVAE.decode (init graph
        # for latent 0 + step graph for the rest), 1-frame memory, fits 24 GB.
        import onnxruntime as ort
        from wannx.inference import onnx_vae_decode_streaming
        step_path = cfg["vae_stream"]
        init_path = step_path[:-5] + "_init.onnx" if step_path.endswith(".onnx") \
            else step_path + "_init.onnx"
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        init_sess = ort.InferenceSession(init_path, so, providers=providers)
        step_sess = ort.InferenceSession(step_path, so, providers=providers)
        raw = onnx_vae_decode_streaming(init_sess, step_sess, lat)
        del init_sess, step_sess
    elif cfg.get("vae_onnx"):
        # Pure-ONNX VAE, temporally tiled so memory is bounded (isolated stage
        # -> the DiT arena is already freed, so 81f fits a 24 GB GPU).
        from wannx.inference import OnnxModelSession, onnx_vae_decode_tiled
        sess = OnnxModelSession(cfg["vae_onnx"], device="cuda", provider="cuda",
                                graph_opt_level="basic", disable_mem_pattern=True)
        sess.load()
        raw = onnx_vae_decode_tiled(
            lambda l: sess(latent=l)[0], lat,
            tile_frames=cfg.get("vae_tile_frames", 8),
            tile_overlap=cfg.get("vae_tile_overlap", 2),
        )
        sess.unload()
    else:
        from wannx.inference import TorchVAEDecoder
        vae = TorchVAEDecoder(checkpoint_dir=cfg["checkpoint_dir"], device="cuda")
        raw = vae(lat)
    frames = latents_to_video(raw)
    if frames.shape[0] > cfg["num_frames"]:
        frames = frames[:cfg["num_frames"]]
    os.makedirs(os.path.dirname(os.path.abspath(cfg["out"])), exist_ok=True)
    save_video(frames, cfg["out"], fps=16)
    print(f"frames {frames.shape} min {frames.min()} max {frames.max()} mean {float(frames.mean()):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["t5", "dit", "vae"], default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="WAN checkpoint (for torch T5/VAE). Optional if --t5-onnx and --vae-onnx given.")
    ap.add_argument("--dit")
    ap.add_argument("--t5-onnx", default=None,
                    help="ONNX T5 encoder. If set, T5 runs on ONNX Runtime (needs --tokenizer); "
                         "otherwise torch T5 from --checkpoint-dir.")
    ap.add_argument("--vae-onnx", default=None,
                    help="ONNX VAE decoder (one-shot/temporally-tiled). Otherwise torch VAE.")
    ap.add_argument("--vae-stream", default=None,
                    help="Streaming (recurrent) ONNX VAE decoder (from export_stream_vae.py). "
                         "Seamless + 1-frame memory; the seamless pure-ONNX 24 GB path. "
                         "Needs <path>.meta.json alongside.")
    ap.add_argument("--vae-tile-frames", type=int, default=8)
    ap.add_argument("--vae-tile-overlap", type=int, default=2)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--prompt", default="A cat walking through a sunlit garden, cinematic, high detail")
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--shift", type=float, default=5.0)  # WAN T2V-14B ref (3.0 is i2v)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler", default="unipc")
    ap.add_argument("--text-len", type=int, default=512)
    a = ap.parse_args()

    if a.stage:
        with open(os.path.join(a.work, "cfg.json")) as f:
            cfg = json.load(f)
        {"t5": stage_t5, "dit": stage_dit, "vae": stage_vae}[a.stage](cfg, a.work)
        if cfg.get("torch_free"):
            print(f"[{a.stage}] torch_imported={'torch' in sys.modules}", flush=True)
        return

    from wannx.inference import find_tokenizer
    tokenizer = a.tokenizer or (find_tokenizer(a.checkpoint_dir) if a.checkpoint_dir else None)
    if (a.t5_onnx or not a.checkpoint_dir) and not tokenizer:
        raise SystemExit("--tokenizer is required with --t5-onnx (or when no --checkpoint-dir)")
    work = tempfile.mkdtemp(prefix="wannx_iso_")
    cfg = dict(
        checkpoint_dir=a.checkpoint_dir, dit=a.dit, tokenizer=tokenizer, out=a.out,
        prompt=a.prompt, negative_prompt=a.negative_prompt, num_frames=a.num_frames,
        latent_shape=list(_latent_shape(a.num_frames, a.height, a.width)),
        steps=a.steps, guidance=a.guidance, shift=a.shift, seed=a.seed,
        scheduler=a.scheduler, text_len=a.text_len,
        t5_onnx=a.t5_onnx, vae_onnx=a.vae_onnx, vae_stream=a.vae_stream,
        vae_tile_frames=a.vae_tile_frames, vae_tile_overlap=a.vae_tile_overlap,
        # Fully torch-free path: ONNX T5 + ONNX VAE + numpy scheduler, scheduler=unipc.
        torch_free=bool(a.t5_onnx and (a.vae_onnx or a.vae_stream)
                        and a.scheduler == "unipc"),
    )
    with open(os.path.join(work, "cfg.json"), "w") as f:
        json.dump(cfg, f)
    for st in ("t5", "dit", "vae"):
        print(f"=== STAGE {st} ===", flush=True)
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--stage", st, "--work", work],
            cwd=ROOT,
        )
        if r.returncode != 0:
            raise SystemExit(f"stage {st} failed (rc={r.returncode})")
    print("ISO_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
