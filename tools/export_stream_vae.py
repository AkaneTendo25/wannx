"""Export WAN VAE decoder as a resolution-dynamic STREAMING (recurrent) ONNX step.

The step takes one latent frame + all per-layer 2-frame caches and returns the
decoded frames + updated caches. Caches are pre-filled with zeros, so WAN's
decoder forward runs with no None/'Rep' sentinel branches (constant under trace)
and the streamed result equals a one-shot non-cached decode — but with 1-frame
memory (no cuDNN tensor-size limit, fits 24 GB) and seamless. Cache spatial dims
are exported dynamic (each is a fixed multiple of the latent size), so one graph
decodes any resolution.

Usage:
  python tools/export_stream_vae.py --checkpoint-dir <ckpt> --out vae_decoder.onnx
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wannx.patcher import apply_patches
apply_patches()
from wannx.converters._wan_import import ensure_wan_import_path


class StreamingVAEDecoder(torch.nn.Module):
    """STEP graph: latent frame i>=1 + caches -> 4 decoded frames + caches."""

    def __init__(self, vae):
        super().__init__()
        self.model = vae.model
        self.register_buffer("mean", vae.mean.view(1, -1, 1, 1, 1).clone())
        self.register_buffer("inv_std", (1.0 / vae.std).view(1, -1, 1, 1, 1).clone())

    def forward(self, z_frame, *caches):
        zz = z_frame / self.inv_std + self.mean
        x = self.model.conv2(zz)
        cache_list = list(caches)
        out = self.model.decoder(x, feat_cache=cache_list, feat_idx=[0])
        return (out, *cache_list)


class InitVAEDecoder(torch.nn.Module):
    """INIT graph: latent frame 0 only -> 1 decoded frame + initial caches.

    Runs the decoder on the reference first-latent path (feat_cache all None),
    which is what the real WanVAE.decode does for latent 0: every upsample3d
    emits 'Rep' (no time_conv, no temporal doubling) so latent 0 yields a single
    frame, and every CausalConv3d zero-pads temporally. The resulting caches are
    sanitized into plain tensors so the STEP graph can consume them: conv-layer
    caches (real tensors) are zero-prepadded to CACHE_T=2, and the 'Rep'/None/
    unused slots become zeros at their steady spatial shape (mult * latent size,
    from caches_meta). The pair (init, step) reproduces WanVAE.decode bit-for-bit
    and emits exactly 1 + 4*(T-1) frames — no overshoot, no trim.
    """

    def __init__(self, vae, caches_meta):
        super().__init__()
        self.model = vae.model
        self.caches_meta = caches_meta
        self.register_buffer("mean", vae.mean.view(1, -1, 1, 1, 1).clone())
        self.register_buffer("inv_std", (1.0 / vae.std).view(1, -1, 1, 1, 1).clone())

    def forward(self, z_frame):
        zz = z_frame / self.inv_std + self.mean
        x = self.model.conv2(zz)
        n = len(self.caches_meta)
        fmap = [None] * n
        out = self.model.decoder(x, feat_cache=fmap, feat_idx=[0])
        lh, lw = z_frame.shape[3], z_frame.shape[4]
        sane = []
        for j, (C, mh, mw, used) in enumerate(self.caches_meta):
            c = fmap[j]
            if torch.is_tensor(c):
                if c.shape[2] < 2:
                    c = torch.cat([torch.zeros_like(c[:, :, :1]), c], dim=2)
                sane.append(c)
            elif used:
                sane.append(z_frame.new_zeros(1, C, 2, mh * lh, mw * lw))
            else:
                sane.append(z_frame.new_zeros(1, 1, 2, 1, 1))
        return (out, *sane)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-height", type=int, default=480)
    ap.add_argument("--ref-width", type=int, default=832)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ensure_wan_import_path(a.checkpoint_dir)
    from wan.modules.vae import WanVAE
    vae_pth = next(os.path.join(a.checkpoint_dir, f) for f in os.listdir(a.checkpoint_dir)
                   if "vae" in f.lower() and f.endswith((".pth", ".pt", ".safetensors")))
    vae = WanVAE(vae_pth=vae_pth, device="cpu")
    m = vae.model
    lh, lw = a.ref_height // 8, a.ref_width // 8

    # discover per-cache (channels, spatial multiple of latent) via a path-A dry run
    z = torch.randn(1, 16, 5, lh, lw)
    x = m.conv2(z)
    m.clear_cache()
    with torch.no_grad():
        for i in range(5):
            m.decoder(x[:, :, i:i + 1], feat_cache=m._feat_map, feat_idx=[0])
    caches_meta = []           # (C, mult_h, mult_w, used)
    shapes = []
    for c in m._feat_map:
        if c is None or isinstance(c, str):
            caches_meta.append([1, 0, 0, 0])      # unused slot (static [1,1,2,1,1])
            shapes.append([1, 1, 2, 1, 1])
        else:
            s = list(c.shape)
            mh, mw = s[3] // lh, s[4] // lw
            caches_meta.append([s[1], mh, mw, 1])
            shapes.append([s[0], s[1], 2, s[3], s[4]])
    n = len(shapes)
    print(f"caches: {n} (used {sum(cm[3] for cm in caches_meta)})", flush=True)

    wrapper = StreamingVAEDecoder(vae).eval()
    z_frame = torch.randn(1, 16, 1, lh, lw)
    caches = [torch.zeros(*s) for s in shapes]
    dummy = (z_frame, *caches)
    in_names = ["latent_frame"] + [f"c{i}" for i in range(n)]
    out_names = ["video_frames"] + [f"nc{i}" for i in range(n)]

    # dynamic spatial: latent Dim H/W, each cache spatial = its multiple of H/W
    from torch.export import Dim
    H = Dim("lh", min=1, max=1024)
    W = Dim("lw", min=1, max=1024)
    z_spec = {3: H, 4: W}
    cache_specs = []
    for (C, mh, mw, used) in caches_meta:
        if used and mh >= 1 and mw >= 1:
            cache_specs.append({3: (mh * H) if mh > 1 else H, 4: (mw * W) if mw > 1 else W})
        else:
            cache_specs.append({})                 # unused slot stays static
    # *caches is a single var-positional param -> dynamic_shapes keyed by name.
    dynamic_shapes = {"z_frame": z_spec, "caches": tuple(cache_specs)}

    with torch.no_grad():
        prog = torch.onnx.export(
            wrapper, dummy, input_names=in_names, output_names=out_names,
            dynamic_shapes=dynamic_shapes, dynamo=True, optimize=True,
            verify=False, fallback=False,
        )
    prog.save(a.out)
    meta = {"ref_height": a.ref_height, "ref_width": a.ref_width,
            "n_caches": n, "caches_meta": caches_meta}  # per cache: [C, mult_h, mult_w, used]
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f)
    print("EXPORT_DONE step", a.out, flush=True)

    # --- INIT graph: latent 0 (None-path) -> 1 frame + initial caches ---
    init_out = a.out[:-5] + "_init.onnx" if a.out.endswith(".onnx") else a.out + "_init.onnx"
    init_wrapper = InitVAEDecoder(vae, caches_meta).eval()
    init_in_names = ["latent_frame"]
    init_out_names = ["video_frames"] + [f"nc{i}" for i in range(n)]
    init_dynamic = {"z_frame": z_spec}
    with torch.no_grad():
        init_prog = torch.onnx.export(
            init_wrapper, (z_frame,), input_names=init_in_names,
            output_names=init_out_names, dynamic_shapes=init_dynamic,
            dynamo=True, optimize=True, verify=False, fallback=False,
        )
    init_prog.save(init_out)
    print("EXPORT_DONE init", init_out, flush=True)


if __name__ == "__main__":
    main()
