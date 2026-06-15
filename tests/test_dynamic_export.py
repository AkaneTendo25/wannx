"""Cross-shape parity test for the dynamic DiT ONNX export.

This is the real guard for the "one ONNX, any resolution / any duration"
guarantee: it exports a tiny WanModel ONCE (with dynamic_shapes) and then runs
the single ONNX through ONNX Runtime at several DIFFERENT latent shapes,
comparing against eager PyTorch each time.

The previous DiT test only checked output shape at the export shape, so a
shape-locked graph (grid_sizes constant-folded into the graph) would pass it.
Here a regression would show up as a cos < 1 or a runtime error at a non-export
shape.

Run:
    python tests/test_dynamic_export.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wannx.patcher import apply_patches

apply_patches()

from wannx.converters._wan_import import ensure_wan_import_path

ensure_wan_import_path(os.path.join(ROOT, "models", "wan2.1"))

from wan.modules.model import WanModel  # noqa: E402
from wannx.converters.dit import _ONNXWanModel, DiTConverter  # noqa: E402

TINY_CFG = dict(
    model_type="t2v", dim=48, eps=1e-6, ffn_dim=96, freq_dim=16,
    in_dim=16, num_heads=4, num_layers=2, out_dim=16, text_len=512,
)


def _make_inputs(F, H, W, L):
    g = torch.Generator().manual_seed(123)
    return (
        torch.randn(1, 16, F, H, W, generator=g),
        torch.tensor([500.0]),
        torch.randn(1, L, 4096, generator=g),
    )


def main() -> int:
    import onnxruntime as ort

    torch.manual_seed(0)
    model = WanModel.from_config(TINY_CFG).float().eval().requires_grad_(False)
    # WanModel uses adaLN-zero init; break the zeroed params so the untrained
    # model produces a non-zero reference (otherwise cosine is 0/0).
    with torch.no_grad():
        for p in model.parameters():
            if float(p.abs().sum()) == 0.0:
                p.normal_(0, 0.02)

    wrapped = _ONNXWanModel(model).eval()
    conv = DiTConverter()

    export_shape = (2, 8, 8, 4)
    xd, td, cd = _make_inputs(*export_shape)
    dynamic_shapes = conv._build_dynamic_shapes(
        dict(dynamic_frames=True, dynamic_resolution=True, dynamic_text_len=True)
    )

    onnx_path = os.path.join(tempfile.mkdtemp(prefix="wannx_dyn_"), "tiny_dit.onnx")
    program = torch.onnx.export(
        wrapped, (xd, td, cd),
        input_names=["latent_input", "timestep", "text_embeddings"],
        output_names=["noise_prediction"],
        dynamic_shapes=dynamic_shapes,
        dynamo=True, optimize=False, verify=False, fallback=False,
    )
    program.save(onnx_path)

    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    # Shapes spanning frames / height / width / text-len, none == export shape.
    test_shapes = [(2, 8, 8, 4), (3, 12, 10, 7), (5, 16, 8, 12), (4, 8, 16, 20), (7, 20, 14, 33)]
    ok = True
    for (F, H, W, L) in test_shapes:
        x, t, c = _make_inputs(F, H, W, L)
        with torch.no_grad():
            ref = wrapped(x, t, c).cpu().numpy()
        out = sess.run(None, {
            "latent_input": x.numpy(),
            "timestep": t.numpy(),
            "text_embeddings": c.numpy(),
        })[0]
        if out.shape != ref.shape:
            print(f"FAIL F{F} H{H} W{W} L{L}: shape {out.shape} != {ref.shape}")
            ok = False
            continue
        maxabs = float(np.abs(out - ref).max())
        rn = np.linalg.norm(ref.ravel())
        on = np.linalg.norm(out.ravel())
        cos = float(np.dot(ref.ravel(), out.ravel()) / (rn * on)) if rn > 0 and on > 0 else float("nan")
        status = "ok " if (cos > 0.9999 and maxabs < 1e-2) else "BAD"
        print(f"[{status}] F{F} H{H} W{W} L{L}: maxabs={maxabs:.3e} cos={cos:.6f}")
        ok = ok and cos > 0.9999 and maxabs < 1e-2

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
