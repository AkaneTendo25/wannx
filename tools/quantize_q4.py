"""4-bit weight-only quantization of a (large) ONNX DiT to MatMulNBits.

Usage:  python tools/quantize_q4.py <in_dit.onnx> <out_dit.onnx>

Requires the export to have used --optimize-graph (so Linear weights are const
initializers, not Transpose(W)); otherwise the MatMuls are skipped. The result
keeps the attention as a fp16 MultiHeadAttention island; only the linear weights
become 4-bit (MatMulNBits), which ORT dequantizes in-kernel (true memory saving).
"""

import os
import sys
import time

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def main() -> int:
    inp, outp = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    t0 = time.time()
    print(f"loading {inp} ...", flush=True)
    m = onnx.load(inp, load_external_data=True)
    print(f"loaded in {time.time()-t0:.0f}s; quantizing 4-bit (block_size=128)...", flush=True)
    t0 = time.time()
    # Older ORT (e.g. 1.23.2) has no `bits` kwarg (4-bit is the default); newer
    # ORT (1.24+) accepts bits=4. Build kwargs adaptively for portability.
    import inspect
    params = inspect.signature(MatMulNBitsQuantizer.__init__).parameters
    kw = dict(block_size=128, is_symmetric=False)
    if "bits" in params:
        kw["bits"] = 4
    q = MatMulNBitsQuantizer(m, **kw)
    q.process()
    qm = q.model.model if hasattr(q.model, "model") else q.model
    print(f"quantized in {time.time()-t0:.0f}s; saving...", flush=True)
    onnx.save(
        qm, outp, save_as_external_data=True, all_tensors_to_one_file=True,
        location=os.path.basename(outp) + ".data",
    )
    import collections
    c = collections.Counter(n.op_type for n in qm.graph.node)
    print(
        f"MatMulNBits={c.get('MatMulNBits',0)} MatMul={c.get('MatMul',0)} "
        f"MHA={c.get('MultiHeadAttention',0)}  ->  {outp}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
