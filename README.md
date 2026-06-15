# wannx

ONNX export and ONNX Runtime inference for [Wan2.1-T2V-14B](https://github.com/Wan-Video/Wan2.1)
(text-to-video). The DiT, T5 encoder, and VAE decoder export to dynamic ONNX graphs (variable
resolution and duration). The 4-bit DiT runs 81 frames at 480×832 within 24 GB of VRAM.

Preconverted weights: `https://huggingface.co/AkaneTendo25/wan21-onnx`

## Components

| Component | Export | Dynamic axes |
|---|---|---|
| DiT (`WanModel`) | dynamo, single graph | frames, height, width, text-len |
| T5 (`umt5-xxl`) | single graph | batch, sequence length |
| VAE decoder | dynamo, streaming/recurrent (init + step graphs) | any resolution; one latent frame at a time (bit-exact to the reference decode, 1-frame memory, fits 24 GB) |

- DiT ONNX vs PyTorch: cosine similarity ≈ 1.0 across tested shapes (`tests/test_dynamic_export.py`).
- DiT attention is emitted as `com.microsoft.MultiHeadAttention` (fp16); linear weights are 4-bit
  `MatMulNBits`, dequantized in-kernel. 4-bit is the default tier (fits 24 GB); `dit_fp16.onnx` is
  provided for higher precision.
- Wan2.1-T2V-14B sampling defaults: 50 steps, shift 5.0, guidance 5.0, 16 fps.

## Setup

```bash
git clone https://github.com/AkaneTendo25/wannx && cd wannx
python -m venv .venv
# Windows: .venv\Scripts\activate   Linux: source .venv/bin/activate
pip install -r requirements.txt              # inference (ONNX Runtime, torch-free)
# pip install -r requirements-convert.txt    # only if converting checkpoints -> ONNX
```

Two dependency sets, by task: **`requirements.txt`** runs inference on ONNX Runtime (no torch);
**`requirements-convert.txt`** is for converting a checkpoint to ONNX (CPU/RAM-bound, needs torch +
onnxscript). Most users only need the first. Python 3.10+, NVIDIA GPU + CUDA. The WAN source package is
bundled at `models/wan2.1/wan` (Apache-2.0). On Linux, expose the bundled cuDNN to the ORT CUDA EP:

```bash
export LD_LIBRARY_PATH=$(python -c "import os,glob,nvidia;b=os.path.dirname(nvidia.__file__);print(':'.join(sorted(glob.glob(b+'/*/lib'))))"):$LD_LIBRARY_PATH
```

## Weights

Download the bundle into `./onnx_output`:

```
onnx_output/
  dit.onnx + dit.onnx.data            # 4-bit DiT                ~7.8 GB
  dit_fp16.onnx + dit_fp16.onnx.data  # fp16 DiT                 ~27 GB
  t5_encoder_fp16.onnx + .data        # umt5-xxl encoder         ~13 GB
  vae_decoder.onnx + .meta.json       # VAE step graph (latents 1+)   ~0.3 GB
  vae_decoder_init.onnx               # VAE init graph (latent 0)     ~0.3 GB
  google/umt5-xxl/                    # tokenizer
```

## Inference

### Fully ONNX, 24 GB (recommended)

`run_isolated.py` runs T5 → DiT → VAE in separate subprocesses, so ORT releases GPU memory between
stages and peak VRAM equals the largest stage. With ONNX T5 + the streaming VAE it runs entirely on
ONNX Runtime — no torch, no original checkpoint. Measured: 81 frames at 480×832, 4-bit, ~23.4 GB peak.

```bash
python tools/run_isolated.py \
  --dit onnx_output/dit.onnx \
  --t5-onnx onnx_output/t5_encoder_fp16.onnx \
  --vae-stream onnx_output/vae_decoder.onnx \
  --tokenizer onnx_output/google/umt5-xxl \
  --out cat.mp4 --prompt "a cat walking in a sunlit garden" \
  --num-frames 81 --height 480 --width 832 --steps 50 --shift 5.0 --guidance 5.0 --seed 42
```

The streaming VAE decodes one latent frame at a time with per-layer caches. It uses two graphs: an
`init` graph for latent 0 (the reference decoder's first-latent path — 1 frame, no temporal doubling)
and a `step` graph for latents 1+ (4 frames each). The result is bit-exact to the reference
`WanVAE.decode` (so frame 0 is correct, no warm-up artifact), with 1-frame memory and no cuDNN
tensor-size limit, at any resolution. `vae_decoder_init.onnx` must sit next to `vae_decoder.onnx`.

### Alternative: ONNX DiT + torch T5/VAE

Omit `--t5-onnx`/`--vae-stream` and pass `--checkpoint-dir <Wan2.1-T2V-14B>` to use the torch T5/VAE
from the original checkpoint (needs the `convert` dependencies).

Notes:
- Output is 16 fps (Wan2.1 native). 81 frames = 5.06 s.
- Omitting `--negative-prompt` uses the Wan shared default negative prompt.

## Conversion (optional)

Conversion is CPU/RAM-bound (the 14B DiT export peaks around 110 GB RAM); it does not use the GPU.
Install the convert deps first: `pip install -r requirements-convert.txt`.

```bash
# DiT -> dynamic fp32 ONNX (optimize=True makes linear weights 4-bit-quantizable)
python -m wannx convert -m dit -c <Wan2.1-T2V-14B> -o onnx_master \
    --dynamic-frames --dynamic-resolution --dynamic-text-len --optimize-graph

# 4-bit quantize the DiT
python tools/quantize_q4.py onnx_master/dit.onnx onnx_output/dit.onnx

# T5 encoder -> dynamic ONNX
python -m wannx convert -m t5 -c <Wan2.1-T2V-14B> -o onnx_output --dynamic-text-len --fp16

# VAE -> streaming (recurrent) ONNX; writes vae_decoder.onnx AND vae_decoder_init.onnx
python tools/export_stream_vae.py -c <Wan2.1-T2V-14B> --out onnx_output/vae_decoder.onnx
```

The DiT uses the dynamo exporter with symbolic `dynamic_shapes`; `--optimize-graph` folds
`MatMul(x, Transpose(W))` into constant-weight MatMuls, required for 4-bit quantization. The VAE export
streams the decoder per latent frame with the per-layer caches as ONNX inputs/outputs, emitting two
graphs (`init` for latent 0, `step` for the rest) that together reproduce `WanVAE.decode` exactly.

## Tests

```bash
python tests/test_inference.py        # scheduler, video, ORT session, synthetic pipeline
python tests/test_dynamic_export.py   # one DiT graph, multiple shapes, cosine ~ 1.0
```

## Demo

4-bit DiT, 832×480, 81 frames, 16 fps, 50 steps, shift 5.0, guidance 5.0.
Negative prompt: `blurry, low quality, distorted, deformed, washed out, watermark, text`.

<table>
<tr>
<td width="50%"><video src="https://github.com/user-attachments/assets/0d257d2a-ba0c-4015-a1c5-340aa1fa4902" controls></video><br>cat in a sunlit garden — seed 42</td>
<td width="50%"><video src="https://github.com/user-attachments/assets/c0b71b3f-62c1-4f6c-a404-732bf73d8ad9" controls></video><br>rally car drifting, mountain hairpin — seed 321</td>
</tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/1d6d982e-5a42-4bbf-9491-1f0eabda0e47" controls></video><br>aerial over forest and river, sunrise — seed 888</td>
<td><video src="https://github.com/user-attachments/assets/81b61017-95a0-4768-94ec-4e049a37648b" controls></video><br>woman on a windy cliff, golden hour — seed 4242</td>
</tr>
<tr>
<td><video src="https://github.com/user-attachments/assets/d9efa33f-a872-4de6-be7b-0f9a8a576eb4" controls></video><br>spaceship over a neon city at night — seed 2026</td>
<td></td>
</tr>
</table>

Full prompts:

- `a bengal cat walking through a sunlit garden, shallow depth of field, cinematic, high detail, natural daylight, coherent motion` (seed 42)
- `a red rally car drifting around a mountain hairpin at high speed, tire smoke and dust, dynamic chase camera, golden hour, cinematic, coherent motion` (seed 321)
- `aerial drone shot sweeping over a misty pine forest and a winding river at sunrise, volumetric light shafts, cinematic, coherent motion` (seed 888)
- `cinematic close-up of a young woman on a windy cliff at golden hour, hair moving in the sea breeze, detailed skin, natural expression, subtle handheld camera, coherent motion` (seed 4242)
- `a futuristic spaceship descending through glowing clouds toward a neon-lit city at night, volumetric light, reflections, cinematic sci-fi, coherent motion` (seed 2026)

## License

Uses [Wan2.1](https://github.com/Wan-Video/Wan2.1) (Apache-2.0). The bundled WAN source retains its
license (`models/wan2.1/LICENSE.txt`).
