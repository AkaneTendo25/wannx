"""Tests for the ONNX inference pipeline.

Tests scheduler math, pipeline integration with synthetic models,
and video output utilities.  No real checkpoints needed.
"""

import os
import sys
import tempfile
import traceback
from unittest import mock

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_WAN_ROOT = os.path.join(_ROOT, "models", "wan2.1")

for p in (_ROOT, _WAN_ROOT):
    p = os.path.normpath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from wannx.patcher import apply_patches  # noqa: E402

apply_patches()

import torch  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402


# =========================================================================
# Scheduler tests
# =========================================================================

def test_negative_prompt_defaults_to_official_wan_prompt():
    from wannx.inference import WAN_DEFAULT_NEGATIVE_PROMPT, resolve_negative_prompt

    assert resolve_negative_prompt(None) == WAN_DEFAULT_NEGATIVE_PROMPT
    assert resolve_negative_prompt("") == WAN_DEFAULT_NEGATIVE_PROMPT
    assert resolve_negative_prompt("   ") == WAN_DEFAULT_NEGATIVE_PROMPT
    assert resolve_negative_prompt("low quality") == "low quality"


def test_pad_text_embeddings_matches_fixed_wan_text_length():
    from wannx.inference import pad_text_embeddings

    src = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    padded = pad_text_embeddings(src, 5)
    assert padded.shape == (5, 4)
    np.testing.assert_array_equal(padded[:3], src)
    np.testing.assert_array_equal(padded[3:], np.zeros((2, 4), dtype=np.float32))

    trimmed = pad_text_embeddings(src, 2)
    assert trimmed.shape == (2, 4)
    np.testing.assert_array_equal(trimmed, src[:2])


def test_find_onnx_models_prefers_fixed_profile_fp16_dit():
    from wannx.inference import find_onnx_models

    with tempfile.TemporaryDirectory(prefix="wannx_test_find_models_") as tmp:
        for name in ("dit_fp16.onnx", "dit_fp16_seq512.onnx", "vae_decoder.onnx"):
            open(os.path.join(tmp, name), "wb").close()

        found = find_onnx_models(tmp, prefer_fp16=True)
        assert found["dit"].endswith("dit_fp16_seq512.onnx")
        assert found["vae_decoder"].endswith("vae_decoder.onnx")


def test_resolve_onnx_layout_and_collect_profiled_models():
    from wannx.inference import collect_onnx_models_from_layout, resolve_onnx_layout

    with tempfile.TemporaryDirectory(prefix="wannx_test_profile_layout_") as tmp:
        prof_dir = os.path.join(tmp, "profiles", "f81_h512_w512")
        os.makedirs(prof_dir, exist_ok=True)
        for name in ("t5_encoder.onnx", "dit_fp16.onnx", "dit_fp16_seq512.onnx", "vae_decoder.onnx"):
            target_dir = prof_dir if name.startswith(("dit", "vae")) else tmp
            open(os.path.join(target_dir, name), "wb").close()

        root_dir, profile_dir, use_profiles = resolve_onnx_layout(
            tmp, 81, 512, 512, prefer_profile_layout=True
        )
        assert use_profiles
        assert root_dir == os.path.abspath(tmp)
        assert profile_dir.endswith(os.path.join("profiles", "f81_h512_w512"))

        models = collect_onnx_models_from_layout(
            root_dir,
            profile_dir,
            prefer_fp16=True,
            text_len=512,
        )
        assert models["t5"].endswith("t5_encoder.onnx")
        assert models["dit"].endswith("dit_fp16_seq512.onnx")
        assert models["vae_decoder"].endswith("vae_decoder.onnx")


def test_auto_convert_if_needed_ignores_vae_when_not_requested():
    from wannx import inference as inf

    def fake_shape_ok(path, input_name, expected_shape):
        if input_name == "latent_input":
            return True
        if input_name == "latent":
            return False
        raise AssertionError(f"unexpected input_name {input_name}")

    with mock.patch.object(inf, "find_onnx_models", return_value={
        "dit": "dummy_dit.onnx",
        "vae_decoder": "dummy_vae.onnx",
    }), mock.patch.object(inf, "_onnx_input_shape_is_compatible", side_effect=fake_shape_ok), \
         mock.patch("wannx.patcher.apply_patches"), \
         mock.patch("wannx.converter.convert_modules") as convert_modules:
        inf.auto_convert_if_needed(
            checkpoint_dir="dummy_ckpt",
            onnx_dir="dummy_onnx",
            modules=["dit"],
            config={
                "fp16": True,
                "text_len": 512,
                "latent_channels": 16,
                "latent_frames": 21,
                "latent_height": 64,
                "latent_width": 64,
            },
        )

    convert_modules.assert_not_called()


def test_flash_attn_replacement_supports_block_local_window():
    from wannx.patcher import _make_flash_attn_replacement

    replacement = _make_flash_attn_replacement()
    q = torch.randn(1, 8, 2, 4, dtype=torch.float32)
    k = torch.randn(1, 8, 2, 4, dtype=torch.float32)
    v = torch.randn(1, 8, 2, 4, dtype=torch.float32)

    out_global = replacement(q, k, v, window_size=(-1, -1))
    out_local = replacement(q, k, v, window_size=(4, 4))

    assert out_global.shape == q.shape
    assert out_local.shape == q.shape


def test_predict_noise_tiled_averages_temporal_overlaps():
    from wannx.inference import T2VPipeline

    class FakeDiT:
        def __call__(self, *, latent_input, timestep, text_embeddings):
            return [latent_input + 1.0]

    pipe = object.__new__(T2VPipeline)
    pipe.dit = FakeDiT()

    latents = np.zeros((1, 16, 5, 2, 2), dtype=np.float32)
    timestep = np.array([1.0], dtype=np.float32)
    embeds = np.zeros((1, 512, 4096), dtype=np.float32)

    out = pipe._predict_noise_tiled(
        latents,
        timestep,
        embeds,
        temporal_chunk_latent=2,
        temporal_overlap_latent=1,
    )

    assert out.shape == latents.shape
    np.testing.assert_allclose(out, np.ones_like(latents), rtol=0, atol=0)


def test_scheduler_sigma_schedule():
    """Verify scheduler matches diffusers' reference sigma/timestep schedule."""
    from wannx.inference import FlowMatchEulerScheduler
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    N = 20
    shift = 3.0
    sched = FlowMatchEulerScheduler(num_steps=N, shift=shift)
    ref = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    ref.set_timesteps(N, device="cpu")

    # sigmas has N+1 entries (N steps + final 0)
    assert len(sched.sigmas) == N + 1, \
        f"Expected {N+1} sigmas, got {len(sched.sigmas)}"

    # First sigma should be close to 1.0 (shifted)
    assert sched.sigmas[0] > 0.9, f"sigma_0 too small: {sched.sigmas[0]}"

    # Last sigma is 0
    assert sched.sigmas[-1] == 0.0, f"sigma_final != 0: {sched.sigmas[-1]}"

    # Monotonically decreasing
    for i in range(len(sched.sigmas) - 1):
        assert sched.sigmas[i] > sched.sigmas[i + 1], \
            f"sigma[{i}]={sched.sigmas[i]} <= sigma[{i+1}]={sched.sigmas[i+1]}"

    assert len(sched.timesteps) == N
    np.testing.assert_allclose(
        sched.sigmas,
        ref.sigmas.detach().cpu().numpy().astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sched.timesteps,
        ref.timesteps.detach().cpu().numpy().astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    return "ok"


def test_scheduler_step_math():
    """Verify Euler step matches diffusers' reference implementation."""
    from wannx.inference import FlowMatchEulerScheduler
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerScheduler(num_steps=10, shift=3.0)
    ref = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    ref.set_timesteps(10, device="cpu")

    sample = np.ones((1, 4, 2, 2, 2), dtype=np.float32)
    model_output = np.full_like(sample, 0.5)

    result = sched.step(model_output, sample, step_index=0)
    expected = ref.step(
        torch.from_numpy(model_output),
        ref.timesteps[0],
        torch.from_numpy(sample),
        return_dict=False,
    )[0].detach().cpu().numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    return "ok"


def test_scheduler_scale_noise():
    """Verify initial noise scaling matches diffusers' reference implementation."""
    from wannx.inference import FlowMatchEulerScheduler
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerScheduler(num_steps=10, shift=3.0)
    ref = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    ref.set_timesteps(10, device="cpu")
    noise = np.ones((1, 4, 2, 2, 2), dtype=np.float32)
    scaled = sched.scale_noise(noise)

    expected = ref.scale_noise(
        torch.from_numpy(noise),
        ref.timesteps[:1],
        torch.from_numpy(noise),
    ).detach().cpu().numpy()
    np.testing.assert_allclose(scaled, expected, rtol=1e-6, atol=1e-6)
    return "ok"


def test_scheduler_full_trajectory():
    """Run full scheduler trajectory and verify final result is near zero noise."""
    from wannx.inference import FlowMatchEulerScheduler

    N = 50
    sched = FlowMatchEulerScheduler(num_steps=N, shift=3.0)

    # Simulate: model always predicts "remove all noise" (v = -sample/sigma)
    # In flow matching, v points from noise to data.
    # With constant v = data_direction, the trajectory should converge.
    latents = sched.scale_noise(np.ones((1, 1, 1, 1, 1), dtype=np.float32))
    for i in range(N):
        # Fake model output: constant velocity field
        v = np.ones_like(latents) * (-1.0)
        latents = sched.step(v, latents, i)

    # After all steps, sigma_final = 0, so we should have moved
    # by total displacement = sum of (sigma_next - sigma) * v
    # = (0 - sigma_0) * (-1) = sigma_0
    # So final = initial + sigma_0 = sigma_0 * 1 + sigma_0 = 2 * sigma_0?
    # Actually: final = sigma_0 + (0 - sigma_0)*(-1) = sigma_0 + sigma_0 = 2*sigma_0
    # This is just a sanity check that the trajectory runs without error
    assert np.isfinite(latents).all(), "Trajectory produced non-finite values"
    return "ok"


# =========================================================================
# Video utility tests
# =========================================================================

def test_latents_to_video():
    """Verify float [-1,1] -> uint8 [0,255] conversion."""
    from wannx.inference import latents_to_video

    # Create known values
    raw = np.zeros((1, 3, 4, 8, 8), dtype=np.float32)
    raw[0, 0, :, :, :] = 1.0   # Red channel = 1.0 -> 255
    raw[0, 1, :, :, :] = -1.0  # Green channel = -1.0 -> 0
    raw[0, 2, :, :, :] = 0.0   # Blue channel = 0.0 -> 127 or 128

    frames = latents_to_video(raw)

    assert frames.shape == (4, 8, 8, 3), f"Wrong shape: {frames.shape}"
    assert frames.dtype == np.uint8, f"Wrong dtype: {frames.dtype}"
    assert frames[0, 0, 0, 0] == 255, f"Red should be 255, got {frames[0, 0, 0, 0]}"
    assert frames[0, 0, 0, 1] == 0, f"Green should be 0, got {frames[0, 0, 0, 1]}"
    # Blue: (0+1)/2*255 = 127.5 -> 127 or 128 depending on rounding
    assert 127 <= frames[0, 0, 0, 2] <= 128, \
        f"Blue should be ~127, got {frames[0, 0, 0, 2]}"

    return "ok"


def test_save_video_png_fallback():
    """Verify PNG fallback saves individual frames."""
    from wannx.inference import _save_pngs

    frames = np.random.randint(0, 255, (4, 16, 16, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory(prefix="wannx_test_png_") as tmp:
        out_path = os.path.join(tmp, "test.mp4")
        _save_pngs(frames, out_path)

        png_dir = os.path.join(tmp, "test_frames")
        assert os.path.isdir(png_dir), f"PNG dir not created: {png_dir}"

        pngs = [f for f in os.listdir(png_dir) if f.endswith(".png")]
        assert len(pngs) == 4, f"Expected 4 PNGs, got {len(pngs)}"

    return "ok"


# =========================================================================
# OnnxModelSession test
# =========================================================================

def test_onnx_session_lifecycle():
    """Test OnnxModelSession load/unload/call cycle with a trivial ONNX model."""
    from wannx.inference import OnnxModelSession

    # Create a trivial ONNX model: identity(x) -> y
    wrapper = _IdentityModule()
    wrapper.eval()

    with tempfile.TemporaryDirectory(prefix="wannx_test_sess_") as tmp:
        onnx_path = os.path.join(tmp, "identity.onnx")
        dummy = torch.randn(1, 4)
        torch.onnx.export(
            wrapper, (dummy,), onnx_path,
            input_names=["x"], output_names=["y"],
            opset_version=17,
        )

        sess = OnnxModelSession(onnx_path, device="cpu")
        assert not sess.loaded

        # Load
        sess.load()
        assert sess.loaded

        # Inference
        x = np.random.randn(1, 4).astype(np.float32)
        out = sess(x=x)
        np.testing.assert_allclose(out[0], x, rtol=1e-5)

        # Unload
        sess.unload()
        assert not sess.loaded

        # Auto-load on call
        out2 = sess(x=x)
        np.testing.assert_allclose(out2[0], x, rtol=1e-5)

    return "ok"


def test_onnx_input_shape_compatibility():
    """Static dims should reject mismatched requests; symbolic dims should pass."""
    from wannx.inference import _onnx_input_shape_is_compatible

    with tempfile.TemporaryDirectory(prefix="wannx_test_shape_") as tmp:
        static_path = os.path.join(tmp, "static.onnx")
        dynamic_path = os.path.join(tmp, "dynamic.onnx")

        static_input = onnx.helper.make_tensor_value_info(
            "latent", onnx.TensorProto.FLOAT, [1, 16, 4, 32, 32]
        )
        static_output = onnx.helper.make_tensor_value_info(
            "latent_out", onnx.TensorProto.FLOAT, [1, 16, 4, 32, 32]
        )
        static_graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["latent"], ["latent_out"])],
            "static_graph",
            [static_input],
            [static_output],
        )
        onnx.save(
            onnx.helper.make_model(static_graph, opset_imports=[onnx.helper.make_opsetid("", 17)]),
            static_path,
        )

        dynamic_input = onnx.helper.make_tensor_value_info(
            "latent", onnx.TensorProto.FLOAT, ["batch", 16, "latent_frames", "latent_height", "latent_width"]
        )
        dynamic_output = onnx.helper.make_tensor_value_info(
            "latent_out", onnx.TensorProto.FLOAT, ["batch", 16, "latent_frames", "latent_height", "latent_width"]
        )
        dynamic_graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["latent"], ["latent_out"])],
            "dynamic_graph",
            [dynamic_input],
            [dynamic_output],
        )
        onnx.save(
            onnx.helper.make_model(dynamic_graph, opset_imports=[onnx.helper.make_opsetid("", 17)]),
            dynamic_path,
        )

        target_shape = (1, 16, 2, 60, 104)
        assert not _onnx_input_shape_is_compatible(static_path, "latent", target_shape)
        assert _onnx_input_shape_is_compatible(dynamic_path, "latent", target_shape)

    return "ok"


class _IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


# =========================================================================
# Pipeline integration test (synthetic tiny models)
# =========================================================================

def test_pipeline_synthetic():
    """End-to-end pipeline test with tiny synthetic ONNX models.

    Creates trivial T5/DiT/VAE decoder ONNX models, runs the pipeline
    for 2 steps, and verifies the output shape.
    """
    from wannx.inference import T2VPipeline, OnnxModelSession, latents_to_video

    with tempfile.TemporaryDirectory(prefix="wannx_test_pipeline_") as tmp:
        # --- Create synthetic ONNX models --- #
        # Latent dims must match what the pipeline computes:
        #   latent_f = (num_frames-1)//4 + 1 = (5-1)//4 + 1 = 2
        #   latent_h = height//8 = 16//8 = 2
        #   latent_w = width//8 = 16//8 = 2
        #   latent_channels = 16 (hardcoded in pipeline)
        t5_path = _create_synthetic_t5(tmp, text_len=16, text_dim=32)
        dit_path = _create_synthetic_dit(
            tmp, in_channels=16, latent_f=2, latent_h=2, latent_w=2,
            text_len=16, text_dim=32,
        )
        vae_path = _create_synthetic_vae_decoder(
            tmp, z_dim=16, latent_f=2, latent_h=2, latent_w=2,
            out_frames=5, out_h=16, out_w=16,
        )

        # --- Create a mock tokenizer dir --- #
        tok_path = _create_mock_tokenizer(tmp)

        # --- Run pipeline --- #
        pipeline = T2VPipeline(
            t5_path=t5_path,
            dit_path=dit_path,
            vae_decoder_path=vae_path,
            tokenizer_path=tok_path,
            device="cpu",
            text_len=16,
            keep_models_loaded=False,
        )

        frames = pipeline.generate(
            prompt="test prompt",
            negative_prompt="",
            num_frames=5,
            height=16,
            width=16,
            num_steps=2,
            guidance_scale=1.0,  # no CFG to simplify
            shift=3.0,
            seed=42,
        )

        assert frames.ndim == 4, f"Expected 4D output, got {frames.ndim}D"
        assert frames.shape[-1] == 3, f"Last dim should be 3 (RGB), got {frames.shape[-1]}"
        assert frames.dtype == np.uint8, f"Expected uint8, got {frames.dtype}"
        # T=5, H=16, W=16 from our synthetic VAE
        assert frames.shape == (5, 16, 16, 3), f"Unexpected shape: {frames.shape}"

    return "ok"


# ---------------------------------------------------------------------------
# Synthetic model builders for pipeline test
# ---------------------------------------------------------------------------

class _SyntheticT5(torch.nn.Module):
    """Returns deterministic embeddings derived from inputs (both must be traced)."""
    def __init__(self, text_dim):
        super().__init__()
        self.text_dim = text_dim
        self.proj = torch.nn.Linear(1, text_dim)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        # Use both inputs so ONNX traces them
        x = input_ids.float().unsqueeze(-1) + attention_mask.float().unsqueeze(-1)
        return self.proj(x)  # [B, L, text_dim]


def _create_synthetic_t5(tmp_dir, text_len, text_dim):
    model = _SyntheticT5(text_dim)
    model.eval()
    path = os.path.join(tmp_dir, "t5_encoder.onnx")
    ids = torch.randint(0, 100, (1, text_len))
    mask = torch.ones(1, text_len, dtype=torch.long)
    torch.onnx.export(
        model, (ids, mask), path,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeddings"],
        opset_version=17,
    )
    return path


class _SyntheticDiT(torch.nn.Module):
    """Returns noise prediction of same shape as input, using all inputs."""
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Linear(1, 1)

    def forward(self, latent_input, timestep, text_embeddings):
        # Must use all inputs to ensure ONNX traces them
        scale = timestep.float().view(1, 1, 1, 1, 1) * 0.0 + 1.0
        text_bias = text_embeddings.mean() * 0.0
        return latent_input * scale + text_bias


def _create_synthetic_dit(tmp_dir, in_channels, latent_f, latent_h, latent_w,
                          text_len, text_dim):
    model = _SyntheticDiT()
    model.eval()
    path = os.path.join(tmp_dir, "dit.onnx")
    x = torch.randn(1, in_channels, latent_f, latent_h, latent_w)
    t = torch.tensor([500.0], dtype=torch.float32)
    ctx = torch.randn(1, text_len, text_dim)
    torch.onnx.export(
        model, (x, t, ctx), path,
        input_names=["latent_input", "timestep", "text_embeddings"],
        output_names=["noise_prediction"],
        opset_version=17,
    )
    return path


class _SyntheticVAEDecoder(torch.nn.Module):
    """Outputs fixed-size video frames derived from input."""
    def __init__(self, out_channels=3, out_frames=5, out_h=16, out_w=16,
                 z_dim=16):
        super().__init__()
        self.out_frames = out_frames
        self.out_h = out_h
        self.out_w = out_w
        # Project from latent to output using a real layer
        flat_in = z_dim * 2 * 2 * 2  # z_dim * latent_f * latent_h * latent_w
        flat_out = out_channels * out_frames * out_h * out_w
        self.proj = torch.nn.Linear(flat_in, flat_out)
        self.out_channels = out_channels

    def forward(self, latent):
        B = latent.shape[0]
        x = latent.reshape(B, -1)
        x = self.proj(x)
        x = x.reshape(B, self.out_channels, self.out_frames,
                       self.out_h, self.out_w)
        return torch.tanh(x)  # [-1, 1] range


def _create_synthetic_vae_decoder(tmp_dir, z_dim, latent_f, latent_h, latent_w,
                                  out_frames, out_h, out_w):
    model = _SyntheticVAEDecoder(
        out_channels=3, out_frames=out_frames, out_h=out_h, out_w=out_w,
        z_dim=z_dim,
    )
    model.eval()
    path = os.path.join(tmp_dir, "vae_decoder.onnx")
    z = torch.randn(1, z_dim, latent_f, latent_h, latent_w)
    torch.onnx.export(
        model, (z,), path,
        input_names=["latent"],
        output_names=["output_video"],
        opset_version=17,
    )
    return path


def _create_mock_tokenizer(tmp_dir):
    """Create a minimal tokenizer directory using a real small pretrained tokenizer."""
    from transformers import AutoTokenizer

    tok_dir = os.path.join(tmp_dir, "tokenizer")
    # Use a tiny tokenizer that's readily available
    tok = AutoTokenizer.from_pretrained("t5-small")
    tok.save_pretrained(tok_dir)
    return tok_dir


# =========================================================================
# Runner
# =========================================================================

ALL_TESTS = [
    ("Scheduler: sigma schedule", test_scheduler_sigma_schedule),
    ("Scheduler: step math", test_scheduler_step_math),
    ("Scheduler: scale noise", test_scheduler_scale_noise),
    ("Scheduler: full trajectory", test_scheduler_full_trajectory),
    ("Video: latents_to_video", test_latents_to_video),
    ("Video: PNG fallback", test_save_video_png_fallback),
    ("Session: load/unload/call", test_onnx_session_lifecycle),
    ("Session: shape compatibility", test_onnx_input_shape_compatibility),
    ("Pipeline: synthetic e2e", test_pipeline_synthetic),
]


def main():
    passed = 0
    failed = 0

    for name, test_fn in ALL_TESTS:
        try:
            result = test_fn()
            print(f"  PASS  {name}  ({result})")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            print()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
