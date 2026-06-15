"""Runtime patches applied before any WAN module imports.

Handles:
  1. Missing flash_attn  ->  mock it so imports succeed
  2. Replace flash_attention() and attention() with standard SDPA
  3. Replace complex-number RoPE (torch.view_as_complex / torch.polar)
     with fully real-valued rotation (float32)
  4. Replace rope_params to return real angles instead of complex freqs (float32)
  5. Replace sinusoidal_embedding_1d to use float32 (ORT lacks float64 Cos/Sin)
  6. Patch VAE Resample to handle temporal down/upsampling without caching
  7. Patch VAE Upsample nearest-exact -> nearest (no ONNX symbolic for exact)
  8. HuggingFace deprecated API shim
"""

import sys
import logging
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PATCHED = False


def _install_post_import_patch(target_name, patch_fn):
    """Run ``patch_fn(module)`` right after ``target_name`` is first imported.

    Uses the modern importlib ``find_spec`` meta-path protocol. The legacy
    ``find_module``/``load_module`` protocol that this module relied on before
    was removed in Python 3.12, which silently disabled every WAN patch there
    (real-valued RoPE, SDPA attention, float32 sinusoidal, VAE fixes). If the
    target module is already imported, patch it immediately.
    """
    import importlib
    import importlib.abc
    import importlib.util

    if target_name in sys.modules:
        patch_fn(sys.modules[target_name])
        return

    class _PostImportFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != target_name:
                return None
            # One-shot: drop ourselves so the real finders resolve the spec
            # (and so the find_spec call below does not recurse into us).
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            real_spec = importlib.util.find_spec(fullname)
            if real_spec is None or real_spec.loader is None:
                return None
            real_loader = real_spec.loader

            class _PatchingLoader(importlib.abc.Loader):
                def create_module(self, spec):
                    if hasattr(real_loader, "create_module"):
                        return real_loader.create_module(spec)
                    return None

                def exec_module(self, module):
                    real_loader.exec_module(module)
                    patch_fn(module)

            real_spec.loader = _PatchingLoader()
            return real_spec

    sys.meta_path.insert(0, _PostImportFinder())


# ====================================================================== #
# 1. Flash-attention replacement                                          #
# ====================================================================== #

def _make_flash_attn_replacement():
    """Drop-in for wan.modules.attention.flash_attention.

    Real signature:
        flash_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
                        q_scale, causal, window_size, deterministic, dtype, version)
    q/k/v: [B, L, N, C].
    """

    def replacement(
        q, k, v,
        q_lens=None, k_lens=None,
        dropout_p=0.0, softmax_scale=None,
        q_scale=None, causal=False,
        window_size=(-1, -1), deterministic=False,
        dtype=torch.bfloat16, version=None,
        fa_version=None,  # accepted by attention() wrapper
    ):
        out_dtype = q.dtype

        def _resolve_block_size(ws):
            if ws is None:
                return 0
            if isinstance(ws, (tuple, list)):
                vals = [int(x) for x in ws if int(x) > 0]
                if not vals:
                    return 0
                # For our ONNX export path we treat equal positive values as an
                # explicit block size instead of FA's left/right radius.
                if len(vals) >= 2 and vals[0] == vals[1]:
                    return vals[0]
                return max(vals)
            value = int(ws)
            return value if value > 0 else 0

        def _block_local_attention(q, k, v, block_size: int):
            # q/k/v are [B, H, L, D]. Compute attention independently inside
            # fixed sequence blocks to cap memory growth for long videos.
            b, h, seq, d = q.shape
            if block_size <= 0 or block_size >= seq:
                return F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=causal, scale=softmax_scale
                )

            pad = (block_size - (seq % block_size)) % block_size
            if pad:
                zeros = lambda t: torch.zeros(
                    (b, h, pad, d), dtype=t.dtype, device=t.device
                )
                q = torch.cat([q, zeros(q)], dim=2)
                k = torch.cat([k, zeros(k)], dim=2)
                v = torch.cat([v, zeros(v)], dim=2)

            seq_pad = q.shape[2]
            num_blocks = seq_pad // block_size

            def _pack_blocks(t):
                return (
                    t.view(b, h, num_blocks, block_size, d)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(b * num_blocks, h, block_size, d)
                )

            q_blk = _pack_blocks(q)
            k_blk = _pack_blocks(k)
            v_blk = _pack_blocks(v)

            out = F.scaled_dot_product_attention(
                q_blk, k_blk, v_blk, dropout_p=0.0, is_causal=causal, scale=softmax_scale
            )
            out = (
                out.view(b, num_blocks, h, block_size, d)
                .permute(0, 2, 1, 3, 4)
                .reshape(b, h, seq_pad, d)
            )
            if pad:
                out = out[:, :, :seq, :]
            return out

        if q_scale is not None:
            q = q * q_scale

        # [B, L, N, C] -> [B, N, L, C]
        q = q.permute(0, 2, 1, 3).float()
        k = k.permute(0, 2, 1, 3).float()
        v = v.permute(0, 2, 1, 3).float()

        block_size = _resolve_block_size(window_size)
        out = _block_local_attention(q, k, v, block_size)
        return out.permute(0, 2, 1, 3).to(out_dtype)

    return replacement


# ====================================================================== #
# 2. Real-valued RoPE (no complex numbers)                                #
# ====================================================================== #

def _make_rope_params_replacement():
    """Replace rope_params to return REAL angle tensor instead of complex.

    Original returns torch.polar(ones, angles) which is complex64 (from float64).
    We return just the angles as float32 — the original uses float64 for precision
    during angle computation, but ORT doesn't implement Cos/Sin for float64, and
    float32 precision is sufficient for the final cos/sin rotation.
    """

    @torch.amp.autocast('cuda', enabled=False)
    def rope_params(max_seq_len, dim, theta=10000):
        assert dim % 2 == 0
        # Compute in float64 for precision, then cast to float32 for ONNX compat
        freqs = torch.outer(
            torch.arange(max_seq_len),
            1.0 / torch.pow(
                theta,
                torch.arange(0, dim, 2).to(torch.float64).div(dim)))
        return freqs.float()  # [max_seq_len, dim//2] float32

    return rope_params


def _make_rope_apply_replacement():
    """Replace rope_apply to work with real angle tensors.

    The original uses torch.view_as_complex / torch.view_as_real.
    We use cos/sin rotation on even/odd pairs.
    """

    @torch.amp.autocast('cuda', enabled=False)
    def rope_apply(x, grid_sizes, freqs):
        """
        x:          [B, S, N, C]       query or key
        grid_sizes: [B, 3]             (F, H, W per sample)
        freqs:      [max_len, C//2]    REAL angles (not complex)
        Returns:    [B, S, N, C]
        """
        c = x.size(3) // 2
        freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # grid_sizes is either a tuple of symbolic sizes (f, h, w) from the ONNX
        # export wrapper, or a [B, 3] long tensor (eager / upstream WanModel path).
        if isinstance(grid_sizes, (tuple, list)):
            # --- Export / wrapper path (batch=1, no token padding) ---------------
            # S == f*h*w exactly, so build the per-token angles by slicing the
            # freqs table to [:f]/[:h]/[:w] and broadcasting over the f-major grid.
            # This uses only Slice/Expand/Reshape (all dynamic-shape friendly) and
            # avoids remainder/div by a *symbolic scalar*, which the ONNX exporter
            # cannot lower (aten_remainder_scalar casts the divisor with int()).
            f, h, w = grid_sizes
            af = freqs_split[0][:f][:, None, None, :].expand(f, h, w, -1)
            ah = freqs_split[1][:h][None, :, None, :].expand(f, h, w, -1)
            aw = freqs_split[2][:w][None, None, :, :].expand(f, h, w, -1)
            angles = torch.cat([af, ah, aw], dim=-1).reshape(f * h * w, c).float()
            angles = angles.unsqueeze(0).unsqueeze(2)  # [1, S, 1, C//2]

            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            x_float = x.float()
            x_even = x_float[..., 0::2]
            x_odd = x_float[..., 1::2]
            out_even = x_even * cos_a - x_odd * sin_a
            out_odd = x_even * sin_a + x_odd * cos_a
            out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
            return out.to(x.dtype)

        # --- Eager / upstream tensor path (supports a padded token tail) --------
        f = grid_sizes[0, 0].to(torch.long)
        h = grid_sizes[0, 1].to(torch.long)
        w = grid_sizes[0, 2].to(torch.long)
        seq_len = f * h * w

        seq_total = x.shape[1]
        idx = torch.arange(seq_total, device=x.device, dtype=torch.long)
        hw = h * w

        f_idx = torch.div(idx, hw, rounding_mode='floor')
        rem = torch.remainder(idx, hw)
        h_idx = torch.div(rem, w, rounding_mode='floor')
        w_idx = torch.remainder(rem, w)

        # Clamp indices to valid range; tail tokens (padding) are masked later.
        f_max = torch.zeros_like(f_idx) + (f - 1)
        h_max = torch.zeros_like(h_idx) + (h - 1)
        w_max = torch.zeros_like(w_idx) + (w - 1)
        f_idx = torch.minimum(torch.maximum(f_idx, torch.zeros_like(f_idx)), f_max)
        h_idx = torch.minimum(torch.maximum(h_idx, torch.zeros_like(h_idx)), h_max)
        w_idx = torch.minimum(torch.maximum(w_idx, torch.zeros_like(w_idx)), w_max)

        freq_f = freqs_split[0].index_select(0, f_idx)
        freq_h = freqs_split[1].index_select(0, h_idx)
        freq_w = freqs_split[2].index_select(0, w_idx)
        angles = torch.cat([freq_f, freq_h, freq_w], dim=-1).float()  # [S, C//2]
        angles = angles.unsqueeze(0).unsqueeze(2)  # [1, S, 1, C//2]

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        x_float = x.float()
        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]

        out_even = x_even * cos_a - x_odd * sin_a
        out_odd = x_even * sin_a + x_odd * cos_a
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)

        # Keep padded tail untouched when seq_total > seq_len.
        valid = (idx < seq_len).view(1, -1, 1, 1)
        out = torch.where(valid, out, x_float)
        return out.to(x.dtype)

    return rope_apply


# ====================================================================== #
# 3. Patch VAE Upsample for ONNX compatibility                           #
# ====================================================================== #

def _patch_vae_upsample():
    """Replace 'nearest-exact' mode with 'nearest' in VAE's Upsample.

    PyTorch's nn.Upsample with mode='nearest-exact' maps to
    aten::_upsample_nearest_exact2d which has no ONNX exporter symbolic.
    We patch the Upsample subclass to use 'nearest' instead, which exports
    cleanly.  The difference is sub-pixel alignment (negligible for 2x upscale).
    """
    from wan.modules.vae import Upsample
    original_init = Upsample.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.mode == 'nearest-exact':
            self.mode = 'nearest'

    Upsample.__init__ = patched_init
    logger.info("Patched wan.modules.vae.Upsample: nearest-exact -> nearest")


# ====================================================================== #
# 4. VAE Resample patch for non-cached temporal operations                #
# ====================================================================== #

def _patch_vae_resample():
    """Monkey-patch wan.modules.vae.Resample.forward to handle temporal
    down/upsampling WITHOUT the caching mechanism.

    The original Resample.forward skips time_conv when feat_cache is None,
    which means temporal downsampling/upsampling never happens in single-pass
    mode. This patch adds a direct non-cached path.
    """
    try:
        from einops import rearrange
    except ImportError:
        logger.warning("einops not available - VAE Resample patch skipped")
        return

    from wan.modules.vae import Resample

    original_forward = Resample.forward

    def patched_forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            # Use original caching path
            return original_forward(self, x, feat_cache, feat_idx)

        b, c, t, h, w = x.size()

        if self.mode == 'upsample3d':
            # Apply temporal upsampling directly (no cache)
            x_time = self.time_conv(x)  # CausalConv3d: [B, 2*C, T, H, W]
            x_time = x_time.reshape(b, 2, c, t, h, w)
            x_time = torch.stack(
                (x_time[:, 0, :, :, :, :], x_time[:, 1, :, :, :, :]), 3)
            x = x_time.reshape(b, c, t * 2, h, w)
            t = x.shape[2]

        # Spatial resampling (always runs)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            # Apply temporal downsampling directly (no cache).
            # time_conv has kernel=(3,1,1), stride=(2,1,1), padding=(0,0,0).
            # In streaming mode, feat_cache provides temporal context.
            # Without cache, we add causal zero-padding (kernel_size-1 = 2 frames).
            x = F.pad(x, (0, 0, 0, 0, 2, 0))
            x = self.time_conv(x)

        return x

    Resample.forward = patched_forward
    logger.info("Patched wan.modules.vae.Resample.forward for non-cached mode")


def _patch_wan_dtype_asserts():
    """Relax hard float32 asserts in Wan blocks for export-friendly fp16 paths.

    The validated runtime already feeds float32 modulation embeddings, so this
    is a semantic no-op there. For direct-fp16 export experiments, however, the
    upstream asserts abort before the block can autocast back to float32 math.
    """
    from wan.modules.model import WanAttentionBlock, Head
    import torch.cuda.amp as amp

    def patched_block_forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        # Modulation / norm math is done in float32 for numerical stability, but
        # the result is cast back to the model dtype BEFORE each sub-module and
        # the residual stream is kept in the model dtype. For fp32 models every
        # .float()/.to(dt) is a no-op (parity preserved); for native-fp16 export
        # this keeps the whole graph a single dtype (no mixed fp32/fp16 MatMuls).
        dt = x.dtype
        e = (self.modulation.float() + e.float()).chunk(6, dim=1)

        y = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).to(dt),
            seq_lens, grid_sizes, freqs,
        )
        x = (x.float() + y.float() * e[2]).to(dt)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dt))
        x = (x.float() + y.float() * e[5]).to(dt)
        return x

    def patched_head_forward(self, x, e):
        dt = x.dtype
        e = (self.modulation.float() + e.float().unsqueeze(1)).chunk(2, dim=1)
        x = self.head((self.norm(x).float() * (1 + e[1]) + e[0]).to(dt))
        return x

    WanAttentionBlock.forward = patched_block_forward
    Head.forward = patched_head_forward
    logger.info("Patched WanAttentionBlock/Head dtype asserts for export-friendly fp16 paths")


# ====================================================================== #
# Main entry point                                                         #
# ====================================================================== #

def apply_patches():
    """Apply all runtime patches.  Safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # ------------------------------------------------------------------ #
    # 0.  Mock torch.cuda.current_device on CPU-only systems               #
    #     (T5EncoderModel uses it as a default param, evaluated at         #
    #      class definition / import time)                                 #
    # ------------------------------------------------------------------ #
    if not torch.cuda.is_available():
        _orig_current_device = torch.cuda.current_device
        torch.cuda.current_device = lambda: 0  # never actually used
        logger.debug("Mocked torch.cuda.current_device for CPU-only system")

    # ------------------------------------------------------------------ #
    # 1.  Mock flash_attn package so imports don't fail                    #
    # ------------------------------------------------------------------ #
    for mod_name in (
        "flash_attn",
        "flash_attn.flash_attn_interface",
        "flash_attn_interface",
    ):
        if mod_name not in sys.modules:
            import importlib
            mock = MagicMock()
            mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
            sys.modules[mod_name] = mock
            logger.debug("Mocked missing module: %s", mod_name)

    # ------------------------------------------------------------------ #
    # 2.  Patch wan.modules.attention after import                         #
    # ------------------------------------------------------------------ #
    replacement_fn = _make_flash_attn_replacement()

    def _patch_attention_module(mod):
        mod.flash_attention = replacement_fn
        mod.attention = replacement_fn
        mod.FLASH_ATTN_2_AVAILABLE = False
        mod.FLASH_ATTN_3_AVAILABLE = False
        logger.info("Patched wan.modules.attention -> SDPA replacement")

    _install_post_import_patch("wan.modules.attention", _patch_attention_module)

    # ------------------------------------------------------------------ #
    # 3.  Patch rope_params + rope_apply to avoid complex numbers          #
    # ------------------------------------------------------------------ #
    rope_params_fn = _make_rope_params_replacement()
    rope_apply_fn = _make_rope_apply_replacement()

    def _sinusoidal_embedding_1d_f32(dim, position):
        """sinusoidal_embedding_1d using float32 instead of float64.

        ORT doesn't implement Cos/Sin for float64.  The original uses float64
        for precision but float32 is more than sufficient for timestep embeddings.
        """
        assert dim % 2 == 0
        half = dim // 2
        position = position.float()
        device = position.device
        sinusoid = torch.outer(
            position,
            torch.pow(10000, -torch.arange(half, device=device).float().div(half)))
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

    def _patch_model_module(mod):
        mod.rope_params = rope_params_fn
        mod.rope_apply = rope_apply_fn
        mod.sinusoidal_embedding_1d = _sinusoidal_embedding_1d_f32
        _patch_wan_dtype_asserts()
        logger.info("Patched wan.modules.model.rope_params -> real-valued")
        logger.info("Patched wan.modules.model.rope_apply -> real-valued rotation")
        logger.info("Patched wan.modules.model.sinusoidal_embedding_1d -> float32")

    _install_post_import_patch("wan.modules.model", _patch_model_module)

    # ------------------------------------------------------------------ #
    # 4.  Patch VAE Resample for non-cached temporal operations            #
    #     (must happen AFTER wan.modules.vae is imported, so we use        #
    #     a meta-path finder)                                              #
    # ------------------------------------------------------------------ #
    def _patch_vae_module(mod):
        # Module is loaded; patch Resample and Upsample.
        _patch_vae_resample()
        _patch_vae_upsample()

    _install_post_import_patch("wan.modules.vae", _patch_vae_module)

    # ------------------------------------------------------------------ #
    # 5.  HuggingFace deprecated API shim                                  #
    # ------------------------------------------------------------------ #
    try:
        import huggingface_hub
        if not hasattr(huggingface_hub, "cached_download"):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    except ImportError:
        pass

    # ------------------------------------------------------------------ #
    # 6.  Log CUDA status                                                  #
    # ------------------------------------------------------------------ #
    if not torch.cuda.is_available():
        logger.info("CUDA not available - all operations will run on CPU")

    logger.info("All runtime patches applied.")
