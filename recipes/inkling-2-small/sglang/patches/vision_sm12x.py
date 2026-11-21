"""Image input on GB10 (SPARK_VISION=1, with the server's --enable-multimodal; off unless set).

Three parts, all only active with SPARK_VISION=1:

1. The vision encoder's attention (sglang.srt.models.inkling2_vision). The engine runs it with flash_attn.cute's
   varlen kernel and refuses every GPU that is not compute capability 9 or 10: on a Spark (12.1) the first image
   request raises "Inkling2 vision attention requires Hopper or Blackwell". On SM12x this hook runs the engine's own
   fallback instead, the per-image torch scaled_dot_product_attention of its CPU branch: the same projections, the
   same rotary embedding (the engine's GPU kernel), the same scale, no causal mask, one attention per image.
   bench/vision_check.py compares it with the engine's CPU branch against a float32 copy: the hook is as close to
   float32 as the engine's own BF16 path. SDPA is limited to its fused kernels (flash, memory-efficient, cuDNN):
   torch's plain "math" kernel would allocate tokens^2 per head, which for a large picture is more memory than a
   node has. The kernel that serves is probed once on the first image and logged.

2. The picture size (sglang.srt.multimodal.inkling2.image_processing). The engine encodes a picture at its native
   resolution: ceil(w/40) x ceil(h/40) prompt tokens and four times as many tokens inside the vision encoder, with
   no limit. The model card says images work best with each side between 40 and 4,096 pixels, and its evaluations
   downsample images whose long side exceeds 4,096 pixels to that limit with the aspect ratio kept. This hook caps
   the long side before encoding (SPARK_VISION_MAX_EDGE, default 2048; 0 = no limit), with Pillow's Lanczos
   filter, and logs each resize. 2,048 rather than 4,096: with part 3, eight 4,096-pixel pictures at once took
   about 4.7 GB of the rank-0 node's memory and eight capped at 2,048 about 0.7 GB, and on the reading tests of
   docs/results.md the model read print 15 px tall as well at 2,048 as at 4,096. SPARK_VISION_MAX_EDGE=4096
   keeps the model card's limit.
   When the node has less than SPARK_VISION_MIN_FREE_GB available (default 3), a new picture is refused with an
   error that says so (HTTP 400), instead of risking the system's out-of-memory killer. A picture that cannot be
   decoded is refused the same way. The engine reads a file on the server when a picture's URL is file://... or a
   bare path; this hook refuses every source that is not a base64 data URL, an http(s) URL or bytes, before
   anything is read, with one fixed message.

3. Memory. A GB10 node shares one memory between the CPU and the GPU, and the rank-0 node, which runs the HTTP
   server, the tokenizer and the encoder, has a few GB to spare while serving. With the engine's own picture path,
   four 4,096 x 4,096 pictures at once took 3.0 GB of it and eight took 4.1 GB (on the test fleet the
   out-of-memory killer acts near 2.5 GB free, and eight left 0.9 GB). The picture's data travels through the engine in three or four copies (the tokenizer's
   tensor, a shared-memory segment, a GPU copy and a CPU copy kept until the prompt is read), and the engine
   carries each patch twice, as the two identical frames of its temporal dimension, which the encoder only
   averages. Four changes, none of which changes a value the model sees:
   - the tokenizer process builds a picture's patches strip by strip, straight into a BF16 tensor, with the
     engine's float32 arithmetic (bench/vision_check.py --cpu-checks compares the two bit for bit), at most two
     pictures at a time, and returns the freed heap to the system afterwards (malloc_trim);
   - the patches travel with one frame instead of two and are widened to the encoder's two frames (a view, no
     copy) just before the encoder: every copy on the way is half the size;
   - the encoder runs the pictures of a batch in groups of at most SPARK_VISION_TOKEN_BUDGET encoder tokens
     (default 45,000, one 4,096 x 4,096 picture); each picture is independent in the encoder, so the output is
     the same;
   - after a large encode (more than 16,384 encoder tokens) the encoder's cached GPU blocks are released, but
     only when the node has less than SPARK_VISION_EMPTY_BELOW_GB available (default 6): a release costs the
     next large picture its allocations.

The encoder is small (a 1,024-wide, 23-layer vision transformer and a projector, 328M parameters, about 0.7 GB in
BF16). Every rank loads a full copy; rank 0 encodes the pictures of a prefill batch and broadcasts the result to
the other ranks.

Random numbers: every rank samples a request's tokens with its own GPU generator, and the ranks stay in step only
while their generators do. Anything that draws random numbers on rank 0 alone (the encoder runs there alone) makes
rank 0 sample other tokens at temperature > 0, end a request at another step than the other ranks, and the fleet
deadlocks in its next collective; greedy requests are not affected. The hook draws none (the kernel probe uses a
constant tensor), and the encoder wrapper puts the GPU generator's offset back to where it was before the encode
if anything moved it, logging that once. bench/vision_check.py checks that an encode leaves the offset alone.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import os
import threading

import torch
import torch.nn.functional as F

_log = logging.getLogger(__name__)

_S = {"installed": False, "processor": False, "calls": 0, "backends": None, "resized": 0, "groups": 0}
MAX_EDGE = int(os.environ.get("SPARK_VISION_MAX_EDGE", "2048") or 0)
MIN_FREE_GB = float(os.environ.get("SPARK_VISION_MIN_FREE_GB", "3") or 0)
TOKEN_BUDGET = int(os.environ.get("SPARK_VISION_TOKEN_BUDGET", "45000") or 0)
EMPTY_CACHE_AFTER = 16384
EMPTY_BELOW_GB = float(os.environ.get("SPARK_VISION_EMPTY_BELOW_GB", "6") or 0)
_LOCK = threading.BoundedSemaphore(2)


def _mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("inf")


def _rng_offset(device):
    """The philox offset of the device's default generator, or None where torch does not expose it."""
    try:
        return torch.cuda.default_generators[device.index if device.index is not None else torch.cuda.current_device()].get_offset()
    except (AttributeError, RuntimeError, IndexError):
        return None


def _rng_restore(device, offset):
    if offset is None:
        return
    gen = torch.cuda.default_generators[device.index if device.index is not None else torch.cuda.current_device()]
    if gen.get_offset() != offset:
        gen.set_offset(offset)
        if not _S.get("rng_logged"):
            _S["rng_logged"] = True
            _log.warning("[vision] the encode moved the GPU generator's offset; put back so that rank 0 samples as the "
                         "other ranks do")


def _trim():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _fused_backends(device):
    """The SDPA backends to allow: the fused ones that run here (probed on a tiny input), never the math kernel."""
    if _S["backends"] is not None:
        return _S["backends"]
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.ones(1, 16, 300, 64, device=device, dtype=torch.bfloat16)  # no random numbers: see the docstring
    ok = []
    for b in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        try:
            with sdpa_kernel([b]):
                F.scaled_dot_product_attention(q, q, q)
            ok.append(b)
        except RuntimeError:
            pass
    _S["backends"] = ok
    _log.warning("[vision] SDPA kernels that run on this GPU: %s%s", [b.name for b in ok],
                 "" if ok else " (none: the unrestricted SDPA is used, large pictures may run out of memory)")
    return ok


def _sdpa_per_image(q, k, v, lengths, scale):
    """[tokens, heads, dim] q, k, v split by image -> [tokens, heads, dim]: torch SDPA, one image at a time."""
    from torch.nn.attention import sdpa_kernel
    backends = _fused_backends(q.device)
    out = []
    for qi, ki, vi in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
        args = (qi.transpose(0, 1).unsqueeze(0), ki.transpose(0, 1).unsqueeze(0), vi.transpose(0, 1).unsqueeze(0))
        if backends:
            with sdpa_kernel(backends):
                o = F.scaled_dot_product_attention(*args, scale=scale)
        else:
            o = F.scaled_dot_product_attention(*args, scale=scale)
        out.append(o.squeeze(0).transpose(0, 1))
    return torch.cat(out)


def install(mod):
    """Wrap Inkling2VisionAttention.forward of the imported sglang.srt.models.inkling2_vision module."""
    if _S["installed"]:
        return
    cls = mod.Inkling2VisionAttention
    stock = cls.forward

    def forward(self, hidden, positions, lengths, cu_seqlens):
        if not hidden.is_cuda or torch.cuda.get_device_capability(hidden.device)[0] in (9, 10):
            return stock(self, hidden, positions, lengths, cu_seqlens)
        shape = (-1, self.num_heads, self.head_dim)
        q, k, v = (proj(hidden).reshape(shape) for proj in (self.q_proj, self.k_proj, self.v_proj))
        q, k = mod._rotary(q, k, positions)
        output = _sdpa_per_image(q, k, v, list(lengths), self.head_dim ** -0.5)
        if _S["calls"] == 0:
            _log.warning("[vision] first image encoded on SM12x with per-image SDPA (%d image(s), %d tokens)",
                         len(lengths), int(hidden.shape[0]))
        _S["calls"] += 1
        return self.out_proj(output.reshape(hidden.shape))

    forward.__wrapped__ = stock
    cls.forward = forward

    enc = mod.Inkling2VisionEncoder
    enc_stock = enc.forward

    def encode(self, vision_patches_bthwc, vision_grid_thw):
        device = next(self.parameters()).device
        offset = _rng_offset(device) if device.type == "cuda" else None
        try:
            return _encode_groups(self, vision_patches_bthwc, vision_grid_thw)
        finally:
            if device.type == "cuda":
                _rng_restore(device, offset)

    def _encode_groups(self, vision_patches_bthwc, vision_grid_thw):
        if vision_patches_bthwc.ndim == 5 and vision_patches_bthwc.shape[1] == 1:  # one frame on the way: widen
            vision_patches_bthwc = vision_patches_bthwc.expand(-1, self.temporal_patch_size, -1, -1, -1)
        grid = vision_grid_thw.tolist()
        m2 = self.merge_size ** 2
        tokens = [h * w * m2 + 1 for _, h, w in grid]
        if TOKEN_BUDGET <= 0 or len(grid) <= 1 or sum(tokens) <= TOKEN_BUDGET:
            out = enc_stock(self, vision_patches_bthwc, vision_grid_thw)
        else:
            outs, start, i = [], 0, 0
            while i < len(grid):
                j, budget = i, 0
                while j < len(grid) and (j == i or budget + tokens[j] <= TOKEN_BUDGET):
                    budget += tokens[j]
                    j += 1
                rows = sum(h * w for _, h, w in grid[i:j])
                outs.append(enc_stock(self, vision_patches_bthwc[start:start + rows], vision_grid_thw[i:j]))
                start += rows
                i = j
                _S["groups"] += 1
            out = torch.cat(outs)
        if sum(tokens) > EMPTY_CACHE_AFTER and out.is_cuda and _mem_available_gb() < EMPTY_BELOW_GB:
            torch.cuda.empty_cache()
        return out

    encode.__wrapped__ = enc_stock
    enc.forward = encode
    _S["installed"] = True
    _log.warning("[vision] SPARK_VISION=1: Inkling2 vision attention uses per-image fused SDPA on SM12x")


def install_processor(mod):
    """Parts 2 and 3 in the tokenizer process (sglang.srt.multimodal.inkling2.image_processing): the long side of a
    decoded picture capped at MAX_EDGE pixels (no cap with 0), and the lean patch builder."""
    if _S["processor"]:
        return
    _S["processor"] = True
    install_lean_encode(mod)
    if MAX_EDGE <= 0:
        return
    stock = mod._decode_rgb

    def _decode_rgb(image_bytes):
        pixels = stock(image_bytes)
        height, width = pixels.shape[:2]
        if max(height, width) <= MAX_EDGE:
            return pixels
        import numpy as np
        from PIL import Image
        scale = MAX_EDGE / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        out = np.asarray(Image.fromarray(pixels).resize(size, Image.LANCZOS), dtype=np.uint8)
        _S["resized"] += 1
        _log.warning("[vision] picture %dx%d downscaled to %dx%d (long side capped at %d px)",
                     width, height, size[0], size[1], MAX_EDGE)
        return out

    _decode_rgb.__wrapped__ = stock
    mod._decode_rgb = _decode_rgb


def lean_patches(mod, pixels):
    """The engine's patches of a decoded picture ([rows*cols, 1, 40, 40, 3] BF16), built strip by strip with the same
    float32 arithmetic as Inkling2ImageProcessor._encode: (padded / 255 - mean) / std, padding -1, then BF16."""
    import numpy as np
    height, width, _ = pixels.shape
    rows, cols = (height + 39) // 40, (width + 39) // 40
    out = torch.empty((rows * cols, 1, 40, 40, 3), dtype=torch.bfloat16)
    for r in range(rows):
        strip = np.full((40, cols * 40, 3), -1, dtype=np.float32)
        y0 = r * 40
        y1 = min(height, y0 + 40)
        strip[: y1 - y0, :width] = pixels[y0:y1]
        norm = (strip / np.float32(255) - mod._MEAN) / mod._STD
        out[r * cols:(r + 1) * cols, 0] = torch.from_numpy(norm).reshape(40, cols, 40, 3).permute(1, 0, 2, 3).to(torch.bfloat16)
    return out, rows, cols


def install_lean_encode(mod):
    cls = mod.Inkling2ImageProcessor
    stock = cls._encode

    def _encode(self, image, image_token_id, expected_size=None, virtual_size=None):
        if expected_size is not None or virtual_size is not None:  # the training paths: as shipped
            return stock(self, image, image_token_id, expected_size, virtual_size)
        if isinstance(image, (str, os.PathLike)) and not str(image).startswith(("data:", "http://", "https://")):
            raise ValueError("a picture must be a base64 data URL or an http(s) URL")
        if MIN_FREE_GB > 0 and _mem_available_gb() < MIN_FREE_GB:
            raise ValueError(f"the server is short of memory for pictures right now (less than {MIN_FREE_GB:g} GB "
                             "available on the node): retry in a moment, or send smaller pictures")
        try:
            with _LOCK:
                try:
                    pixels = mod._decode_rgb(mod._load_image_bytes(image))
                except (TypeError, ValueError):
                    raise
                except Exception as e:  # noqa: BLE001  (a broken picture is the client's error: HTTP 400)
                    raise ValueError(f"the picture could not be read: {e}") from e
                height, width, _ = pixels.shape
                patches, rows, cols = lean_patches(mod, pixels)
                del pixels
            return mod.HydratedVisionTokenSpan(
                int_tokens=[image_token_id] * (rows * cols),
                vision_patches_bthwc=patches,  # one frame; the encoder hook widens it to two
                vision_grid_thw=torch.tensor([[1, rows, cols]], dtype=torch.int64),
                vision_original_thw=torch.tensor([[1, height, width]], dtype=torch.int64),
            )
        finally:
            _trim()

    _encode.__wrapped__ = stock
    cls._encode = _encode
