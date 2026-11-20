"""NVFP4 expert block scales kept packed in memory and expanded for the routed experts only (SPARK_SCALE_PACK=1, off
by default; needs SPARK_FI_MOE=1). Exact: the GEMM reads the same scale bytes.

A rank holds 191 MB of e4m3 block scales for every NVFP4 expert layer (7.6 GB for 40 layers, 9.2 GB for 48), one byte
for 16 weights. They carry about 4 bits each (39 distinct values a tensor, neighbours within two octaves). Packed
here, in the 128x4 tile layout the kernels read (so a group is 32 bytes that are neighbours in memory):
  bases  one byte a group of 32 scales: the smallest byte of the group, or 0xFF for a group stored raw
  codes  16 bytes a group: 32 offsets from the base, 4 bits each (low nibble first); for a raw group its first four
         bytes are the group's index in `raw` (little endian)
  raw    32 bytes for each group whose bytes span more than 15 (2.4-3.6% of the groups)
17 bytes for 32 plus the raw groups: 55.5-56.7% of the bytes (measured on layers 0, 13, 27, 39), fixed stride, so
any expert's scales expand without a walk. The idea is hyperflare-inference's compressed scale layout ("CSF":
resident packed scales, expanded into a buffer the serial expert layers share); the format and code are this file's.

Run time: before each of a layer's two GEMMs `expand` rewrites, in one scratch tensor the size of a layer's scales
(shared by every layer: 127 + 64 MB a rank), the scales of the experts that have rows in this call (m_indptr), and
the GEMM reads that tensor. Experts without rows keep whatever an earlier layer left there: the GEMM reads no tile
of theirs. The grid is fixed (experts x chunks; an expert without rows returns at once), so the call captures into
CUDA graphs. A decode step expands about 36 experts a layer (18 MB written); a 4,096-token chunk nearly all.

`adopt` (at the end of the load, never under a capture) packs a layer's two scale tensors, expands every expert
back and compares with the original bytes (raises on any difference), then frees the originals.
MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os

import torch

_log = logging.getLogger("spark_scale_pack")
ON = os.environ.get("SPARK_SCALE_PACK") == "1"
GROUP = 32
CHUNK = 128  # groups a program (4 KB of scales)
_SCRATCH = {}  # (device, shape) -> the shared expansion tensor
_S = {"layers": 0, "raw_bytes": 0, "packed_bytes": 0, "expand": None}


class Packed:
    __slots__ = ("bases", "codes", "raw", "shape", "groups")

    def __init__(self, bases, codes, raw, shape):
        self.bases, self.codes, self.raw, self.shape = bases, codes, raw, tuple(shape)
        self.groups = bases.shape[1]

    def nbytes(self):
        return self.bases.numel() + self.codes.numel() + self.raw.numel()


def pack(scales: torch.Tensor, experts_at_a_time: int = 48) -> Packed:
    """(experts, n, k) uint8 (or e4m3) scale bytes, n * k a multiple of 32 -> Packed on the same device."""
    s = scales.view(torch.uint8) if scales.dtype != torch.uint8 else scales
    experts, n, k = s.shape
    if (n * k) % GROUP:
        raise ValueError(f"scale_pack: {n} x {k} scale bytes an expert do not cut into groups of {GROUP}")
    groups = n * k // GROUP
    bases = torch.empty((experts, groups), dtype=torch.uint8, device=s.device)
    codes = torch.empty((experts, groups, GROUP // 2), dtype=torch.uint8, device=s.device)
    raws, count = [], 0
    for first in range(0, experts, experts_at_a_time):
        g = s[first:first + experts_at_a_time].reshape(-1, groups, GROUP)
        lo = g.amin(-1)
        wide = (g.amax(-1) - lo) > 15  # uint8 difference of max and min: never wraps
        delta = (g - lo[..., None]) & 0xF
        c = delta[..., 0::2] | (delta[..., 1::2] << 4)
        if bool((lo == 0xFF).any()):
            raise ValueError("scale_pack: a scale byte of 0xFF (the raw-group marker) is not an e4m3 scale")
        raw = g[wide]
        index = torch.arange(count, count + raw.shape[0], device=s.device, dtype=torch.int64)
        head = torch.stack([(index >> shift) & 0xFF for shift in (0, 8, 16, 24)], -1).to(torch.uint8)
        c[wide] = torch.cat([head, torch.zeros((raw.shape[0], GROUP // 2 - 4), dtype=torch.uint8, device=s.device)], -1)
        bases[first:first + experts_at_a_time] = torch.where(wide, torch.full_like(lo, 0xFF), lo)
        codes[first:first + experts_at_a_time] = c
        raws.append(raw.clone())
        count += raw.shape[0]
        del g, lo, wide, delta, c, raw, index, head
    raw = torch.cat(raws) if count else torch.zeros((1, GROUP), dtype=torch.uint8, device=s.device)
    return Packed(bases, codes, raw.contiguous(), s.shape)


def expand_reference(p: Packed) -> torch.Tensor:
    """Every expert's scales from the packed form, in torch (the load-time check and the CPU test)."""
    lo, hi = p.codes & 0xF, p.codes >> 4
    out = torch.stack((lo, hi), -1).reshape(p.codes.shape[0], p.groups, GROUP) + p.bases[..., None]
    wide = p.bases == 0xFF
    if bool(wide.any()):
        head = p.codes[wide][:, :4].to(torch.int64)
        index = head[:, 0] | (head[:, 1] << 8) | (head[:, 2] << 16) | (head[:, 3] << 24)
        out[wide] = p.raw[index]
    return out.view(p.shape)


def _kernel():
    if _S["expand"] is not None:
        return _S["expand"]
    import triton
    import triton.language as tl

    @triton.jit
    def _expand(Bases, Codes, Raw, Indptr, Out, GROUPS: tl.constexpr, CHUNK_: tl.constexpr):
        e = tl.program_id(0).to(tl.int64)
        if tl.load(Indptr + e + 1) <= tl.load(Indptr + e):
            return
        g = tl.program_id(1).to(tl.int64) * CHUNK_ + tl.arange(0, CHUNK_)
        live = g < GROUPS
        at = e * GROUPS + g
        base = tl.load(Bases + at, live, other=0)
        j = tl.arange(0, 16)
        code = tl.load(Codes + at[:, None] * 16 + j[None, :], live[:, None], other=0)
        wide = live & (base == 255)
        index = (tl.load(Codes + at * 16, wide, other=0).to(tl.int64)
                 | (tl.load(Codes + at * 16 + 1, wide, other=0).to(tl.int64) << 8)
                 | (tl.load(Codes + at * 16 + 2, wide, other=0).to(tl.int64) << 16)
                 | (tl.load(Codes + at * 16 + 3, wide, other=0).to(tl.int64) << 24))
        raw_low = tl.load(Raw + index[:, None] * 32 + 2 * j[None, :], wide[:, None], other=0)
        raw_high = tl.load(Raw + index[:, None] * 32 + 2 * j[None, :] + 1, wide[:, None], other=0)
        low = tl.where(wide[:, None], raw_low, base[:, None] + (code & 0xF))
        high = tl.where(wide[:, None], raw_high, base[:, None] + (code >> 4))
        tl.store(Out + at[:, None] * 32 + 2 * j[None, :], low, live[:, None])
        tl.store(Out + at[:, None] * 32 + 2 * j[None, :] + 1, high, live[:, None])

    _S["expand"] = (_expand, triton.cdiv)
    return _S["expand"]


def scratch(p: Packed) -> torch.Tensor:
    key = (p.bases.device, p.shape)
    if key not in _SCRATCH:  # made at the load (adopt), never under a capture
        _SCRATCH[key] = torch.zeros(p.shape, dtype=torch.uint8, device=p.bases.device)
    return _SCRATCH[key]


def expand(p: Packed, m_indptr: torch.Tensor) -> torch.Tensor:
    """The layer's scale tensor with the scales of every expert that has rows (m_indptr[e + 1] > m_indptr[e])."""
    kernel, cdiv = _kernel()
    out = _SCRATCH[(p.bases.device, p.shape)]
    kernel[(p.bases.shape[0], cdiv(p.groups, CHUNK))](p.bases, p.codes, p.raw, m_indptr, out, p.groups, CHUNK)
    return out


_GEMM = {"module": None, "buffers": {}, "ok": None}


def kernel_ready():
    """True when the forked GEMM is loaded and passed its check at the load (adopt): fi_moe then hands it the packed
    scales instead of expanding them."""
    return _GEMM["ok"] is True


def _check_kernel(layer, name, original, p):
    """The forked GEMM on this layer's real weights: packed scales against the plain ones, a few experts, random
    activations. The bytes must be equal; anything else (or an error) leaves the expansion path in charge."""
    try:
        weight = getattr(layer, name.replace("_scale", "_weight"))
        experts, n, half_k = weight.shape
        device = weight.device
        counts = torch.zeros(experts, dtype=torch.int32, device=device)
        counts[: min(experts, 24): 3] = 8
        m_indptr = torch.zeros(experts + 1, dtype=torch.int32, device=device)
        m_indptr[1:] = torch.cumsum(counts, 0)
        rows = int(m_indptr[-1])
        gen = torch.Generator(device="cpu").manual_seed(5)
        a = torch.randint(0, 256, (rows, half_k), dtype=torch.uint8, generator=gen).to(device)
        sf_rows = ((rows + 127 * experts) // 128 + 2) * 128
        a_sf = torch.randint(0x30, 0x50, (sf_rows, half_k // 8), dtype=torch.uint8, generator=gen).to(device)
        plain = gemm_packed(a, weight, a_sf, original.view(torch.uint8), m_indptr, out=torch.zeros((rows, n), dtype=torch.bfloat16, device=device))
        packed = gemm_packed(a, weight, a_sf, p, m_indptr, out=torch.zeros((rows, n), dtype=torch.bfloat16, device=device))
        same = torch.equal(plain.view(torch.int16), packed.view(torch.int16)) and bool(plain.float().abs().sum() > 0)
        return same, "" if same else "the packed scales give other bytes than the plain ones"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)[:300]


def gemm_module():
    """The forked grouped NVFP4 GEMM that reads packed weight scales in the kernel (kernels/csf_gemm, built into the
    file SPARK_CSF_GEMM names), or None."""
    if _GEMM["module"] is None:
        path = os.environ.get("SPARK_CSF_GEMM")
        if not path or not os.path.isfile(path):
            return None
        import tvm_ffi
        _GEMM["module"] = tvm_ffi.load_module(path)
    return _GEMM["module"]


def _workspace(name, device):
    key = (name, device)
    if key not in _GEMM["buffers"]:  # as FlashInfer's own: 40 MB, made outside any capture (the first call is eager)
        _GEMM["buffers"][key] = torch.empty(40 * 1024 * 1024, dtype=torch.uint8, device=device)
    return _GEMM["buffers"][key]


def gemm_packed(a, w, a_sf, scales, m_indptr, *, alpha=None, out=None, out_dtype=None):
    """FlashInfer's group_gemm_nvfp4_nt_groupwise(a, w, a_sf, <weight scales>, m_indptr, tile 128 x 64 x 256,
    swap_ab=True) with the weights' block scales given as a Packed (read in the kernel: nothing is expanded) or as
    the plain 128x4 tensor (the fork's stock path, for comparison)."""
    module = gemm_module()
    n, k = w.shape[1], w.shape[2] * 2
    if out is None:
        out = torch.empty((a.shape[0], n), dtype=out_dtype or torch.bfloat16, device=a.device)
    empty8 = torch.empty(0, dtype=torch.uint8, device=a.device)
    if alpha is None:
        alpha = torch.empty(0, dtype=torch.float32, device=a.device)
    if isinstance(scales, Packed):
        k4 = scales.shape[2] // 4
        if scales.shape[1] != n or scales.shape[2] * 16 < k or scales.groups != (n // 128) * k4 * 16:
            raise ValueError(f"scale_pack: packed scales {scales.shape} do not belong to weights of {n} x {k}")
        module.group_gemm_nvfp4_csf(_workspace("int", a.device), _workspace("float", a.device), a, w, a_sf, empty8, out,
                                    alpha, m_indptr, scales.bases, scales.codes, scales.raw, n, k, k4, scales.groups)
    else:
        module.group_gemm_nvfp4_csf(_workspace("int", a.device), _workspace("float", a.device), a, w, a_sf, scales, out,
                                    alpha, m_indptr, empty8, empty8, empty8, n, k, 0, 0)
    return out


def adopt(layer):
    """Pack a loaded NVFP4 expert layer's w13 and w2 scales (already in the 128x4 layout), check them, free the
    originals. Returns (packed w13, packed w2) for fi_moe.experts_forward, or None when nothing was done."""
    if not ON:
        return None
    out = []
    for name in ("w13_scale", "w2_scale"):
        scale = getattr(layer, name)
        if not scale.is_cuda:
            raise RuntimeError("SPARK_SCALE_PACK: the scales are packed on the GPU after the load")
        original = scale.data
        p = pack(original)
        target = scratch(p)
        experts = p.bases.shape[0]
        every = torch.arange(experts + 1, dtype=torch.int32, device=original.device)  # one row an expert
        got = expand(p, every)
        if got is not target or not torch.equal(got, original.view(torch.uint8)):
            raise RuntimeError(f"SPARK_SCALE_PACK: {name} does not expand back to the checkpoint's bytes")
        if _S["layers"] == 0 and not torch.equal(expand_reference(p), original.view(torch.uint8)):
            raise RuntimeError(f"SPARK_SCALE_PACK: {name}: the reference expansion differs")
        if _S["layers"] == 0 and _GEMM["ok"] is not False and gemm_module() is not None:
            ok, why = _check_kernel(layer, name, original, p)
            _GEMM["ok"] = ok if _GEMM["ok"] is None else (_GEMM["ok"] and ok)
            _log.warning("[scale_pack] forked GEMM on %s: %s", name, "packed scales give the stock bytes: the kernel "
                         "reads them, nothing is expanded" if ok else f"NOT used ({why}): scales are expanded per call")
        _S["raw_bytes"] += original.numel()
        _S["packed_bytes"] += p.nbytes()
        scale.data = torch.empty((0,), dtype=original.dtype, device=original.device)
        del original, got
        out.append(p)
    _S["layers"] += 1
    if _S["layers"] in (1, 40, 48):
        _log.warning("[scale_pack] %d expert layers: block scales %.1f MB packed to %.1f MB (%.1f%%), every expert "
                     "checked against the checkpoint's bytes; shared expansion buffers %.1f MB", _S["layers"],
                     _S["raw_bytes"] / 2 ** 20, _S["packed_bytes"] / 2 ** 20,
                     100.0 * _S["packed_bytes"] / max(1, _S["raw_bytes"]),
                     sum(t.numel() for t in _SCRATCH.values()) / 2 ** 20)
    torch.cuda.empty_cache()
    return tuple(out)
