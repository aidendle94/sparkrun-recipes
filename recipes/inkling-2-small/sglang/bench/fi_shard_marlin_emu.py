"""fi_shard_test.py's `marlin` checks without a GPU: runs `fi_shard_test.py --emulate` with a plain-torch stand-in
for the image's MXFP8 Hopper path loaded in place of sglang.srt.models.inkling_common.inkling2_moe (the module the
checks drive on the GPU). The stand-in has the names and the calling sequence the checks rely on (repack_mxfp8, route,
gemm / gemv, swiglu_fake_quantize_mxfp8, the all-reduce helpers, Inkling2Experts._forward_source) and float64
products rounded to BF16 where Marlin rounds; rows a rank does not own are NaN, as they are unset on the GPU. It
proves nothing about the kernels: it is there so that a mistake in the checks themselves (the cut of the shards, the
float64 reference, the bookkeeping of what ran) shows before a window instead of in it.

    python3 fi_shard_marlin_emu.py --patches patches

Exits like fi_shard_test.py. MIT License, Copyright (c) 2026 Aiden Le.
"""
import os
import runpy
import sys
import types
from functools import partial

import torch

E4M3 = torch.tensor([float("nan") if b & 127 == 127 else (-1) ** (b >> 7) * (
    (b & 7) / 8 * 2.0 ** -6 if not (b >> 3) & 15 else (1 + (b & 7) / 8) * 2.0 ** (((b >> 3) & 15) - 7))
    for b in range(256)], dtype=torch.float64)

marlin = types.ModuleType("marlin_sm90")
marlin.ROUTE_BLOCKS = 256
marlin.GEMV_MAX_EXPERTS = 8  # between a rank's experts and all of them at --emulate's shapes, as 128 is on the GPU


def repack_mxfp8(weight, scales):
    if weight.dtype != torch.float8_e4m3fn or int(scales.max()) >= 135:
        raise ValueError("repack_mxfp8: e4m3 weights and e8m0 scales below 135")
    w = E4M3.float()[weight.view(torch.uint8).int()] * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return w, scales, None  # exact in FP32 (four significant bits times a power of two)


def _block_size(rows_per_expert):
    for size in (8, 16, 32, 64):
        if rows_per_expert / size < (0.5 if size == 8 else 0.9):
            return size
    return 64


def route(counts, offsets, rows, tokens, dst2src=None, block_size=None):
    return dst2src, None, None, block_size or _block_size(min(tokens, rows) / counts.numel())


def _rows(xq, weight, out_size, local, own, top_k):
    out = torch.full((xq.shape[0] * top_k, out_size), float("nan"), dtype=torch.bfloat16)
    if weight.shape[1] != out_size or weight.shape[2] != xq.shape[1]:
        raise ValueError(f"the weights {tuple(weight.shape)} do not take rows of {xq.shape[1]} to {out_size}")
    for r in own.nonzero().flatten().tolist():
        out[r] = (weight[local[r]].double() @ xq[r // top_k].double()).to(torch.bfloat16)
    return out


def gemm(xq, marlin_weights, out_size, routing, top_k):
    return _rows(xq, marlin_weights[0], out_size, *routing[0], top_k)


def gemv(xq, marlin_weights, out_size, metadata, top_k, tokens, splits=None, warps=None):
    return _rows(xq, marlin_weights[0], out_size, *metadata[2], top_k)


def uses_gemv(marlin_weights, tokens, num_local):
    return num_local <= marlin.GEMV_MAX_EXPERTS and tokens <= 2


def _blocks(y):
    """Rows quantized 32 values at a time to a few bits under the block's largest value (BF16-exact)."""
    rows, width = y.shape
    if width % 32:
        raise ValueError("a width of whole 32-wide blocks")
    out = torch.empty((rows, width), dtype=torch.bfloat16)
    for i in range(0, rows, 256):  # float64 temporaries of a few rows at a time (the container this runs in is small)
        b = y[i:i + 256].double().view(-1, width // 32, 32)
        step = torch.exp2(torch.floor(torch.log2(b.abs().amax(-1, keepdim=True).clamp_min(1e-30))) - 3)
        out[i:i + 256] = (torch.round(b / step) * step).view(-1, width).to(torch.bfloat16)
    return out


def swiglu_fake_quantize_mxfp8(acc, ids, gammas, start, num_local, caps):
    local = ids.long() - start
    own = (local >= 0) & (local < num_local)
    out = torch.full((acc.shape[0], acc.shape[1] // 2), float("nan"), dtype=torch.bfloat16)
    rows = own.nonzero().flatten()
    for i in range(0, rows.numel(), 256):  # a few rows at a time, as in _blocks
        r = rows[i:i + 256]
        gate, up = acc[r, 0::2].float().clamp(max=caps[0]), acc[r, 1::2].float().clamp(-caps[1], caps[1])
        out[r] = _blocks((torch.nn.functional.silu(gate) * up * gammas.float()[r, None]).to(torch.bfloat16))
    return out


for fn in (repack_mxfp8, _block_size, route, gemm, gemv, uses_gemv, swiglu_fake_quantize_mxfp8):
    setattr(marlin, fn.__name__, fn)

STOCK = '''
BF16_COMBINE_MIN_TOKENS = 512


def get_parallel():
    raise RuntimeError("the checks replace get_parallel")


multicast_all_reduce_input = multicast_all_reduce = oneshot_all_reduce_input = oneshot_all_reduce = get_parallel


class Inkling2Experts:
    def _forward_source(self, x, weights, ids, combine):
        tokens, top_k = ids.shape
        num_local = self.w13_weight.shape[0]
        start = self.ep_rank * num_local
        flat = ids.flatten()
        n = flat.numel()
        local = flat.long() - start
        own = (local >= 0) & (local < num_local)
        counts = torch.bincount(local[own], minlength=num_local)
        if marlin_sm90.uses_gemv(self._marlin["w13"], tokens, num_local):
            matmul = partial(marlin_sm90.gemv, metadata=(counts, None, (local, own)), tokens=tokens)
        else:
            routing = marlin_sm90.route(counts, None, n, tokens, (local, own))
            matmul = partial(marlin_sm90.gemm, routing=routing)
        xq = blocks(x)
        acc = matmul(xq, self._marlin["w13"], self.output_sizes["w13"], top_k=top_k)
        hidden = marlin_sm90.swiglu_fake_quantize_mxfp8(
            acc, flat, weights.flatten(), start, num_local, (self.gate_cap, self.up_cap))
        down = matmul(hidden, self._marlin["w2"], self.output_sizes["w2"], top_k=1)
        group = get_parallel().moe_ep_group
        wide = tokens > BF16_COMBINE_MIN_TOKENS
        if wide:
            out = oneshot_all_reduce_input(group, tokens, down.shape[1], torch.bfloat16, tag="moe")
        else:
            out = multicast_all_reduce_input(group, (tokens, down.shape[1]), torch.float32)
        rows = torch.where(own[:, None], down.float(), torch.zeros((), dtype=torch.float32))
        out.copy_(rows.view(tokens, top_k, -1).sum(1))
        if wide:
            combined = oneshot_all_reduce(out, group, acc_f32=True, tag="moe")
        else:
            combined = multicast_all_reduce(out, group)
        if not combine:
            return combined, self._identity[:tokens], 1
        return combined.to(x.dtype)
'''

if __name__ == "__main__":
    name = "sglang.srt.models.inkling_common.inkling2_moe"
    stock = types.ModuleType(name)
    stock.__dict__.update(torch=torch, partial=partial, marlin_sm90=marlin, blocks=_blocks)
    exec(compile(STOCK, name, "exec"), stock.__dict__)
    import sglang.srt  # noqa: F401  (the real package: fi_moe imports stock kernels through it)
    parts = name.split(".")
    for i in range(3, len(parts)):  # the model packages above the stand-in: empty ones (the real are the serving stack)
        pkg = types.ModuleType(".".join(parts[:i]))
        pkg.__path__ = []
        sys.modules[pkg.__name__] = pkg
    sys.modules[name] = stock
    for i in range(2, len(parts)):
        setattr(sys.modules[".".join(parts[:i])], parts[i], sys.modules[".".join(parts[:i + 1])])
    test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fi_shard_test.py")
    sys.argv = [test, *sys.argv[1:], "--emulate", "--only", "marlin"]
    runpy.run_path(test, run_name="__main__")
