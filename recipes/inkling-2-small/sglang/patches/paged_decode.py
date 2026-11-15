"""Decode/verify attention for GB10 that looks up the page of every key (SPARK_PAGED_DECODE=1, or an e4m3 KV pool).

SGLang's Triton kernel (kernels/ops/inkling2/decode_attention_sm90.py) reads one page-table entry per tile of 32 or
128 keys and takes the tile's other keys from the pool slots that follow it. That is exact when the page size is a
multiple of the tile (SM100 runs 128) or when a request's slots happen to be consecutive. GB10 runs page size 1, and
the slots are consecutive only for a request that was alone since boot: with two or more requests in a decode batch
every step's slots are dealt out in batch order, a prefill that lands during a decode leaves a hole, and the free
lists are FIFO and never sorted, so later requests inherit the fragments. The request then reads other requests' K/V
in place of its newest tokens (2026-11-13: garbage whenever two requests decoded together, see the README).

This module rebuilds that kernel at import from the image's own source with the page loaded per key, and for an e4m3
pool (--kv-cache-dtype fp8_e4m3) also with a cast after the two K/V loads (e4m3 -> BF16 is exact; the FP32 partials
still differ from the BF16 kernel's in their last bits, 1e-6 of the result). Deriving it instead
of copying keeps it the image's kernel across updates; if the rewritten lines ever change, the import fails, and the
server with it, instead of serving a kernel that reads the wrong rows.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import inspect
import linecache
import sys
import types

_TILE = """\
        page = tl.load(PageTable + b * SP + (tile * BLOCK_N) // PAGE).to(tl.int64)
        slot = (tile * BLOCK_N) % PAGE + cols
        kv_offset = page * SK0 + slot[:, None] * SK1 + hk * SK2 + d[None, :]
        valid = (pos < seqlen) & (pos >= first)
"""
_KEY = """\
        valid = (pos < seqlen) & (pos >= first)
        page = tl.load(PageTable + b * SP + pos // PAGE, valid, other=0).to(tl.int64)
        kv_offset = (page * SK0 + (pos % PAGE) * SK1)[:, None] + hk * SK2 + d[None, :]
"""
# A masked load has no defined value where the mask is off, and Triton has no int -> fp8 cast for `other=0`: mask after
# the cast instead. Via FP32 because fp8 -> fp32 and fp32 -> bf16 are the casts the image's own kernels already use.
_LOAD = "{x} = tl.load({X} + kv_offset, valid[:, None], other=0)"
_LOAD_FP8 = ("{x} = tl.where(valid[:, None], tl.load({X} + kv_offset, valid[:, None]).to(tl.float32), 0.0)"
             ".to(Q.dtype.element_ty)")


def _replace(src, old, new):
    if src.count(old) != 1:
        raise RuntimeError(f"paged_decode: decode_attention_sm90 changed, cannot find `{old.strip()}`")
    return src.replace(old, new)


def _derive(fp8):
    from sglang.kernels.ops.inkling2 import decode_attention_sm90 as stock
    src = _replace(inspect.getsource(stock), _TILE, _KEY)
    if fp8:
        for x in "kv":
            src = _replace(src, _LOAD.format(x=x, X=x.upper()), _LOAD_FP8.format(x=x, X=x.upper()))
    name = "paged_decode_fp8" if fp8 else "paged_decode_bf16"
    filename = f"<{name}: decode_attention_sm90, page per key>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)  # triton.jit reads the source back
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(src, filename, "exec"), mod.__dict__)
    return mod.decode_partials


decode_partials = _derive(False)
decode_partials_fp8 = _derive(True)
