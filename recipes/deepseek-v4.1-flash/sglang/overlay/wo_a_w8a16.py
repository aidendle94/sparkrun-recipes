"""DeepSeek-V4.1's attention output projection `wo_a` with FP8 weights and BF16 activations (W8A16).

`wo_a` maps each head group's attention output to the output LoRA rank ([T, G, D] x [G, R, D] -> [T, G, R]; per rank on
TP4 G = 2, R = 1024, D = 4096). The checkpoint stores it as FP8 E4M3 with a UE8M0 (power-of-two) scale per 32x32 block;
SGLang's FP8 path for it (DeepGEMM fp8_einsum) takes 128x128 blocks only, so it dequantizes the weight to BF16 at load
and every decode step reads 16.8 MB per layer per rank instead of 8.4.

This hook stores the weight as FP8 again, exactly: every BF16 value is q * 2^k, and a scale of 2^e per 1x32 row segment
with e = ceil(log2(amax / 448)) <= k represents it without loss (checked for every layer at load). The GEMM is a Triton
kernel that loads the FP8 tile, applies the scale and multiplies in BF16 with FP32 accumulation, so the activations
are never quantized: products are the same as with the BF16 weight and only the summation order differs from cuBLAS.

At load the hook checks the kernel on layer 0 against the BF16 reference for several batch sizes and times both; if the
relative error exceeds SPARK_WO_A_W8A16_TOL the whole conversion is abandoned and `wo_a` stays BF16.

  SPARK_WO_A_W8A16       1 enables (default 0)
  SPARK_WO_A_W8A16_TOL   largest accepted relative error against the BF16 reference (default 0.02)

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os
import time

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_E4M3_MAX = 448.0


def _enabled() -> bool:
    return os.environ.get("SPARK_WO_A_W8A16", "0") == "1"


@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, T,
                  sxt, sxg, swg, swr, ssg, ssr, syt, syg,
                  D: tl.constexpr, BT: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr, F32: tl.constexpr = False):
    pid_t = tl.program_id(0)
    pid_r = tl.program_id(1)
    g = tl.program_id(2)
    ot = pid_t * BT + tl.arange(0, BT)
    orr = pid_r * BR + tl.arange(0, BR)
    acc = tl.zeros((BT, BR), dtype=tl.float32)
    for d0 in range(0, D, BD):
        od = d0 + tl.arange(0, BD)
        x = tl.load(x_ptr + g * sxg + ot[:, None] * sxt + od[None, :], mask=ot[:, None] < T, other=0.0)
        # E4M3 decoded from its bits (sign, 4-bit exponent, 3-bit mantissa, subnormals), then the block scale 2^(e-127):
        # exact, and independent of how a Triton build converts float8 types
        b = tl.load(w_ptr + g * swg + orr[:, None] * swr + od[None, :]).to(tl.int32)
        e = tl.load(s_ptr + g * ssg + orr[:, None] * ssr + (od // 32)[None, :]).to(tl.int32)
        ex = (b >> 3) & 15
        man = (b & 7).to(tl.float32) * 0.125
        mag = tl.where(ex == 0, man, 1.0 + man) * tl.exp2((tl.where(ex == 0, -6, ex - 7) + e - 127).to(tl.float32))
        wv = tl.where((b & 128) != 0, -mag, mag)
        if F32:   # test mode (the CPU interpreter has no bfloat16): same decode and tiling, fp32 operands
            acc += tl.sum(x.to(tl.float32)[:, None, :] * wv[None, :, :], axis=2)
        else:
            acc += tl.dot(x, tl.trans(wv.to(tl.bfloat16)))
    if F32:
        tl.store(y_ptr + ot[:, None] * syt + g * syg + orr[None, :], acc, mask=ot[:, None] < T)
    else:
        tl.store(y_ptr + ot[:, None] * syt + g * syg + orr[None, :], acc.to(tl.bfloat16), mask=ot[:, None] < T)


def w8a16_grouped(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, _f32: bool = False) -> torch.Tensor:
    """x [T, G, D] bf16 (last dim contiguous); w [G, R, D] float8_e4m3fn (read as bytes); s [G, R, D/32] uint8 (UE8M0)
    -> [T, G, R] bf16."""
    T, G, D = x.shape
    w = w.view(torch.uint8)
    R = w.shape[1]
    y = torch.empty((T, G, R), dtype=torch.float32 if _f32 else torch.bfloat16, device=x.device)
    if T == 0:
        return y
    bt = 16 if T <= 16 else (32 if T <= 64 else 64)
    br = 32 if T <= 128 else 64
    grid = (triton.cdiv(T, bt), triton.cdiv(R, br), G)
    _w8a16_kernel[grid](x, w, s, y, T,
                        x.stride(0), x.stride(1), w.stride(0), w.stride(1), s.stride(0), s.stride(1),
                        y.stride(0), y.stride(1), D=D, BT=bt, BR=br, BD=128, F32=_f32, num_warps=4)
    return y


def _quantize_exact(w: torch.Tensor):
    """BF16 [N, K] -> (E4M3 [N, K], UE8M0 uint8 [N, K/32], mismatching elements)."""
    n, k = w.shape
    seg = w.float().view(n, k // 32, 32)
    amax = seg.abs().amax(dim=-1)
    mant, ex = torch.frexp(amax / _E4M3_MAX)
    e = torch.where(mant > 0.5, ex, ex - 1)
    e = torch.where(amax > 0, e, torch.full_like(e, -127)).clamp(-127, 127)
    scale = torch.ldexp(torch.ones_like(amax), e)
    q = (seg / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    mismatches = int(((q.float() * scale.unsqueeze(-1)) != seg).sum())
    return q.view(n, k), (e + 127).to(torch.uint8), mismatches


def _self_check(q: torch.Tensor, s: torch.Tensor, ref_w: torch.Tensor) -> tuple[bool, str]:
    """Kernel vs torch.bmm on the BF16 weight for decode, verify and prefill batch sizes; times both."""
    tol = float(os.environ.get("SPARK_WO_A_W8A16_TOL", "0.02"))
    G, R, D = q.shape
    notes, ok = [], True
    gen = torch.Generator(device=q.device).manual_seed(0)
    for T in (1, 6, 48, 96, 2048):
        x = torch.randn((T, G, D), generator=gen, device=q.device, dtype=torch.float32).to(torch.bfloat16)
        ref = torch.bmm(x.transpose(0, 1), ref_w.transpose(1, 2)).transpose(0, 1)
        out = w8a16_grouped(x, q, s)
        err = float((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12))
        ok &= err <= tol
        times = []
        for fn in (lambda: w8a16_grouped(x, q, s),
                   lambda: torch.bmm(x.transpose(0, 1), ref_w.transpose(1, 2))):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / 20 * 1e6)
        notes.append(f"T={T}: err {err:.4f}, {times[0]:.0f} us vs bf16 bmm {times[1]:.0f} us")
    return ok, "; ".join(notes)


def _convert(model) -> None:
    layers = [m for m in model.modules() if type(m).__name__ == "MQALayer" and not getattr(m, "wo_a_fp8", True)]
    done = kept = 0
    freed = 0
    checked = False
    for mod in layers:
        w = getattr(mod.wo_a, "weight", None)
        G, R = mod.n_local_groups, mod.o_lora_rank
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[0] != G * R or w.shape[1] % 128:
            kept += 1
            continue
        D = w.shape[1]
        q, s, bad = _quantize_exact(w.data)
        if bad:
            logger.warning("wo_a W8A16: layer %s kept in BF16, %d elements not exact", getattr(mod, "layer_id", "?"), bad)
            kept += 1
            continue
        q, s = q.view(G, R, D).contiguous(), s.view(G, R, D // 32).contiguous()
        if not checked:
            ok, notes = _self_check(q, s, w.data.view(G, R, D))
            logger.info("wo_a W8A16 self-check (layer %s): %s", getattr(mod, "layer_id", "?"), notes)
            if not ok:
                logger.warning("wo_a W8A16: self-check failed; wo_a stays BF16 on every layer")
                return
            checked = True
        freed += w.numel() * w.element_size()
        ph = torch.nn.Parameter(torch.empty(G * R, 1, dtype=torch.bfloat16, device=w.device), requires_grad=False)
        ph._spark_w8a16 = (q, s)
        mod.wo_a.weight = ph
        done += 1
    torch.cuda.empty_cache()
    logger.info("wo_a W8A16: %d attention layers converted exactly (%d kept in BF16), %.2f GB of BF16 weight released",
                done, kept, freed / 2**30)


def install(module) -> None:
    """sglang.srt.models.deepseek_v4: W8A16 wo_a for the target model's attention layers."""
    if not _enabled():
        return
    stock_mm = module._apply_wo_a_bf16_matmul

    def _apply_wo_a_bf16_matmul(o, wo_a, *args, **kwargs):
        base = wo_a._base if wo_a._base is not None else wo_a
        spec = getattr(base, "_spark_w8a16", None)
        if spec is None or kwargs.get("fuse_mxfp8_quant"):
            return stock_mm(o, wo_a, *args, **kwargs)
        return w8a16_grouped(o, *spec)

    module._apply_wo_a_bf16_matmul = _apply_wo_a_bf16_matmul
    cls = module.DeepseekV4ForCausalLM
    stock_load = cls.load_weights

    def load_weights(self, weights, is_nextn=False):
        out = stock_load(self, weights, is_nextn=is_nextn)
        if not is_nextn:
            _convert(self)
        return out

    cls.load_weights = load_weights
    logger.info("wo_a W8A16 on: FP8 weights (exact), BF16 activations, FP32 accumulation")
