"""The confidence head's draft statistics in one pass over the vocabulary (SPARK_GATE_FUSED=1, with SPARK_GATE_HEAD).

patches/spark_gate_head.stats_torch computes, per request and drafted position, five numbers from the draft's step
logits z: log p of the drafted token, the two largest log p and the entropy at temperature 1, and the drafted token's
probability at the request's temperature (greedy requests: 1 if the drafted token is the draft's one best token, 0 if
another token ties with it or beats it). It runs a request at a time, eight passes over a [7, V] float32 copy of the
vocabulary-wide logits (about 75 MB of traffic per request-step), inside the draft graph's tail.

`stats(z, d, temps, greedy, live)` returns the same five [bs, 7] float32 tensors from one Triton launch for the
whole step: program (b, j) reads row z[b, j, :live] once in blocks and keeps, online,
  m1, m2   the two largest values counted with multiplicity (a tie at the top gives m2 == m1, as the reference's
           scatter of -inf at one index of the maximum does)
  s, t     sum exp(z - m) and sum exp(z - m) * z with the running maximum m, rescaled when m grows
  sT       sum exp((z - m) / T) at the request's temperature T (1 for greedy requests, whose q is the tie rule)
and then  lse = m + log s, entropy = lse - t / s, log p(d) = z_d - lse, top1 = m1 - lse, top2 = m2 - lse,
          q = exp((z_d - m) / T) / sT  (greedy: (z_d >= m1) and (m1 > m2)).
The sums are fp32 in another order than torch's softmax, so the numbers differ from the reference in the last bits
(bench/gate_stats_test.py bounds them; the head reads them through a GELU MLP and a threshold on a product of
sigmoids, so K can move only where a product sits on a threshold). All ranks run this same kernel; the ranks still
exchange K every step (spark_gate_head.exchange), so they verify rank 0's rows whatever the bits.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the CPU checks import the reference only
    triton = None

BLOCK = 2048

if triton is not None:
    @triton.jit
    def _stats_kernel(Z, D, TEMP, GREEDY, OUT, stride_b, stride_j, live, G: tl.constexpr, BLOCK: tl.constexpr):
        b = tl.program_id(0)
        j = tl.program_id(1)
        row = Z + b * stride_b + j * stride_j
        greedy = tl.load(GREEDY + b) != 0
        t_req = tl.load(TEMP + b).to(tl.float32)
        inv_t = tl.where(greedy, 1.0, 1.0 / t_req)
        ninf = float("-inf")
        m1 = ninf
        m2 = ninf
        m = ninf
        s = 0.0
        t = 0.0
        st = 0.0
        for off in range(0, live, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            ok = idx < live
            v = tl.load(row + idx, mask=ok, other=ninf).to(tl.float32)
            b1 = tl.max(v, 0)
            many = tl.sum(tl.where(v == b1, 1, 0), 0) > 1
            b2 = tl.where(many, b1, tl.max(tl.where(v == b1, ninf, v), 0))
            n1 = tl.maximum(m1, b1)
            m2 = tl.maximum(tl.minimum(m1, b1), tl.maximum(m2, b2))
            m1 = n1
            mn = tl.maximum(m, b1)
            e = tl.exp(v - mn)                                   # 0 for the masked tail (-inf)
            scale = tl.exp(m - mn)                               # 0 on the first block (m = -inf)
            s = s * scale + tl.sum(e, 0)
            t = t * scale + tl.sum(tl.where(ok, e * v, 0.0), 0)
            st = st * tl.exp((m - mn) * inv_t) + tl.sum(tl.exp((v - mn) * inv_t), 0)
            m = mn
        zd = tl.load(row + tl.load(D + b * G + j)).to(tl.float32)
        lse = m + tl.log(s)
        q = tl.where(greedy, tl.where((zd >= m1) & (m1 > m2), 1.0, 0.0), tl.exp((zd - m) * inv_t) / st)
        o = OUT + (b * G + j) * 5
        tl.store(o + 0, zd - lse)
        tl.store(o + 1, m1 - lse)
        tl.store(o + 2, m2 - lse)
        tl.store(o + 3, lse - t / s)
        tl.store(o + 4, q)


def stats(z, d, temps, greedy, live, out=None):
    """z [>= bs, G, >= live] step logits (any float dtype, last dim contiguous), d [bs, G] int64 drafted tokens,
    temps [bs] float, greedy [bs] bool, live = the columns read -> (lq, top1, top2, ent, q), each [bs, G] float32
    (views of one [bs, G, 5] buffer; pass `out` to reuse one under graph capture)."""
    bs, g = d.shape
    assert z.stride(-1) == 1 and z.shape[1] == g and z.shape[2] >= live
    if out is None:
        out = torch.empty((bs, g, 5), dtype=torch.float32, device=z.device)
    _stats_kernel[(bs, g)](z, d.contiguous(), temps, greedy.to(torch.int8), out, z.stride(0), z.stride(1), live,
                           G=g, BLOCK=BLOCK)
    return tuple(out[..., i] for i in range(5))
