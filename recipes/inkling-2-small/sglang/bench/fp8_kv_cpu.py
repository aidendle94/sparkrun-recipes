"""Offline (CPU) check of unit-scale e4m3 KV for Inkling-2: what torch's cast does, and how much attention moves.

K is L2-clipped to norm <= 22.627 and V is RMS-normed per head (head_dim 128), so nothing can reach e4m3's 448.
1. torch's bf16 -> float8_e4m3fn cast against a brute-force nearest-value reference; behaviour above 448 and on NaN.
2. Every e4m3 value is exact in bf16 (so a reader that casts loaded tiles to bf16 is exact).
3. One attention head in fp32: bf16 K/V against e4m3 K/V (logit deltas in nats, output rel-L2), at several K norms.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import torch

torch.manual_seed(0)
F8 = torch.float8_e4m3fn
table = torch.arange(256, dtype=torch.uint8).view(F8).to(torch.float32)
finite = table[torch.isfinite(table)]
print("e4m3 finite values:", finite.numel(), "max", finite.max().item(), "min positive", finite[finite > 0].min().item())
print("all e4m3 values exact in bf16:", torch.equal(finite.to(torch.bfloat16).to(torch.float32), finite))

x = (torch.randn(1 << 20) * 3).to(torch.bfloat16)
x[:4096] = torch.linspace(-30, 30, 4096).to(torch.bfloat16)
x[4096:8192] = (torch.randn(4096) * 0.01).to(torch.bfloat16)
got = x.to(F8).to(torch.float32)
vals = torch.sort(finite).values
xf = x.to(torch.float32)
hi = torch.searchsorted(vals, xf).clamp(1, vals.numel() - 1)
lo_v, hi_v = vals[hi - 1], vals[hi]
nearest = torch.where((xf - lo_v) <= (hi_v - xf), lo_v, hi_v)
tie = (xf - lo_v) == (hi_v - xf)
bad = (got != nearest) & ~tie
print(f"torch cast == nearest value: mismatches {int(bad.sum())} of {x.numel()} (ties {int(tie.sum())}, "
      f"tie results equal to lower/upper: {int((got[tie] == lo_v[tie]).sum())}/{int((got[tie] == hi_v[tie]).sum())})")
edge = torch.tensor([448.0, 449.0, 464.0, 480.0, 1000.0, float("inf"), float("nan"), -1000.0]).to(torch.bfloat16)
print("cast of", edge.tolist(), "->", edge.to(F8).to(torch.float32).tolist(), "bytes", edge.to(F8).view(torch.uint8).tolist())
rel = ((got - xf).abs() / xf.abs().clamp_min(1e-30))[xf.abs() >= 2 ** -6]
print(f"relative rounding error in the normal range: rms {rel.pow(2).mean().sqrt():.4f}, max {rel.max():.4f}")

D, N, QN = 128, 4096, 64
for k_norm in (22.627, 8.0, 2.0, 0.5):
    k = torch.randn(N, D)
    k = (k / k.norm(dim=-1, keepdim=True) * k_norm).to(torch.bfloat16)
    v = torch.randn(N, D)
    v = (v / v.pow(2).mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
    q = torch.randn(QN, D)
    q = (q / q.norm(dim=-1, keepdim=True) * 22.627).to(torch.bfloat16)
    # make a few keys relevant so the softmax is peaked the way retrieval heads are
    k[:QN] = (q.float() / 22.627 * k_norm).to(torch.bfloat16)
    def attend(kk, vv):
        s = q.float() @ kk.float().T * D ** -0.5
        return s, torch.softmax(s, -1) @ vv.float()
    s0, o0 = attend(k, v)
    s1, o1 = attend(k.to(F8), v.to(F8))
    sub = (k.float().abs() < 2 ** -6).float().mean()
    print(f"|k| {k_norm:>6}: logit delta rms {float((s1 - s0).pow(2).mean().sqrt()):.4f} max {float((s1 - s0).abs().max()):.4f} nats "
          f"(logit max {float(s0.max()):.1f}); output rel-L2 {float((o1 - o0).norm() / o0.norm()):.4f}; "
          f"K elements below 2^-6: {float(sub):.1%}")
