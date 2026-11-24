"""Check patches/gate_stats.py (the fused draft statistics) against spark_gate_head.stats_torch, and time both.

CPU (Triton interpreter, small vocabulary):  TRITON_INTERPRET=1 python3 bench/gate_stats_test.py --cpu
GPU (an idle GB10, the real vocabulary):      python3 bench/gate_stats_test.py
Cases: peaky and flat rows, a masked tail, a tie at the top (greedy reads 0), a drafted token that is not the best,
greedy and sampled requests at temperatures 0.7 / 1.0 / 1.3, bf16 logits like the engine's. Prints the largest
differences per statistic, the head's K from both (the exported head file if given with --head) and, on the GPU, the
time of the reference loop and of the fused launch for 1, 4 and 8 requests.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "patches"))
import torch  # noqa: E402
import gate_stats  # noqa: E402
import spark_gate_head as sgh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--head", default="")
a = ap.parse_args()
dev = torch.device("cpu" if a.cpu else "cuda")
G = 7
V, LIVE = (6000, 5000) if a.cpu else (201088, 201024)
gen = torch.Generator().manual_seed(7)


def make(bs):
    z = torch.randn((bs, G, V), generator=gen) * torch.tensor([0.5, 2.0, 4.0, 1.0] * 2)[:bs].view(bs, 1, 1)
    best = torch.randint(0, LIVE, (bs, G), generator=gen)
    z.scatter_(-1, best.unsqueeze(-1), z.amax(-1, keepdim=True) + torch.rand((bs, G, 1), generator=gen) * 6)
    z = z.to(torch.bfloat16)
    z[..., LIVE:] = float("-inf")
    d = best.clone()
    d[:, 4:] = torch.randint(0, LIVE, (bs, G - 4), generator=gen)       # some drafts are not the best token
    if bs > 1:                                                           # a tie at the top of request 1, position 2
        z[1, 2, (int(best[1, 2]) + 1) % LIVE] = z[1, 2, int(best[1, 2])]
    temps = torch.tensor([1.0, 0.7, 1.3, 1.0, 0.0, 1.0, 0.7, 1.0][:bs])
    greedy = torch.tensor([False, True, False, True, True, False, False, True][:bs])
    return z.to(dev), d.to(dev), temps.to(dev), greedy.to(dev)


def ref(z, d, temps, greedy):
    A = torch.zeros((G, LIVE), dtype=torch.float32, device=dev)
    B = torch.zeros((G, LIVE), dtype=torch.float32, device=dev)
    per = [sgh.stats_torch(z[b], d[b], temps[b], greedy[b], A, B) for b in range(d.shape[0])]
    return tuple(torch.stack([r[i] for r in per]) for i in range(5))


names = ("log p(d)", "top1", "top2", "entropy", "q")
worst = [0.0] * 5
for bs in (1, 4, 8):
    z, d, temps, greedy = make(bs)
    want = ref(z, d, temps, greedy)
    got = gate_stats.stats(z, d, temps, greedy, LIVE)
    for i in range(5):
        diff = (want[i] - got[i]).abs().max().item()
        worst[i] = max(worst[i], diff)
    gq = got[4][greedy].tolist()
    print(f"bs {bs}: max |diff| " + ", ".join(f"{n} {(want[i] - got[i]).abs().max().item():.2e}" for i, n in enumerate(names))
          + f"; greedy q values {sorted(set(round(x, 3) for x in sum(gq, [])))}", flush=True)
    if bs > 1:
        print(f"   tie case (request 1, position 2): q ref {want[4][1, 2].item()} fused {got[4][1, 2].item()}; "
              f"top1 - top2 ref {(want[1] - want[2])[1, 2].item():.3g} fused {(got[1] - got[2])[1, 2].item():.3g}")
ok = all(w < 2e-3 for w in worst[:4]) and worst[4] < 1e-4
print(f"worst: " + ", ".join(f"{n} {w:.2e}" for n, w in zip(names, worst)) + f" -> {'PASS' if ok else 'FAIL'}")

if a.head and os.path.isfile(a.head):
    head = sgh.load(a.head, dev)
    H = head["wh"].shape[0]
    has_next = torch.ones(G, device=dev); has_next[-1] = 0
    consts = (has_next, torch.eye(G, device=dev))
    same = total = 0
    for trial in range(50):
        z, d, temps, greedy = make(8)
        hidden = (torch.randn((8 * G, H), generator=gen) * 0.5).to(torch.bfloat16).to(dev)
        k_ref = sgh.rows(sgh.predict(head, hidden, sgh.features(*ref(z, d, temps, greedy), temps, greedy, consts)), head["thr"])
        k_fus = sgh.rows(sgh.predict(head, hidden, sgh.features(*gate_stats.stats(z, d, temps, greedy, LIVE), temps, greedy, consts)), head["thr"])
        same += int((k_ref == k_fus).sum()); total += k_ref.numel()
    print(f"K equal on {same} of {total} request-steps (random inputs through the head)")

if not a.cpu:
    for bs in (1, 4, 8):
        z, d, temps, greedy = make(bs)
        out = torch.empty((bs, G, 5), dtype=torch.float32, device=dev)
        for name, fn in (("reference", lambda: ref(z, d, temps, greedy)),
                         ("fused", lambda: gate_stats.stats(z, d, temps, greedy, LIVE, out=out))):
            for _ in range(5):
                fn()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(50):
                fn()
            torch.cuda.synchronize()
            print(f"bs {bs} {name}: {(time.perf_counter() - t0) / 50 * 1e6:.0f} us (eager, launches included)")
