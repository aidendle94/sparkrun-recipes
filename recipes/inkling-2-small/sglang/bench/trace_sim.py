"""Read a step trace (patches/spark_trace.py, driven by bench/trace_run.py) and simulate decode-step remedies offline.

    python3 bench/trace_sim.py <trace dir> [--out logs/trace-sim-<label>.json] [--sweeps 3]
CPU only, needs torch (run it in the image with --runtime runc and no GPU; the host has no torch).
A rank's expert time in a layer follows the distinct local experts the step's rows reach, and every rank waits for the
busiest one at the layer's all-reduce, so a step's expert cost is  sum over layers of bytes x max over ranks(distinct).
1. Acceptance by phase; step time by batch size; a regression of step time on that cost (does the busiest rank or the
   mean explain the time?).
2. Rank skew: busiest rank / mean under the shipped contiguous ownership, and under
   - a static placement fitted on the even prompts (deal by usage, then swaps that lower the summed maximum),
   - R replicated experts per layer (the R most used on the even prompts), each active one computed by the rank with
     the least to do (lowest rank on ties): what an all-integer in-graph assignment would give,
   - both, and the bound (every rank reads the mean, what tensor-parallel experts would give).
   All scored on the odd prompts.
3. Verifying only the first K rows of a request, K chosen from the draft's own logits (rows after the first doubtful
   draft are dropped): tokens/s against the full 8 rows with the fitted step-time model, threshold fitted on even
   prompts and scored on odd ones, for several confidence features; and the oracle (K = accepted + 1).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import json
import os
import re

import torch

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--out", default="")
ap.add_argument("--sweeps", type=int, default=3)
ap.add_argument("--experts", type=int, default=384)
ap.add_argument("--ranks", type=int, default=4)
ap.add_argument("--no-placement", action="store_true", help="skip the placement fit of section 2 (minutes)")
ap.add_argument("--model", default="", help="t0_ms,ms_per_MB for section 3 instead of the g1 regression on the busiest rank")
args = ap.parse_args()
torch.manual_seed(0)
E, R4 = args.experts, args.ranks
LOCAL = E // R4
steps = [s for f in sorted(glob.glob(os.path.join(args.dir, "trace-*.pt"))) for s in torch.load(f)]
steps = [s for s in steps if "ids" in s and s["rids"] and s["rids"][0] != "flush"]
L = steps[0]["ids"].shape[1]
W = torch.tensor([17.9] * 40 + [32.9] * (L - 40))[:L]  # MB read per distinct expert (NVFP4 layers 0-39, MXFP8 after)
print(f"{len(steps)} steps, {L} expert layers, top-{steps[0]['ids'].shape[3]}", flush=True)
report = {"steps": len(steps)}


def key(rid):
    m = re.match(r"(\w+?)-([a-z])(\d+)$", rid)
    return (m.group(1), int(m.group(3))) if m else (rid, 0)


# first[s, l, e]: the first verify row (0-7, over the step's requests) that routes to expert e; 8 = not reached
S = len(steps)
first = torch.full((S, L, E), 8, dtype=torch.uint8)
pos = torch.arange(8)[None, None, :, None]
for i, s in enumerate(steps):
    ids = s["ids"].long()  # [bs, L, 8, k]
    p = pos.expand_as(ids).permute(1, 0, 2, 3).reshape(L, -1).to(torch.uint8)
    first[i].scatter_reduce_(1, ids.permute(1, 0, 2, 3).reshape(L, -1), p, "amin")
    del s["ids"]  # the head has little memory to spare beside a serving rank
phase = [key(s["rids"][0])[0] for s in steps]
prompt = [key(s["rids"][0])[1] for s in steps]
bs = torch.tensor([s["bs"] for s in steps])
t = torch.tensor([s["t"] for s in steps], dtype=torch.float64)
dt = torch.cat([torch.tensor([float("nan")], dtype=torch.float64), (t[1:] - t[:-1]) * 1e3])
same = torch.tensor([False] + [steps[i]["rids"] == steps[i - 1]["rids"] for i in range(1, S)])
dt[~same] = float("nan")  # only steps that follow a step of the same requests are timed
phases = list(dict.fromkeys(phase))
sel = lambda ph, parity=None: torch.tensor([p == ph and (parity is None or q % 2 == parity) for p, q in zip(phase, prompt)])
identity = (torch.arange(E) // LOCAL)[None].expand(L, E).contiguous()


def counts(rows, place, k=8, exclude=None):
    """[n, L, ranks] distinct experts per rank among the first k verify rows; exclude: [L, E] bool left out."""
    out = torch.zeros(len(rows), L, R4)
    for at in range(0, len(rows), 512):
        act = first[rows[at:at + 512]] < k
        if exclude is not None:
            act = act & ~exclude[None]
        out[at:at + 512].scatter_add_(2, place[None].expand(act.shape[0], L, E), act.float())
    return out


def cost(c):
    """[n] MB: busiest rank per layer, and the mean rank."""
    return (c.max(-1).values * W).sum(-1), (c.mean(-1) * W).sum(-1)


# ---- 1. acceptance, step time, regression
print("\n== acceptance and step time by phase ==")
report["phases"] = {}
for ph in phases:
    rows = sel(ph)
    correct = torch.cat([steps[i]["correct"] for i in rows.nonzero().flatten().tolist()]).float()
    commit = torch.cat([steps[i]["commit"] for i in rows.nonzero().flatten().tolist()]).float()
    hist = torch.bincount(correct.long(), minlength=8)[:8]
    ok = rows & ~dt.isnan() & (dt < 2000)
    med = dt[ok].median().item() if ok.any() else float("nan")
    mx, mean = cost(counts(rows.nonzero().flatten(), identity))
    info = {"steps": int(rows.sum()), "bs": round(bs[rows].float().mean().item(), 2), "commit": round(commit.mean().item(), 2),
            "accepted_hist": hist.tolist(), "step_ms_median": round(med, 1), "skew": round((mx.sum() / mean.sum()).item(), 3),
            "distinct_per_rank_layer": round((counts(rows.nonzero().flatten(), identity).mean()).item(), 1)}
    report["phases"][ph] = info
    print(ph, info, flush=True)

print("\n== step time against expert bytes (steps of one request, timed) ==")
fit = {}
for ph in phases:
    rows = sel(ph) & ~dt.isnan() & (dt < 2000) & (bs == 1)
    if rows.sum() < 200:
        continue
    mx, mean = cost(counts(rows.nonzero().flatten(), identity))
    y = dt[rows].float()
    keep = (y - y.median()).abs() < 4 * (y - y.median()).abs().median() + 5  # drop steps a prefill or a flush stretched
    mx, mean, y = mx[keep], mean[keep], y[keep]
    for name, cols in (("max", [mx]), ("mean", [mean]), ("mean+excess", [mean, mx - mean])):
        X = torch.stack([torch.ones_like(y)] + cols, 1).double()
        beta = torch.linalg.solve(X.T @ X, X.T @ y.double())
        r2 = 1 - ((X @ beta - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        fit[(ph, name)] = beta.tolist()
        print(f"{ph} {name:12s} n={len(y)} step_ms = {beta[0]:.1f} + " + " + ".join(f"{b * 1000:.2f} ms/GB x {n}" for b, n in
              zip(beta[1:], name.split("+"))) + f"   R2 {r2:.3f}   (mean step {y.mean():.1f} ms, busiest-rank bytes "
              f"{mx.mean():.0f} MB, mean-rank {mean.mean():.0f} MB)", flush=True)
rows = ~dt.isnan() & (dt < 2000) & torch.tensor([p in ("g1", "g4", "g8") for p in phase])
if rows.sum() > 200:  # across batch sizes: a per-request term beside the bytes
    mx, mean = cost(counts(rows.nonzero().flatten(), identity))
    y = dt[rows].double()
    for name, col in (("max", mx), ("mean", mean)):
        X = torch.stack([torch.ones_like(y), bs[rows].double(), col.double()], 1)
        beta = torch.linalg.solve(X.T @ X, X.T @ y)
        r2 = 1 - ((X @ beta - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        fit[("all", name)] = beta.tolist()
        print(f"g1+g4+g8 {name:5s} n={len(y)} step_ms = {beta[0]:.1f} + {beta[1]:.2f} x requests + {beta[2] * 1000:.2f} ms/GB x {name}"
              f"   R2 {r2:.3f}", flush=True)
report["fit"] = {f"{a}/{b}": v for (a, b), v in fit.items()}

# ---- 2. rank skew remedies
train = (sel("g1", 0) | sel("s1", 0)).nonzero().flatten()
tests = {"g1 odd": sel("g1", 1), "s1 odd": sel("s1", 1)}
tests.update({ph: sel(ph) for ph in phases if ph not in ("g1", "s1")})
tests = {k: v.nonzero().flatten() for k, v in tests.items() if v.any()}
usage = sum((first[train[at:at + 512]] < 8).float().sum(0) for at in range(0, len(train), 512)) / len(train)  # [L, E]
fit_steps = train[::max(1, len(train) // 1500)]  # the swap search is quadratic in experts: 1,500 steps are enough


def fit_placement(layer):
    A = (first[fit_steps, layer] < 8).float()  # [steps, E]
    order = usage[layer].argsort(descending=True)
    place = torch.empty(E, dtype=torch.long)
    snake = torch.arange(E) % (2 * R4)
    place[order] = torch.where(snake < R4, snake, 2 * R4 - 1 - snake)  # deal by usage: equal marginals, LOCAL each
    dealt = place.clone()
    c = torch.zeros(A.shape[0], R4).index_add_(1, place, A)
    best = c.max(-1).values.sum() + 1e-3 * (c * c).sum()
    for _ in range(args.sweeps):
        moved = 0
        for i in torch.randperm(E).tolist():
            a = place[i].item()
            others = (place != a).nonzero().flatten()
            d = A[:, others] - A[:, i:i + 1]  # [steps, J]: rank a gains d, the other expert's rank loses d
            new = c[:, None, :].repeat(1, len(others), 1)
            new[:, :, a] += d
            new.scatter_add_(2, place[others][None, :, None].expand(A.shape[0], -1, 1), -d[:, :, None])
            s = new.max(-1).values.sum(0) + 1e-3 * (new * new).sum((0, 2))
            j = s.argmin()
            if s[j] < best - 1e-6:
                o = others[j].item()
                best, c = s[j], new[:, j].clone()
                place[i], place[o] = place[o].item(), a
                moved += 1
        if not moved:
            break
    return dealt, place


print("\n== rank skew: busiest rank / mean (bytes), scored on held-out prompts ==", flush=True)
if args.no_placement:
    dealt = fitted = identity
else:
    dealt, fitted = zip(*[fit_placement(l) for l in range(L)])
    dealt, fitted = torch.stack(dealt), torch.stack(fitted)
assert all((fitted[l].bincount(minlength=R4) == LOCAL).all() for l in range(L))


def leveled(rows, place, r, layers):
    """Busiest-rank cost with the r most used experts of `layers` replicated on every rank and leveled per step."""
    flex = torch.zeros(L, E, dtype=torch.bool)
    if r:
        top = usage.argsort(1, descending=True)[:, :r]
        flex.scatter_(1, top, True)
        flex[layers:] = False
    c = counts(rows, place, exclude=flex)
    n = ((first[rows] < 8) & flex[None]).sum(-1)  # [n, L] active replicated experts
    for i in range(r):
        low = c.argmin(-1, keepdim=True)  # ties: the lowest rank
        c.scatter_add_(2, low, (n > i).float()[..., None])
    return c


report["skew"] = {}
for name, rows in tests.items():
    base_max, base_mean = cost(counts(rows, identity))
    out = {"shipped": (base_max.sum() / base_mean.sum()).item()}
    out["dealt by usage"] = (cost(counts(rows, dealt))[0].sum() / base_mean.sum()).item()
    out["fitted placement"] = (cost(counts(rows, fitted))[0].sum() / base_mean.sum()).item()
    for r in (4, 8, 16, 32):
        out[f"replicate {r} (NVFP4 layers)"] = (cost(leveled(rows, identity, r, 40))[0].sum() / base_mean.sum()).item()
        out[f"replicate {r} (all layers)"] = (cost(leveled(rows, identity, r, L))[0].sum() / base_mean.sum()).item()
        out[f"fitted + replicate {r} (all layers)"] = (cost(leveled(rows, fitted, r, L))[0].sum() / base_mean.sum()).item()
    report["skew"][name] = out
    print(f"{name} ({len(rows)} steps, busiest-rank {base_max.mean():.0f} MB, mean {base_mean.mean():.0f} MB): "
          + "  ".join(f"{k} {v:.3f}" for k, v in out.items()), flush=True)
print("replicated-expert usage (share of training steps that reach the r-th most used expert of a layer, mean over layers):",
      [round(usage.sort(1, descending=True).values[:, r].mean().item(), 2) for r in (0, 3, 7, 15, 31, 63, 127)])
report["placement"] = fitted.tolist()

# ---- 3. confidence-gated verify rows (single-stream phases)
print("\n== verify only the first K rows, K from the draft's logits (single stream) ==", flush=True)
model = [float(x) for x in args.model.split(",")] if args.model else fit.get(("g1", "max"))
if model and all("conf" in steps[i] for i in sel("g1").nonzero().flatten().tolist()):
    t0, slope = model

    def features(i):
        c = steps[i]["conf"][0]  # [7, 5]: top1 top2 top3 lse drafted
        p1 = (c[:, 4] - c[:, 3]).exp()
        return {"margin": c[:, 0] - c[:, 1], "p": p1, "cumulative p": p1.cumprod(0),
                "min margin so far": (c[:, 0] - c[:, 1]).cummin(0).values}

    def evaluate(rows, feature, tau, kmin=1):
        tokens = time = 0.0
        ks = []
        for i in rows.tolist():
            f = features(i)[feature]
            doubt = (f < tau).nonzero().flatten()
            k = max(kmin, (doubt[0].item() if len(doubt) else 7) + 1)  # anchor + the drafts before the first doubtful
            ks.append(k)
            tokens += min(int(steps[i]["correct"][0]), k - 1) + 1
        ks = torch.tensor(ks)
        for k in ks.unique().tolist():
            part = rows[ks == k]
            time += (t0 + slope * cost(counts(part, identity, k=k))[0]).sum().item()
        return tokens / time * 1e3, ks.float().mean().item()

    report["gating"] = {}
    for ph in ("g1", "s1", "copy", "long"):
        fit_rows, test_rows = (sel(ph, 0) & (bs == 1)).nonzero().flatten(), (sel(ph, 1) & (bs == 1)).nonzero().flatten()
        if len(fit_rows) < 50 or len(test_rows) < 50:
            fit_rows = test_rows = (sel(ph) & (bs == 1)).nonzero().flatten()
            if len(fit_rows) < 50:
                continue
        base, _ = evaluate(test_rows, "p", -1.0)
        correct = torch.tensor([int(steps[i]["correct"][0]) for i in test_rows.tolist()])
        oracle_t = sum((t0 + slope * cost(counts(test_rows[correct == k], identity, k=k + 1))[0]).sum().item()
                       for k in correct.unique().tolist())
        oracle = (correct + 1).sum().item() / oracle_t * 1e3
        out = {"full 8 rows tok/s (model)": round(base, 1), "oracle": round(oracle / base - 1, 3)}
        for feature, grid in (("margin", [0.25, 0.5, 1, 1.5, 2, 3, 4, 6]), ("p", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                              ("cumulative p", [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]),
                              ("min margin so far", [0.25, 0.5, 1, 1.5, 2, 3])):
            for kmin in (1, 2, 3):
                tau = max(grid, key=lambda g: evaluate(fit_rows, feature, g, kmin)[0])
                got, mean_k = evaluate(test_rows, feature, tau, kmin)
                out[f"{feature} >= {tau}, K >= {kmin}"] = [round(got / base - 1, 3), round(mean_k, 2)]
        report["gating"][ph] = out
        print(ph, f"({len(test_rows)} held-out steps)", out, flush=True)
    # how well the draft's own probability predicts acceptance, by position
    rows = (sel("g1") & (bs == 1)).nonzero().flatten().tolist()
    table = []
    for j in range(7):
        reached = [i for i in rows if int(steps[i]["correct"][0]) >= j]
        p = torch.tensor([features(i)["p"][j].item() for i in reached])
        hit = torch.tensor([int(steps[i]["correct"][0]) > j for i in reached])
        bands = [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
        table.append([round(hit.float().mean().item(), 2)] + [
            (int(((p >= a) & (p < b)).sum()), round(hit[(p >= a) & (p < b)].float().mean().item(), 2)) for a, b in bands])
    print("g1: draft j accepted given 1..j-1 accepted: overall, then (count, rate) for draft p in "
          "[0,.3) [.3,.5) [.5,.7) [.7,.9) [.9,1]:")
    for j, row in enumerate(table):
        print(f"  draft {j + 1}: {row}")
    report["calibration"] = table
if args.out:
    json.dump(report, open(args.out, "w"))
