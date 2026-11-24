"""Offline simulation on the confidence-head capture (CPU, through cc-run.sh, beside cc_lab.py and the saved stores):

1. thresholds by batch size. The head's leading rule adds row j + 1 while the chance that every draft through j is
   accepted beats (ms one more row costs) x (tokens a ms). With w requests in a step, one more row of one request
   costs T_w(K + 1) - T_w(K) over w, and the batch makes w requests' tokens in the same step. Per request that is the
   one-request problem on the table T_w / w: the rule's rate is tuned on each fold's validation rows on that table,
   and the gain is measured on its test rows against the deployed thresholds (the 2026-11-20 one-stream table at rate
   60). --tables: {"1": [T(1), T(2), T(4), T(8)], "4": [...], "8": [...]} in ms a step (all requests at the same K).
2. the context head against the serving head by prompt family (key folds and family folds), on the one-request table.
Prints one line per result and writes <out>/batch-sim.json.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cc_lab  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/d")
ap.add_argument("--out", default="/o")
ap.add_argument("--tables", required=True)
ap.add_argument("--serving", default="combo-mlp-reg")
ap.add_argument("--context", default="combo-lite-reg")
a = ap.parse_args()
G = 7
MS0 = cc_lab.MS                                             # the table the deployed thresholds were made on
RATE0 = 60.0


def full(t):
    """T at K = 1, 2, 4, 8 -> K = 1..8 (linear between the measured points)."""
    t = [float(x) for x in t]
    return np.array([t[0], t[1], (t[1] + t[2]) / 2, t[2], t[2] + (t[3] - t[2]) / 4, t[2] + (t[3] - t[2]) / 2,
                     t[2] + 3 * (t[3] - t[2]) / 4, t[3]])


def leading(pred, rate, ms):
    surv = np.cumprod(pred, 1)
    inc = (ms[1:] - ms[:-1]) * rate / 1000
    return 1 + (surv >= inc[None, :]).cumprod(1).sum(1)


def load(lab, name):
    z = np.load(os.path.join(a.out, f"{name}-store.npz"))
    st = {}
    for key in z.files:
        f, k = key.split("_", 1)
        st.setdefault(int(f[1:]), {})[k] = np.clip(z[key].astype(np.float64), 0, 1)
    return st


def plan(lab):
    return {f: (np.flatnonzero(lab.live & (lab.part == (f + 1) % lab.nfolds)), np.flatnonzero(lab.live & (lab.part == f)))
            for f in range(lab.nfolds)}


def tokens(lab, k, rows):
    cum = np.concatenate([np.zeros((len(rows), 1)), np.cumsum(lab.survive[rows], 1)], 1)
    return 1 + cum[np.arange(len(rows)), k - 1]


def tune(lab, pred, rows, ms, grid):
    return max(grid, key=lambda g: tokens(lab, leading(pred, g, ms), rows).sum() / ms[leading(pred, g, ms) - 1].sum())


tables = {int(w): full(t) for w, t in json.loads(a.tables).items()}
out = {"tables": {w: t.tolist() for w, t in tables.items()}, "thresholds": {}, "batch": {}, "context": {}}
lab = cc_lab.Lab(a.data, fold_by="key")
P = plan(lab)
st = load(lab, a.serving)
folds = sorted(st)
print(f"1. thresholds by batch size ({a.serving}, {len(folds)} folds; deployed = 2026-11-20 one-stream table at rate {RATE0:g}: "
      f"{np.round((MS0[1:] - MS0[:-1]) * RATE0 / 1000, 3).tolist()})", flush=True)
for w, T in sorted(tables.items()):
    ms = T / w                                              # per request
    grid = np.arange(30.0, 260.0, 2.0)
    num0 = den0 = num1 = den1 = 0.0
    rates, cls = [], {}
    for f in folds:
        V, Tt = P[f]
        rate = tune(lab, st[f]["val"], V, ms, grid)
        rates.append(rate)
        k0, k1 = leading(st[f]["test"], RATE0, MS0), leading(st[f]["test"], rate, ms)
        t0, t1 = tokens(lab, k0, Tt), tokens(lab, k1, Tt)
        for name, m in [("all", np.ones(len(Tt), bool))] + [(c, lab.klass[Tt] == c) for c in ("T 1.0", "greedy", "T 0.7")]:
            c = cls.setdefault(name, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
            c[0] += t0[m].sum(); c[1] += ms[k0 - 1][m].sum(); c[2] += t1[m].sum(); c[3] += ms[k1 - 1][m].sum()
            c[4] += k0[m].sum(); c[5] += k1[m].sum(); c[6] += int(m.sum())
    rate = float(np.mean(rates))
    thr = (ms[1:] - ms[:-1]) * rate / 1000
    dep = (MS0[1:] - MS0[:-1]) * RATE0 / 1000
    out["thresholds"][w] = {"rate": rate, "thr": thr.tolist(), "factor": (thr / dep).tolist()}
    out["batch"][w] = {n: {"gain": (c[2] / c[3]) / (c[0] / c[1]), "rows_deployed": c[4] / c[6], "rows_tuned": c[5] / c[6],
                           "steps": c[6]} for n, c in cls.items()}
    print(f"   {w} request(s): rate {rate:6.1f} (folds {[int(r) for r in rates]}) thresholds {np.round(thr, 3).tolist()} "
          f"= deployed x {np.round(thr / dep, 2).tolist()}", flush=True)
    print("      " + "; ".join(f"{n}: x{v['gain']:.4f} (rows {v['rows_deployed']:.2f} -> {v['rows_tuned']:.2f})"
                               for n, v in out["batch"][w].items()), flush=True)

print(f"2. {a.context} against {a.serving} by family, one request, table {np.round(tables[1], 1).tolist()}", flush=True)
for fold_by, suffix in (("key", ""), ("family", "-fam")):
    lab2 = lab if fold_by == "key" else cc_lab.Lab(a.data, fold_by="family")
    P2 = plan(lab2)
    try:
        sa, sb = load(lab2, a.serving + suffix), load(lab2, a.context + suffix)
    except FileNotFoundError as e:
        print(f"   {fold_by} folds: {e}", flush=True)
        continue
    ms = tables[1]
    grid = np.arange(30.0, 100.0, 2.0)
    acc = {}
    for f in sorted(set(sa) & set(sb)):
        V, Tt = P2[f]
        ka = leading(sa[f]["test"], tune(lab2, sa[f]["val"], V, ms, grid), ms)
        kb = leading(sb[f]["test"], tune(lab2, sb[f]["val"], V, ms, grid), ms)
        ta, tb = tokens(lab2, ka, Tt), tokens(lab2, kb, Tt)
        fam = lab2.family[Tt].astype(int)
        for i, name in list(enumerate(lab2.families)) + [(-1, "all")]:
            m = np.ones(len(Tt), bool) if i < 0 else fam == i
            for cl in ("all", "T 1.0"):
                mm = m & (np.ones(len(Tt), bool) if cl == "all" else lab2.klass[Tt] == cl)
                c = acc.setdefault((name, cl), [0.0, 0.0, 0.0, 0.0, 0])
                c[0] += ta[mm].sum(); c[1] += ms[ka - 1][mm].sum(); c[2] += tb[mm].sum(); c[3] += ms[kb - 1][mm].sum()
                c[4] += int(mm.sum())
    res = {f"{n} | {cl}": {"ratio": (c[2] / c[3]) / (c[0] / c[1]), "steps": c[4]} for (n, cl), c in acc.items() if c[4] >= 200}
    out["context"][fold_by] = res
    print(f"   {fold_by} folds: " + "; ".join(f"{n}: x{v['ratio']:.4f} ({v['steps']})" for n, v in res.items()), flush=True)
json.dump(out, open(os.path.join(a.out, "batch-sim.json"), "w"), indent=1)
print("wrote batch-sim.json", flush=True)
