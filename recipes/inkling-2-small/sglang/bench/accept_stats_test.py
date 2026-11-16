"""Offline test of patches/accept_stats.py (CPU, a small vocabulary): the running means and the SPARK_ACCEPT_DUMP
lines against a direct computation.

    SPARK_ACCEPT_DUMP=<dir> python3 bench/accept_stats_test.py --patches patches
Needs torch (run it in the image with --runtime runc, no GPU, capped memory; the host has no torch).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import json
import os
import sys
import types

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--patches", required=True)
args = ap.parse_args()
sys.path.insert(0, args.patches)
gate = types.ModuleType("spark_gate")  # the dump reads the draft's step logits where the gate keeps the sampler
gate._s = {"sampler": types.SimpleNamespace(step_logits=None)}
sys.modules["spark_gate"] = gate
import accept_stats  # noqa: E402

bad = 0


def check(name, ok, detail=""):
    global bad
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)


torch.manual_seed(0)
V, G, MAXBS = 500, 7, 8
calls = []
mod = types.SimpleNamespace(_accept_sampling_core=lambda **kw: mod.chain_speculative_sampling_triton(**kw["kernel"]),
                            chain_speculative_sampling_triton=lambda **kw: calls.append(1))
accept_stats.install(mod)
check("dump armed by the environment", not accept_stats._s["dump_off"], accept_stats.DUMP)
want = []
for step, (bs, temp) in enumerate([(1, 0.7), (1, 1.0), (3, 0.7), (8, 1.3)]):
    z = (torch.randn(MAXBS, G, V) * 3).to(torch.bfloat16)  # the draft sampler's static buffer: all 8 requests' rows
    gate._s["sampler"].step_logits = z
    temps = torch.full((bs, 1), temp)
    dp = torch.softmax(z[:bs].float() / temp, -1)
    tp = torch.softmax((z[:bs].float() + torch.randn(bs, G, V)) / temp, -1)
    tp = torch.cat([tp, torch.softmax(torch.randn(bs, 1, V), -1)], 1)  # 8 verify rows
    cand = torch.cat([torch.zeros(bs, 1, dtype=torch.int64), torch.multinomial(dp.view(-1, V), 1).view(bs, G)], 1)
    lens = torch.randint(1, 9, (bs,))
    acc = torch.randint(0, 8, (bs,), dtype=torch.int32)
    mod._accept_sampling_core(cutoff_verify_lens=lens, sampling_info=types.SimpleNamespace(temperatures=temps),
                              kernel=dict(candidates=cand, target_probs=tp, draft_probs=dp, accept_token_num=acc))
    for b in range(bs):
        tok = cand[b, 1:]
        zf = z[b].float()
        alphas, under = {}, {}
        for a in accept_stats.ALPHAS:
            qa = torch.nn.functional.one_hot(zf.argmax(-1), V).float() if a == 0 else torch.softmax(zf / (a * temp), -1)
            alphas["0" if a == 0 else str(a)] = torch.minimum(tp[b, :G], qa).sum(-1)
            under["0" if a == 0 else str(a)] = qa[torch.arange(G), tok]
        want.append({"T": temp, "k": int(lens[b]), "n": int(acc[b]), "x": tok.tolist(),
                     "lq1": torch.log_softmax(zf, -1)[torch.arange(G), tok], "mq1": torch.log_softmax(zf, -1).max(-1).values,
                     "p": tp[b, :G][torch.arange(G), tok], "q": dp[b][torch.arange(G), tok], "A": alphas,
                     "Q": under})
check("the stock kernel ran every step", len(calls) == 4, f"({len(calls)} calls)")
check("no error inside the patch", not accept_stats._s["failed"])
files = glob.glob(os.path.join(accept_stats.DUMP, "accept-*.jsonl"))
got = [json.loads(line) for f in files for line in open(f)]
check("one line per request of every step", len(got) == len(want) == 13, f"({len(got)} lines in {len(files)} file(s))")
close = lambda a, b: bool((torch.tensor(a) - b).abs().max() < 2e-5)
near = lambda a, b: bool(((torch.tensor(a) - b).abs() <= 2e-4 * b.abs() + 1e-12).all())  # five significant digits
for i, (g, w) in enumerate(zip(got, want)):
    ok = (abs(g["T"] - w["T"]) < 1e-4 and g["k"] == w["k"] and g["n"] == w["n"] and g["x"] == w["x"]
          and all(close(g[key], w[key]) for key in ("lq1", "mq1")) and all(near(g[key], w[key]) for key in ("p", "q"))
          and set(g["A"]) == set(w["A"]) == set(g["Q"]) and all(close(g["A"][a], w["A"][a]) for a in w["A"])
          and all(near(g["Q"][a], w["Q"][a]) for a in w["Q"]) and near(g["Q"]["1.0"], w["q"]) and g["chk"] < 1e-5)
    if not ok or i in (0, 12):
        check(f"line {i} (T {g['T']}, k {g['k']})", ok, f"A[1.0] {g['A']['1.0'][:3]} chk {g['chk']}")
one = got[0]
check("alpha 1 accepts 1 - TV, alpha 0 the target's probability of the draft's best token",
      all(0 < v <= 1 for v in one["A"]["1.0"]) and all(0 <= v <= 1 for v in one["A"]["0"]))
gate._s["sampler"].step_logits = torch.zeros(1, G, V - 1)  # a narrower buffer: the dump must stop, not raise
mod._accept_sampling_core(cutoff_verify_lens=None, sampling_info=types.SimpleNamespace(temperatures=torch.ones(1, 1)),
                          kernel=dict(candidates=cand[:1], target_probs=tp[:1], draft_probs=dp[:1],
                                      accept_token_num=acc[:1]))
check("a step-logit buffer of another width turns the dump off without an error",
      accept_stats._s["dump_off"] and not accept_stats._s["failed"])
accept_stats._s.update(file=None, dump_off=False)
gate._s["sampler"].step_logits = z
one_step = lambda: mod._accept_sampling_core(
    cutoff_verify_lens=None, sampling_info=types.SimpleNamespace(temperatures=torch.ones(1, 1)),
    kernel=dict(candidates=cand[:1], target_probs=tp[:1], draft_probs=dp[:1], accept_token_num=acc[:1]))
lines = lambda: sum(1 for f in glob.glob(os.path.join(accept_stats.DUMP, "accept-*.jsonl")) for _ in open(f))
before = (lines(), accept_stats._s["calls"], len(calls))
open(accept_stats.PAUSE, "w").close()
one_step()
check("while <dir>/pause exists nothing is written or counted, and the stock kernel runs",
      (lines(), accept_stats._s["calls"], len(calls)) == (before[0], before[1], before[2] + 1))
os.remove(accept_stats.PAUSE)
one_step()
check("without it the step is written and counted again",
      (lines(), accept_stats._s["calls"]) == (before[0] + 1, before[1] + 1) and not accept_stats._s["failed"])
accept_stats.DUMP = os.path.join(accept_stats.DUMP, "absent")
before = accept_stats._s["calls"]
one_step()
check("a rank without the directory writes nothing, counts nothing and makes nothing",
      accept_stats._s["calls"] == before and not accept_stats._s["failed"] and not os.path.exists(accept_stats.DUMP))
print("ALL OK" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
