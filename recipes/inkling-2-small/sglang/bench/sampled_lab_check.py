"""What a dump of patches/accept_stats.py says about requests run through patches/sampled_lab.py (pure Python, offline):

    python3 bench/sampled_lab_check.py <8-row dump.jsonl> <lab-token_0.4_0.7- dump.jsonl> <lab-top_0.3_1.0- dump.jsonl>
For each file: the draft probabilities the accept used (q) against softmax(step logits / (alpha T)) for the alpha
the request id asked for and for another one, and whether the verify rows of every step are the rule's. The GPU side
of the lab (the draft graph sampling at the staged temperatures, the accept reading them) is only visible here.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import math
import sys

load = lambda p: [json.loads(line) for line in open(p)]
k8, alpha, top = (load(p) for p in sys.argv[1:4])


def rows_of(score, log_tau):
    k = 1
    for s in score:
        if s < log_tau:
            break
        k += 1
    return k


def cumsum(v):
    out, t = [], 0.0
    for x in v:
        t += x
        out.append(t)
    return out


def draft_probs(name, recs, key, other):
    """the accept's draft probabilities (q) against softmax(step logits / (alpha T)) for two alphas"""
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-12)
    mine = sorted(max(rel(q, x) for q, x in zip(r["q"], r["Q"][key])) for r in recs)
    theirs = sorted(max(rel(q, x) for q, x in zip(r["q"], r["Q"][other])) for r in recs)
    n = len(recs)
    first = sum(r["Q"]["0"][0] for r in recs) / n
    took = sum(1 + r["n"] for r in recs) / n
    print(f"{name}: {n} lines; q against Q[{key}]: median rel {mine[n // 2]:.1e}, 99% {mine[int(n * .99)]:.1e}; against "
          f"Q[{other}]: median {theirs[n // 2]:.1e}; first draft = the draft's best token on {first:.3f} of steps; "
          f"committed {took:.3f} a step at {sum(r['k'] for r in recs) / n:.2f} rows; T {sorted({r['T'] for r in recs})}")


def rule(name, recs, key, tau):
    same = sum(rows_of(cumsum(r[key]), math.log(tau)) == r["k"] for r in recs)
    print(f"{name}: k = the rule on {key} at tau {tau} on {same} of {len(recs)} lines ({same / max(len(recs), 1):.4f}); "
          f"mean rows {sum(r['k'] for r in recs) / max(len(recs), 1):.2f}")


draft_probs("8-row dump (draft at T)", [r for r in k8 if r["T"] == 1.0], "1.0", "0.7")
if alpha:
    draft_probs("lab-token_0.4_0.7 (draft at 0.7 T)", alpha, "0.7", "1.0")
    rule("lab-token_0.4_0.7", alpha, "lq1", 0.4)
if top:
    draft_probs("lab-top_0.3_1.0 (draft at T)", top, "1.0", "0.7")
    rule("lab-top_0.3_1.0", top, "mq1", 0.3)
    rule("  (the serving rule would give)", top, "lq1", 0.4)
print("DONE")
