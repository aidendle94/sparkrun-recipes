"""Exact check (rational arithmetic, no GPU) of block verification under the confidence gate's row cap.

Block verification (Sun et al., arXiv 2403.10444, Algorithm 2) accepts the longest prefix of a drafted block with one
joint test instead of one test per token. The gate (patches/spark_gate.py) gives a step K valid verify rows, K a
function of the drafted tokens themselves, and caps the accept at K - 1 drafts. This enumerates, for a toy Markov
target and draft, the exact distribution of the first GAMMA + 1 emitted tokens under
  token   the stock token-by-token rejection with the cap (what production runs for sampled requests)
  block   block verification over the first min(K, GAMMA) drafts only (a patch that was not kept)
  late    SGLang's block verification over all GAMMA drafts with the cap applied afterwards (rows >= K are garbage)
and compares each with the target's own distribution. All three must match exactly: the probability that a block
keeps at least s drafts is P_s = min(1, P_(s-1) * p_s(x_s) / q_s(x_s)), whatever the rows behind s hold, so late is
block and the server flag is exact under the gate as it stands. The same P_s gives the expected tokens per step.
Usage: python3 bench/bv_exact.py [seeds]
MIT License, Copyright (c) 2026 Aiden Le.
"""
import itertools
import random
import sys
from fractions import Fraction as F

V, GAMMA = 3, 3


def table(rng):
    rows = []
    for _ in range(V + 1):  # previous token (V = the anchor)
        w = [rng.choice([1, 1, 2, 9, 30]) for _ in range(V)]  # peaked rows, so that K varies
        rows.append([F(x, sum(w)) for x in w])
    return rows


def residual(scale, p, q):
    r = [max(scale * a - b, F(0)) for a, b in zip(p, q)]
    return r, sum(r)


def rows_of(x, p, q, g, k):
    """Target row r verifies draft r + 1 and is real only below K; draft row r is the draft's law of draft r + 1."""
    prev = [V] + list(x)
    tp = [(p if r < k else g)[prev[r]] for r in range(GAMMA + 1)]
    dp = [q[prev[r]] for r in range(GAMMA)]
    return tp, dp


def kernel(x, tp, dp, last, block):
    """[(probability, accepted drafts, law of the kernel's final token)] of SGLang's chain kernel."""
    out = []
    if not block:
        reach = F(1)
        for s in range(1, GAMMA + 1):
            a = min(F(1), tp[s - 1][x[s - 1]] / dp[s - 1][x[s - 1]])
            r, z = residual(F(1), tp[s - 1], dp[s - 1])
            if reach * (1 - a):
                out.append((reach * (1 - a), s - 1, [v / z for v in r]))
            reach *= a
        out.append((reach, GAMMA, tp[GAMMA]))
        return out
    prefix, pr = [], F(1)
    for s in range(1, GAMMA + 1):
        pr = min(pr * tp[s - 1][x[s - 1]] / dp[s - 1][x[s - 1]], F(1))
        prefix.append(pr)
    reach = F(1)
    for s in range(last, 0, -1):
        pr = prefix[s - 1]
        h, law = pr, None
        if s < last and 0 < pr < 1:
            r, z = residual(pr, tp[s], dp[s])
            h = z / (z + 1 - pr)
            law = [v / z for v in r] if z else None
        elif s < GAMMA:
            r, z = residual(pr, tp[s], dp[s])
            law = [v / z for v in r] if z else None
        else:
            law = tp[GAMMA]
        if reach * h:
            out.append((reach * h, s, law))
        reach *= 1 - h
    r, z = residual(F(1), tp[0], dp[0])
    if reach:
        out.append((reach, 0, [v / z for v in r]))
    return out


def emitted(mode, p, q, g, tau):
    dist, length, formula = {}, F(0), F(1)
    for x in itertools.product(range(V), repeat=GAMMA):
        prev, qx, cum, k = V, F(1), F(1), 1
        lead = True
        for t in x:
            qx *= q[prev][t]
            cum *= q[prev][t]
            if lead and cum >= tau:
                k += 1
            else:
                lead = False
            prev = t
        tp, dp = rows_of(x, p, q, g, k)
        last = min(k, GAMMA) if mode == "block" else GAMMA
        # what patches/accept_stats.py logs: 1 + the sum over s <= K - 1 of P(at least s drafts accepted)
        pr = F(1)
        for s in range(1, min(k - 1, GAMMA) + 1):
            r = tp[s - 1][x[s - 1]] / dp[s - 1][x[s - 1]]
            pr = pr * min(r, F(1)) if mode == "token" else min(pr * r, F(1))
            formula += qx * pr
        for prob, acc, law in kernel(x, tp, dp, last, mode != "token"):
            n = min(acc, k - 1)
            if acc > n:
                outs = [(F(1), x[:n + 1])]
            else:
                outs = [(law[t], x[:n] + (t,)) for t in range(V) if law[t]]
            for pt, seq in outs:
                length += qx * prob * pt * len(seq)
                # the target itself continues the sequence
                for tail in itertools.product(range(V), repeat=GAMMA + 1 - len(seq)):
                    w, prev = qx * prob * pt, seq[-1]
                    for t in tail:
                        w *= p[prev][t]
                        prev = t
                    dist[seq + tail] = dist.get(seq + tail, F(0)) + w
    return dist, length, formula


def target(p):
    dist = {}
    for seq in itertools.product(range(V), repeat=GAMMA + 1):
        w, prev = F(1), V
        for t in seq:
            w *= p[prev][t]
            prev = t
        dist[seq] = w
    return dist


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    bad = {"token": 0, "block": 0, "late": 0}
    gain, wrong = [], 0
    for seed in range(seeds):
        rng = random.Random(seed)
        p, q, g = table(rng), table(rng), table(rng)
        tau = F(rng.choice([0, 1, 2, 3, 5]), 10)  # 0: no gate, every row is real
        want = target(p)
        lens = {}
        for mode in bad:
            dist, lens[mode], formula = emitted(mode, p, q, g, tau)
            if formula != lens[mode]:
                wrong += 1
                print(f"seed {seed} tau {float(tau)}: {mode} commits {float(lens[mode]):.4f} tokens per step, "
                      f"the formula says {float(formula):.4f}")
            err = max(abs(dist.get(s, F(0)) - want[s]) for s in want)
            bad[mode] += err != 0
            if err:
                print(f"seed {seed} tau {float(tau)}: {mode} is off by {float(err):.3g}")
        gain.append((float(tau), [float(lens[m]) for m in bad]))
    for tau in sorted({t for t, _ in gain}):
        rows = [r for t, r in gain if t == tau]
        mean = [sum(r[i] for r in rows) / len(rows) for i in range(3)]
        print(f"tau {tau}: {len(rows)} toys, tokens per step: token {mean[0]:.3f}, block {mean[1]:.3f}, "
              f"late {mean[2]:.3f}")
    print(f"{seeds} toys: token-wise with the cap inexact on {bad['token']}, block over K drafts inexact on "
          f"{bad['block']}, block over all drafts then capped inexact on {bad['late']}")
    print(f"expected tokens per step from the acceptance ratios alone (patches/accept_stats.py): wrong on {wrong} "
          f"of {3 * seeds}")
    ok = bad["token"] == 0 and bad["block"] == 0 and bad["late"] == 0 and wrong == 0
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


main()
