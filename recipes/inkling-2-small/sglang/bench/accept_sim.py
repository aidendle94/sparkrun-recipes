"""Other gate rules and other draft temperatures on a trace of sampled DSpark steps (offline, pure Python).

    python3 bench/accept_sim.py <accept-*.jsonl>... [--tau 0.4] [--c1 5.0] [--c0 38.7] [--ms 0.7=66.15,1.0=63.14]
    python3 bench/accept_sim.py --check <accept-*.jsonl>...
Input: the lines patches/accept_stats.py writes with SPARK_ACCEPT_DUMP, from requests that ran with 8 verify rows
(SPARK_GATE_RID and a "gate-k8-" request id: every draft met a real target row). Lines with fewer rows are counted
and left out. --check instead reads the lines of an ordinary gated run and compares each k with the running rule
recomputed here (they must agree: the simulator and the engine apply the same rule).

Sampled requests accept drafts by rejection sampling: draft s is taken with probability a_s = min(1, p_s / q_s) and
the first refusal ends the step. With K verify rows a step commits 1 + min(accepted, K - 1) tokens, so row s (1..7)
is worth P(drafts 1..s all taken) = a_1 .. a_s of a token and costs --c1 ms. Everything below is the expectation
over the accept's coins, per step, from the probabilities of the trace: no coin noise, every rule on the same steps.

  gate rules   row s stays while the running sum over drafts 1..s of a log feature is >= log tau. Any feature of
               the drafted tokens 1..s and of the draft's distributions is exact (the choice of testing draft s + 1
               must not depend on that draft's own token):
                 token      log probability of the drafted token at temperature 1 (the running rule)
                 token@T    the same at the request's temperature
                 top        log probability of the draft's best token at temperature 1, whatever was drafted
                 top+       top, with the position the row tests included (drafts 1..s+1)
                 mix        the mean of token and top
               "ceiling" keeps row s when a_1 .. a_s itself is high: it reads the target, no gate can do it.
  alpha        the draft samples at alpha x T and the accept knows it (exact for any alpha): position s then
               accepts 1 - TV(target at T, draft at alpha T) = A[alpha]. The trace's drafts were sampled at T, so
               a step's drafts are re-weighted by Q[alpha] / q per draft ("ess" = the effective number of steps of
               those weights; "path" = no re-weighting, the product of A along the drafted path, biased).
Step time: c0 + c1 x rows (one request alone; bench/gate_run.py). With --ms, c0 is fitted so that the running rule
gives the measured ms a step of that temperature. Standard errors treat steps as independent.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--tau", type=float, default=0.4, help="the running gate's tau")
ap.add_argument("--c1", type=float, default=5.0, help="ms of a step per verify row")
ap.add_argument("--c0", type=float, default=38.7, help="ms of a step before its rows")
ap.add_argument("--ms", default="", help="<T>=<ms a step under the running gate>,...: fits c0 per temperature")
ap.add_argument("--check", action="store_true", help="an ordinary gated run: k against the running rule")
args = ap.parse_args()
TAUS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
G = 7
measured = {round(float(t), 2): float(ms) for t, ms in (kv.split("=") for kv in args.ms.split(",") if kv)}


def cumsum(values):
    out, total = [], 0.0
    for v in values:
        total += v
        out.append(total)
    return out


FEATURES = {
    "token": lambda r: cumsum(r["lq1"]),
    "token@T": lambda r: cumsum([math.log(max(q, 1e-30)) for q in r["q"]]),
    "top": lambda r: cumsum(r["mq1"]),
    "top+": lambda r: (lambda c: c[1:] + c[-1:])(cumsum(r["mq1"])),
    "mix": lambda r: cumsum([(a + b) / 2 for a, b in zip(r["lq1"], r["mq1"])]),
}


def rows_of(score, log_tau):
    """K: 1 + the leading rows whose score is >= log tau (scores never rise)."""
    k = 1
    for s in score:
        if s < log_tau:
            break
        k += 1
    return k


records, broken = {}, 0
for name in args.files:
    for line in open(name):
        try:
            rec = json.loads(line)
        except ValueError:  # a line cut by the copy
            broken += 1
            continue
        records.setdefault(round(rec["T"], 2), []).append(rec)
if broken:
    print(f"{broken} unreadable line(s) left out")

if args.check:
    for temp, recs in sorted(records.items()):
        same = sum(rows_of(cumsum(r["lq1"]), math.log(args.tau)) == r["k"] for r in recs)
        near = sum(abs(rows_of(cumsum(r["lq1"]), math.log(args.tau)) - r["k"]) <= 1 for r in recs)
        print(f"T={temp:.2f}: {len(recs)} steps, k equals the rule (tau {args.tau}) recomputed here on {same} "
              f"({near} within one row), mean k {sum(r['k'] for r in recs) / len(recs):.3f}, committed "
              f"{sum(1 + min(r['n'], r['k'] - 1) for r in recs) / len(recs):.3f} tokens a step, chk max "
              f"{max(r['chk'] for r in recs):.6f}")
    raise SystemExit(0 if all(sum(rows_of(cumsum(r["lq1"]), math.log(args.tau)) == r["k"] for r in recs) >= 0.995 * len(recs)
                              for recs in records.values()) else 1)


def rate(xs, ys):
    return 1e3 * sum(xs) / sum(ys)


def gain(xs, ys, x0, y0):
    """rate of (xs, ys) over rate of (x0, y0), with the standard error of the ratio (paired steps, delta method)."""
    n = len(xs)
    mx, my, m0, n0 = sum(xs) / n, sum(ys) / n, sum(x0) / n, sum(y0) / n
    infl = [x / mx - y / my - a / m0 + b / n0 for x, y, a, b in zip(xs, ys, x0, y0)]
    mean = sum(infl) / n
    se = (sum((v - mean) ** 2 for v in infl) / max(n - 1, 1) / n) ** 0.5
    ratio = (mx / my) / (m0 / n0)
    return ratio, ratio * se


for temp, every in sorted(records.items()):
    recs = [r for r in every if r["k"] == G + 1]
    print(f"\nT={temp:.2f}: {len(recs)} steps with 8 rows ({len(every) - len(recs)} other lines left out), chk max "
          f"{max(r['chk'] for r in every):.6f}")
    if len(recs) < 50:
        print("  too few steps")
        continue
    n = len(recs)
    acc = [[min(1.0, p / q) if q > 0 else 0.0 for p, q in zip(r["p"], r["q"])] for r in recs]
    cum = []  # cum[i][s - 1] = P(drafts 1..s all taken) of step i
    for a in acc:
        prod, row = 1.0, []
        for v in a:
            prod *= v
            row.append(prod)
        cum.append(row)
    took = sum(1 + r["n"] for r in recs) / n
    full = [1 + sum(c) for c in cum]
    se_full = (sum((v - sum(full) / n) ** 2 for v in full) / (n - 1) / n) ** 0.5
    print(f"  the trace: committed {took:.3f} tokens a step with all 8 rows; expected from its ratios "
          f"{sum(full) / n:.3f} (se {se_full:.3f})")
    print("  accept by position, mean of min(1, p/q): " + " ".join(f"{sum(a[s] for a in acc) / n:.3f}" for s in range(G))
          + "   | mean of A[1.0]: " + " ".join(f"{sum(r['A']['1.0'][s] for r in recs) / n:.3f}" for s in range(G)))
    scores = {name: [f(r) for r in recs] for name, f in FEATURES.items()}
    k0 = [rows_of(s, math.log(args.tau)) for s in scores["token"]]
    c0 = args.c0
    if temp in measured:
        c0 = measured[temp] - args.c1 * sum(k0) / n
    x0 = [1 + sum(c[: k - 1]) for c, k in zip(cum, k0)]
    y0 = [c0 + args.c1 * k for k in k0]
    print(f"  step time = {c0:.1f} + {args.c1:.1f} ms x rows" + (f" (c0 fitted to the measured {measured[temp]:.2f} ms a step)" if temp in measured else "")
          + f"; the running rule (token, tau {args.tau}): {sum(k0) / n:.2f} rows, {sum(x0) / n:.3f} tokens, "
          f"{sum(y0) / n:.2f} ms a step, {rate(x0, y0):.2f} tok/s")

    def line(label, xs, ks, note=""):
        ys = [c0 + args.c1 * k for k in ks]
        ratio, se = gain(xs, ys, x0, y0)
        return (f"    {label:<22} rows {sum(ks) / n:5.2f}  tokens {sum(xs) / n:.3f}  {sum(ys) / n:6.2f} ms  "
                f"{rate(xs, ys):6.2f} tok/s  x{ratio:.3f} (se {se:.3f}){note}")
    print("  gate rules, each at its tau grid (x = tok/s over the running rule's, same steps):")
    print(line("all 8 rows", full, [G + 1] * n))
    print(line("1 row", [1.0] * n, [1] * n))
    half = [i % 2 == 0 for i in range(n)]
    for name, sc in scores.items():
        table = {}
        for tau in TAUS:
            ks = [rows_of(s, math.log(tau)) for s in sc]
            table[tau] = ([1 + sum(c[: k - 1]) for c, k in zip(cum, ks)], ks)
        best = max(TAUS, key=lambda t: rate(table[t][0], [c0 + args.c1 * k for k in table[t][1]]))
        for tau in TAUS:
            if tau in (best, args.tau) or abs(TAUS.index(tau) - TAUS.index(best)) == 1:
                print(line(f"{name} tau {tau}", *table[tau], "  <- best of the rule" if tau == best else ""))
        # the tau chosen on the even steps, scored on the odd ones
        pick = max(TAUS, key=lambda t: rate([x for x, h in zip(table[t][0], half) if h],
                                            [c0 + args.c1 * k for k, h in zip(table[t][1], half) if h]))
        odd = lambda v: [a for a, h in zip(v, half) if not h]
        ratio, se = gain(odd(table[pick][0]), [c0 + args.c1 * k for k in odd(table[pick][1])], odd(x0), odd(y0))
        print(f"      {name}: tau {pick} chosen on the even steps scores x{ratio:.3f} (se {se:.3f}) on the odd steps")
    top = None
    for theta in [i / 50 for i in range(1, 50)]:
        ks = [1 + sum(v >= theta for v in c) for c in cum]  # a_1..a_s never rises with s
        xs = [1 + sum(c[: k - 1]) for c, k in zip(cum, ks)]
        if top is None or rate(xs, [c0 + args.c1 * k for k in ks]) > top[0]:
            top = (rate(xs, [c0 + args.c1 * k for k in ks]), theta, xs, ks)
    print(line(f"ceiling (a >= {top[1]:.2f})", top[2], top[3], "  <- reads the target: a bound, not a rule"))

    print("  draft temperature alpha x T (re-weighted steps; x = tok/s over the running rule at alpha 1):")
    alphas = sorted(recs[0]["A"], key=float)
    for alpha in alphas:
        w, v, path, rb = [], [], [], []
        for r, c in zip(recs, cum):
            pw, pv, pp, ws, vs, rbs, ps = 1.0, 1.0, 1.0, [], [], 0.0, 0.0
            for s in range(G):
                rbs += pw * r["A"][alpha][s]  # drafts 1..s-1 as drafted and taken, draft s over the whole vocabulary
                q = r["q"][s]
                pw *= min(r["Q"][alpha][s], r["p"][s]) / q if q > 0 else 0.0
                pv *= r["Q"][alpha][s] / q if q > 0 else 0.0
                pp *= r["A"][alpha][s]
                ws.append(pw)
                vs.append(pv)
                ps += pp
            w.append(ws)
            v.append(vs)
            path.append(1 + ps)
            rb.append(1 + rbs)
        ess = sum(x[-1] for x in v) ** 2 / max(sum(x[-1] ** 2 for x in v), 1e-30)
        per_pos = " ".join(f"{sum(r['A'][alpha][s] for r in recs) / n:.2f}" for s in range(G))
        print(f"    alpha {alpha:<4} A by position {per_pos}  | all 8 rows: tokens {sum(rb) / n:.3f} re-weighted, "
              f"{sum(path) / n:.3f} path  | ess {ess:.0f} of {n}, mean weight {sum(x[-1] for x in v) / n:.3f}")
        shown = None
        for tau in TAUS:
            kept = [[s >= math.log(tau) for s in sc] for sc in scores["token"]]
            xs = [1 + sum(a for a, k in zip(ws, kp) if k) for ws, kp in zip(w, kept)]
            ks = [1 + sum(a for a, k in zip(vs, kp) if k) for vs, kp in zip(v, kept)]
            got = (rate(xs, [c0 + args.c1 * k for k in ks]), tau, xs, ks)
            if tau == args.tau:
                print("  " + line(f"token tau {tau}", xs, ks))
            if shown is None or got[0] > shown[0]:
                shown = got
        if shown[1] != args.tau:
            print("  " + line(f"token tau {shown[1]}", shown[2], shown[3], "  <- best tau at this alpha"))
