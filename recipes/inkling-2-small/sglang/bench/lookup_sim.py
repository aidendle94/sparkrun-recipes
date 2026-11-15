"""What prompt-lookup drafts would add to DSpark on a request log (CPU only, no server needed for rows with prompt_ids).

    python3 bench/lookup_sim.py logs/trace-<label>.json [--phase copy] [--port 8210]
For every greedy request of bench/trace_run.py: walk the output as the server would, and at each step let a lookup
draft propose the 7 tokens that followed the latest earlier occurrence of the last n tokens (n = 8 down to --min) in
prompt + output so far. Under greedy decoding the target accepts the leading proposals that equal the real output, so
the lookup's acceptance is exact; the MTP draft's is only known as the request's mean (tokens per step, bonus
included), and a step that falls back to MTP is charged that mean. Rows written before prompt_ids were stored are
re-tokenized through the server (--port).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--phase", default="")
ap.add_argument("--min", type=int, default=2)
ap.add_argument("--port", type=int, default=8210)
args = ap.parse_args()
GAMMA = 7


def lookup(ctx, lo):
    """(n, proposal): longest suffix of ctx (<= 8 tokens, >= lo) seen earlier, and the tokens after its latest match."""
    for n in range(8, lo - 1, -1):
        if len(ctx) <= n:
            continue
        tail = ctx[-n:]
        for at in range(len(ctx) - n - 1, -1, -1):
            if ctx[at:at + n] == tail:
                return n, ctx[at + n:at + n + GAMMA]
    return 0, []


rows = [r for r in json.load(open(args.log)) if not args.phase or r["phase"] == args.phase]
print(f"{'request':>10} {'tokens':>6} {'mtp':>5} | lookup-only | hybrid by match length n >= 2..8 (tokens/step)")
for r in rows:
    if "prompt_ids" not in r:
        continue
    out, mtp = r["ids"], r["accept"]
    line = []
    for lo in [None] + list(range(args.min, 9)):
        at = steps = took_lookup = 0
        owed = 0.0  # the MTP mean is fractional: carry the remainder
        while at < len(out):
            n, prop = lookup(r["prompt_ids"] + out[:at], lo or args.min)
            ok = 0
            while ok < len(prop) and at + ok < len(out) and prop[ok] == out[at + ok]:
                ok += 1
            if lo is None or n >= lo:  # lookup-only takes whatever it finds (1 token/step when nothing matches)
                at += ok + 1
                took_lookup += 1
            else:
                owed += mtp
                at, owed = at + max(int(owed), 1), owed - max(int(owed), 1)
            steps += 1
        line.append(f"{len(out) / steps:.2f}" + ("" if lo is None else f"({100 * took_lookup // steps}%)"))
    print(f"{r['phase'] + '-' + r['prompt']:>10} {len(out):>6} {mtp:>5.2f} | {line[0]:>11} | " + " ".join(line[1:]))
