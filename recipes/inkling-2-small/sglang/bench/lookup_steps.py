"""Prompt-lookup drafts against the MTP draft, step by step (CPU only): logs of one bench/trace_run.py run on a boot
with SGLANG_DSPARK_DEBUG_DUMP=all.

    python3 bench/lookup_steps.py <label>        # reads logs/trace-<label>.json and logs/dspark-info-<label>.json
At every decode step the server took (greedy phases), the tokens it committed with the MTP draft are known, and what
a lookup draft would have committed is exact: the 7 tokens after the latest earlier occurrence of the last n tokens
of prompt + output, accepted while they equal the real output. Reported per phase: tokens per step of MTP, of a rule
that takes the lookup draft when its match is at least n tokens long, and of an oracle that picks the better draft at
every step. (Positions are those the MTP run visited; a different draft would visit others, so this is first order.)
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import os
import sys
from collections import defaultdict

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
label = sys.argv[1]
GAMMA = 7
rows = {f"{r['phase']}-{r['prompt']}": r for r in json.load(open(os.path.join(LOGS, f"trace-{label}.json")))}
steps = defaultdict(list)
for rec in json.load(open(os.path.join(LOGS, f"dspark-info-{label}.json")))[0]["records"]:
    for q in rec["reqs"]:
        steps[q["rid"]].append(q)


def lookup(ctx):
    """(n, proposal): longest suffix of ctx (2..12 tokens) seen earlier, and the tokens after its latest occurrence."""
    best = (0, [])
    for at in range(len(ctx) - 2, 0, -1):
        if ctx[at] != ctx[-1]:
            continue
        n = 1
        while n < 12 and at - n >= 0 and ctx[at - n] == ctx[-1 - n]:
            n += 1
        if n > best[0]:
            best = (n, ctx[at + 1:at + 1 + GAMMA])
            if n == 12:
                break
    return best


stats = defaultdict(lambda: defaultdict(float))
for rid, row in rows.items():
    if row["phase"] not in ("g1", "copy", "long") or "prompt_ids" not in row:
        continue
    out, plen = row["ids"], len(row["prompt_ids"])
    s = stats[row["phase"]]
    for q in steps.get(rid, []):
        at = q["prefix_len"] - plen + 1  # first output position this step decides
        if at < 0 or at >= len(out) or q["cap_trim"]:
            continue
        n, prop = lookup(row["prompt_ids"] + out[:at])
        hit = 0
        while hit < len(prop) and at + hit < len(out) and prop[hit] == out[at + hit]:
            hit += 1
        mtp, look = q["acc_len"], hit + 1
        s["steps"] += 1
        s["mtp"] += mtp
        s["oracle"] += max(mtp, look)
        for n0 in (2, 3, 4, 6, 8, 12):
            s[f"n>={n0}"] += look if n >= n0 else mtp
            s[f"used n>={n0}"] += n >= n0
        # the two drafts side by side: take the lookup only where the MTP draft's first token agrees with it
        agree = n >= 3 and prop[:1] == q["draft_tokens"][:1]
        s["agree1 n>=3"] += look if agree else mtp
        s["used agree1 n>=3"] += agree
for ph, s in stats.items():
    n = s["steps"]
    print(f"{ph}: {int(n)} steps, MTP {s['mtp'] / n:.2f} tokens/step, oracle choice {s['oracle'] / n:.2f} "
          f"({s['oracle'] / s['mtp'] - 1:+.1%})")
    for k in [k for k in s if k.startswith(("n>=", "agree"))]:
        print(f"   lookup when {k:12s}: {s[k] / n:.2f} ({s[k] / s['mtp'] - 1:+.1%}), used on {s['used ' + k] / n:.0%} of steps")
