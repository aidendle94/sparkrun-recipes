r"""Arms of one boot, side by side: the same prompts under several request-id prefixes and temperatures.

    python3 bench/sampled_arms.py <label> --arms ctl=lab-token_0.4_1.0-:1.0,top=lab-top_0.3_1.0-:1.0 \
        [--prompts 24] [--width 1] [--tokens 400] [--pick spread] [--port 8210] --trace logs/trace-A.json --out <dir>
An arm is name=<request id prefix>:<temperature> (an empty prefix = no request id). Prompts are the g1 prompts of
the trace (12 code, 12 prose); --pick spread takes them alternately from both halves. Every prompt (width 1) or
group of prompts (width > 1) runs under every arm before the next one, the arms' order rotating, so that drift
within the boot falls on all arms alike. Per arm: tok/s per stream (all tokens / all decode seconds), tokens per
step (completion tokens / spec_verify_ct), ms per step, the smallest distinct-4-gram ratio, and against the first
arm the mean per-prompt ratio of tok/s with its standard error. Numbers go to <out>/arms-<label>.json, the heads of
the texts to <out>/arms-<label>.txt.
Request-id prefixes: gate-k<n>- / gate-t<tau>- (patches/spark_gate.py with SPARK_GATE_RID=1) and
lab-<feature>_<tau>_<alpha>- (patches/sampled_lab.py); both are for diagnosis windows. Without prefixes it is an
ordinary load of 400-token requests: fine on a serving fleet, where it delays other clients while it runs.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import math
import os
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--arms", required=True)
ap.add_argument("--prompts", type=int, default=24)
ap.add_argument("--width", type=int, default=1)
ap.add_argument("--tokens", type=int, default=400)
ap.add_argument("--pick", default="spread", choices=("spread", "first"))
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--trace", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

arms = []
for spec in args.arms.split(","):
    name, rest = spec.split("=", 1)
    prefix, temp = rest.rsplit(":", 1)
    arms.append((name, prefix, float(temp)))
g1 = [(r["prompt"], r["prompt_ids"]) for r in json.load(open(args.trace)) if r["phase"] == "g1"]
if args.pick == "spread":
    half = len(g1) // 2
    g1 = [p for pair in zip(g1[:half], g1[half:]) for p in pair]
prompts = g1[:args.prompts]
rows, texts = [], []


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def d4(o):
    grams = [tuple(o[i:i + 4]) for i in range(len(o) - 3)]
    return len(set(grams)) / max(len(grams), 1)


for at in range(0, len(prompts), args.width):
    group = prompts[at:at + args.width]
    for name, ids in group:
        post("generate", input_ids=ids, sampling_params={"temperature": 0, "max_new_tokens": 1})
    turn = at // args.width
    for arm, prefix, temp in arms[turn % len(arms):] + arms[:turn % len(arms)]:
        outs = {}

        def one(name, ids):
            rid = {"rid": f"{prefix}{args.label}-{arm}-{name}"} if prefix else {}
            outs[name] = post("generate", input_ids=ids, **rid,
                              sampling_params={"temperature": temp, "max_new_tokens": args.tokens})
        threads = [threading.Thread(target=one, args=p) for p in group]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for name, _ in group:
            m, o = outs[name]["meta_info"], outs[name]["output_ids"]
            decode_s = m["e2e_latency"] - m["prefill_latency"]
            steps = m.get("spec_verify_ct") or 0
            row = {"arm": arm, "prompt": name, "tokens": m["completion_tokens"], "steps": steps,
                   "decode_s": round(decode_s, 3), "d4": round(d4(o), 3)}
            rows.append(row)
            texts.append(f"{arm} {name} d4 {row['d4']}: {outs[name]['text'][:160]!r}")
        done = [r for r in rows if r["arm"] == arm]
        print(time.strftime("%T"), f"{arm:10s} {group[0][0]}.. tokens {sum(r['tokens'] for r in rows[-len(group):])} "
              f"running tok/s {sum(r['tokens'] for r in done) / sum(r['decode_s'] for r in done):5.1f}", flush=True)
os.makedirs(args.out, exist_ok=True)
json.dump({"args": vars(args), "rows": rows}, open(os.path.join(args.out, f"arms-{args.label}.json"), "w"))
open(os.path.join(args.out, f"arms-{args.label}.txt"), "w").write("\n".join(texts) + "\n")

print(f"\n== {args.label}: width {args.width}, {len(prompts)} prompts, {args.tokens} tokens; per stream")
base = {r["prompt"]: r["tokens"] / r["decode_s"] for r in rows if r["arm"] == arms[0][0]}
for arm, prefix, temp in arms:
    sel = [r for r in rows if r["arm"] == arm]
    tokens, secs, steps = (sum(r[k] for r in sel) for k in ("tokens", "decode_s", "steps"))
    ratios = [r["tokens"] / r["decode_s"] / base[r["prompt"]] for r in sel]
    mean = sum(ratios) / len(ratios)
    se = math.sqrt(sum((x - mean) ** 2 for x in ratios) / max(len(ratios) - 1, 1) / len(ratios))
    print(f"{arm:10s} T {temp:<4} {prefix or '(no id)':24s} tok/s {tokens / secs:5.1f}  tokens/step "
          f"{tokens / max(steps, 1):5.3f}  ms/step {1000 * secs / max(steps, 1):5.1f}  vs {arms[0][0]} x{mean:5.3f} "
          f"+-{se:5.3f}  min d4 {min(r['d4'] for r in sel):4.2f}  short {sum(r['tokens'] < args.tokens for r in sel)}",
          flush=True)
print("DONE")
