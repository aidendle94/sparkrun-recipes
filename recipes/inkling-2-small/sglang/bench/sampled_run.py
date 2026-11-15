"""Sampled requests in a window: speed, tokens per step and the text, per temperature and batch width.

    python3 bench/sampled_run.py <label> [--ref A] [--plan 0.7:1:24,1.0:1:12,0.7:4:24] [--tokens 400] [--port 8210]
A plan entry is temperature:width:prompts. Prompts come from the g1 phase of logs/trace-<ref>.json and are put in
the prefix cache first, so a request times decode only. Per entry: per-stream tok/s (tokens / decode seconds, over
all requests and as the mean of the requests), tokens per step (the server's spec_accept_length), the smallest
distinct-4-gram ratio and the head of every text: sampled output cannot be compared with a reference, so read it.
With SGLANG_DSPARK_DEBUG_DUMP=all on the boot, rank 0's step records (/server_info) give ms per step.
Rows go to logs/sampled-<label>.json. Compare two boots (e.g. with and without --speculative-use-block-verification;
patches/accept_stats.py gives the expected tokens per step of both accept rules from one boot).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--ref", default="A")
ap.add_argument("--plan", default="0.7:1:24,1.0:1:12,0.7:4:24")
ap.add_argument("--tokens", type=int, default=400)
ap.add_argument("--port", type=int, default=8210)
args = ap.parse_args()
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
ref = json.load(open(os.path.join(LOGS, f"trace-{args.ref}.json")))
prompts = {r["prompt"]: r["prompt_ids"] for r in ref if r["phase"] == "g1"}
rows = []


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def d4(o):
    grams = [tuple(o[i:i + 4]) for i in range(len(o) - 3)]
    return len(set(grams)) / max(len(grams), 1)


for entry in args.plan.split(","):
    temp, width, count = entry.split(":")
    names = list(prompts)[:int(count)]
    phase = f"t{temp}w{width}"
    for at in range(0, len(names), int(width)):
        group = names[at:at + int(width)]
        for name in group:
            post("generate", input_ids=prompts[name], rid=f"warm-{phase}-{name}",
                 sampling_params={"temperature": 0, "max_new_tokens": 1})
        outs = {}

        def one(name):
            outs[name] = post("generate", input_ids=prompts[name], rid=f"samp-{phase}-{name}",
                              sampling_params={"temperature": float(temp), "max_new_tokens": args.tokens})
        threads = [threading.Thread(target=one, args=(n,)) for n in group]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for name in group:
            m, o = outs[name]["meta_info"], outs[name]["output_ids"]
            decode_s = m["e2e_latency"] - m["prefill_latency"]
            row = {"phase": phase, "prompt": name, "tokens": m["completion_tokens"], "decode_s": round(decode_s, 2),
                   "tok_s": round(m["completion_tokens"] / decode_s, 1),
                   "accept": round(m.get("spec_accept_length") or 0, 2), "d4": round(d4(o), 2)}
            print(time.strftime("%T"), row, repr(outs[name]["text"][:110]), flush=True)
            rows.append(dict(row, ids=o, text=outs[name]["text"]))
json.dump(rows, open(os.path.join(LOGS, f"sampled-{args.label}.json"), "w"))

steps = {}
try:
    info = json.load(urllib.request.urlopen(f"http://localhost:{args.port}/server_info", timeout=600))
    records = [s.get("dspark_info_record") for s in info.get("internal_states", [])]
    for rec in (records[0] or {}).get("records", []) if records else []:
        rid = rec["reqs"][0]["rid"].split("-")
        if rid[0] == "samp" and all(r["rid"].split("-")[:2] == rid[:2] for r in rec["reqs"]):
            steps.setdefault((rid[1], rec["bs"]), []).append(rec)
except Exception as e:  # the requests' own numbers stand without the step records
    print("no step records:", repr(e)[:200])

print("\n== per temperature and width: per-stream tok/s (all tokens / all decode seconds; mean of requests), tokens "
      "per step, smallest distinct-4-gram ratio")
for phase in dict.fromkeys(r["phase"] for r in rows):
    sel = [r for r in rows if r["phase"] == phase]
    tokens = sum(r["tokens"] for r in sel)
    n_steps = sum(r["tokens"] / r["accept"] for r in sel if r["accept"])
    print(f"{phase:9s} n={len(sel):2d} tokens {tokens:6d} tok/s {tokens / sum(r['decode_s'] for r in sel):5.1f} "
          f"(mean {sum(r['tok_s'] for r in sel) / len(sel):5.1f}) tokens/step {tokens / max(n_steps, 1):4.2f} "
          f"min d4 {min(r['d4'] for r in sel):.2f}", flush=True)
if steps:
    print("\n== steps (rank 0's DSpark records): ms per step, tokens per request and step")
for (phase, bs), sel in steps.items():
    timed = ("step_gpu_ms", "draft_gpu_ms", "target_verify_gpu_ms")
    untimed = sum(1 for r in sel if not all(k in r for k in timed))
    sel = [r for r in sel if all(k in r for k in timed)]  # a record can come without its timers (seen once, boot B)
    n = len(sel)
    if not n:
        print(f"{phase:9s} bs {bs}: {untimed} records, none with timers", flush=True)
        continue
    acc = [r["acc_len"] for rec in sel for r in rec["reqs"]]
    trim = sum(r["cap_trim"] for rec in sel for r in rec["reqs"])
    print(f"{phase:9s} bs {bs} steps {n:5d} step {sum(r['step_gpu_ms'] for r in sel) / n:6.1f} ms "
          f"(draft {sum(r['draft_gpu_ms'] for r in sel) / n:4.1f}, verify "
          f"{sum(r['target_verify_gpu_ms'] for r in sel) / n:5.1f}) tokens/step {sum(acc) / len(acc):4.2f} "
          f"cap-trimmed drafts {trim}", flush=True)
print("DONE")
