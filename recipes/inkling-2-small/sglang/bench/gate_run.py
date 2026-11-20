"""Measure the confidence gate (patches/spark_gate.py) in a window: one boot with
NCCL_EXTRA="-e SPARK_GATE=0.4 -e SPARK_GATE_RID=1 -e SGLANG_DSPARK_DEBUG_DUMP=all" (never the serving default: the
request-id overrides let any client choose its K).

    python3 bench/gate_run.py <label> [--ref A] [--taus 0.25,0.4,0.6] [--ks 1,2,3,4,6] [--tokens 400] [--phases ...]
                              [--limit N: the first N prompts of g1, g4 and g8] [--head]
--head (a boot with SPARK_GATE_HEAD, patches/spark_gate_head.py): every phase also runs the arm "h1", the learned
confidence head's row count, and reports its paired ratio to the middle tau's arm (the gate's own rule, same boot).
Prompts and reference outputs come from logs/trace-<ref>.json (bench/trace_run.py on a boot without the gate). Every
request names its arm in its id: "gate-k8-..." verifies all 8 rows (the stock step), "gate-t<tau>-..." gates at tau,
"gate-k<n>-..." always verifies n rows. The arms of a phase run back to back on the same prompts (their order rotates,
the prompt is in the prefix cache before the first), so a paired per-prompt ratio is free of thermal drift:
  g1    each prompt alone, greedy: k8 and every tau. A greedy stream must not depend on the arm (the gate only decides
        how much of the target's own continuation one step commits): `same` counts outputs identical to k8's
  kc    six prompts alone, greedy, with a forced row count: step time against K
  s1    alone, temperature 0.7         g4, g8   four and eight at a time, greedy
  s10   alone, temperature 1.0 (what a request without a temperature gets)      s4   four at a time, temperature 1.0
  copy  alone, greedy, an edit of a given file            long   alone, greedy, behind a 60K-token prompt
Per phase and arm: per-stream tok/s, accepted length, paired speed-up over k8, `same`, the smallest distinct-4-gram
ratio; then from the server's DSpark step records (/server_info) ms per step (draft, verify) and tokens per step.
Rows go to logs/gate-<label>.json, the step records to logs/dspark-info-<label>.json.
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
ap.add_argument("--taus", default="0.25,0.4,0.6")
ap.add_argument("--ks", default="1,2,3,4,6")
ap.add_argument("--tokens", type=int, default=400)
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--phases", default="g1,kc,s1,g4,g8,copy,long")
ap.add_argument("--limit", type=int)
ap.add_argument("--head", action="store_true", help="also the arm h1: the confidence head's rows (SPARK_GATE_HEAD boots)")
args = ap.parse_args()
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
ref = json.load(open(os.path.join(LOGS, f"trace-{args.ref}.json")))
taus = [f"t{t}" for t in args.taus.split(",")]
mid = taus[len(taus) // 2]
head = ["h1"] if args.head else []
rows = []


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def d4(o):
    grams = [tuple(o[i:i + 4]) for i in range(len(o) - 3)]
    return len(set(grams)) / max(len(grams), 1)


def common(a, b):
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return n, len(a) == len(b) == n


def run(phase, width, arms, tokens=None, source=None, limit=None, **sampling):
    prompts = {r["prompt"]: r["prompt_ids"] for r in ref if r["phase"] == (source or phase)}
    names = list(prompts)[:limit]
    params = dict({"temperature": 0, "max_new_tokens": tokens or args.tokens}, **sampling)
    for g, at in enumerate(range(0, len(names), width)):
        group = names[at:at + width]
        for name in group:  # into the prefix cache, so that no arm pays the prefill
            post("generate", input_ids=prompts[name], rid=f"warm-{phase}-{name}",
                 sampling_params={"temperature": 0, "max_new_tokens": 1})
        for arm in arms[g % len(arms):] + arms[:g % len(arms)]:
            outs = {}

            def one(name):
                outs[name] = post("generate", input_ids=prompts[name], rid=f"gate-{arm}-{phase}-{name}",
                                  sampling_params=params)
            threads = [threading.Thread(target=one, args=(n,)) for n in group]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            for name in group:
                m, o = outs[name]["meta_info"], outs[name]["output_ids"]
                decode_s = m["e2e_latency"] - m["prefill_latency"]
                row = {"phase": phase, "arm": arm, "prompt": name, "width": len(group), "tokens": m["completion_tokens"],
                       "decode_s": round(decode_s, 2), "tok_s": round(m["completion_tokens"] / decode_s, 1),
                       "accept": round(m.get("spec_accept_length") or 0, 2), "d4": round(d4(o), 2)}
                print(time.strftime("%T"), row, repr(outs[name]["text"][:50]), flush=True)
                rows.append(dict(row, ids=o, text=outs[name]["text"]))
            json.dump(rows, open(os.path.join(LOGS, f"gate-{args.label}.json"), "w"))  # a run that is cut keeps its rows


phases = args.phases.split(",")
if "g1" in phases:
    run("g1", 1, ["k8"] + taus + head, limit=args.limit)
if "kc" in phases:
    run("kc", 1, [f"k{k}" for k in args.ks.split(",")] + ["k8"], tokens=256, source="g1", limit=6)
if "s1" in phases:
    run("s1", 1, ["k8", mid] + head, source="g1", limit=min(16, args.limit or 16), temperature=0.7)
if "s10" in phases:
    run("s10", 1, ["k8", mid] + head, source="g1", limit=args.limit, temperature=1.0)
if "g4" in phases:
    run("g4", 4, ["k8", mid] + head, source="g1", limit=args.limit)
if "s4" in phases:
    run("s4", 4, ["k8", mid] + head, source="g1", limit=args.limit, temperature=1.0)
if "g8" in phases:
    run("g8", 8, ["k8", mid] + head, source="g1", limit=args.limit)
if "copy" in phases:
    run("copy", 1, ["k8"] + taus + head, tokens=600)
if "long" in phases:
    run("long", 1, ["k8"] + taus + head)
json.dump(rows, open(os.path.join(LOGS, f"gate-{args.label}.json"), "w"))

print("\n== requests: per-stream tok/s; x = mean per-prompt ratio to k8; same = outputs identical to k8's "
      "(mean common prefix)")
for phase in dict.fromkeys(r["phase"] for r in rows):
    base = {r["prompt"]: r for r in rows if r["phase"] == phase and r["arm"] == "k8"}
    for arm in dict.fromkeys(r["arm"] for r in rows if r["phase"] == phase):
        sel = [r for r in rows if r["phase"] == phase and r["arm"] == arm]
        pairs = [(r, base[r["prompt"]]) for r in sel if r["prompt"] in base]
        ratio = sum(r["tok_s"] / b["tok_s"] for r, b in pairs) / max(len(pairs), 1)
        cmp = [common(r["ids"], b["ids"]) for r, b in pairs]
        line = (f"{phase:5s} {arm:6s} n={len(sel):2d} tok/s {sum(r['tok_s'] for r in sel) / len(sel):5.1f} "
                f"accept {sum(r['accept'] for r in sel) / len(sel):4.2f} x{ratio:5.3f} "
                f"same {sum(s for _, s in cmp):2d}/{len(cmp)} ({sum(n for n, _ in cmp) / max(len(cmp), 1):5.0f}) "
                f"min d4 {min(r['d4'] for r in sel):.2f}")
        if phase == "g1" and arm == "k8":
            want = {r["prompt"]: r["ids"] for r in ref if r["phase"] == "g1"}
            cmp = [common(r["ids"], want[r["prompt"]]) for r in sel]
            line += (f" | vs trace-{args.ref}: same {sum(s for _, s in cmp)}/{len(cmp)} "
                     f"({sum(n for n, _ in cmp) / len(cmp):.0f})")
        if arm == "h1":  # the head against the gate's own rule, by prompt (a width-4 group shares its steps: per group)
            today = {r["prompt"]: r for r in rows if r["phase"] == phase and r["arm"] == mid}
            rs = [r["tok_s"] / today[r["prompt"]]["tok_s"] for r in sel if r["prompt"] in today]
            if rs:
                mean = sum(rs) / len(rs)
                se = (sum((x - mean) ** 2 for x in rs) / max(len(rs) - 1, 1) / len(rs)) ** 0.5
                tot = sum(r["tokens"] for r in sel) / sum(r["decode_s"] for r in sel)
                tot_t = sum(today[r["prompt"]]["tokens"] for r in sel if r["prompt"] in today) / \
                    sum(today[r["prompt"]]["decode_s"] for r in sel if r["prompt"] in today)
                line += f" | vs {mid}: x{mean:5.3f} +- {se:.3f} per prompt, x{tot / tot_t:5.3f} over all tokens"
        print(line, flush=True)

info = json.load(urllib.request.urlopen(f"http://localhost:{args.port}/server_info", timeout=600))
records = [s.get("dspark_info_record") for s in info.get("internal_states", [])]
json.dump(records, open(os.path.join(LOGS, f"dspark-info-{args.label}.json"), "w"))
steps = {}
for rec in (records[0] or {}).get("records", []) if records else []:
    rid = rec["reqs"][0]["rid"].split("-")
    if rid[0] != "gate" or any(r["rid"].split("-")[1:3] != rid[1:3] for r in rec["reqs"]):
        continue
    steps.setdefault((rid[2], rid[1], rec["bs"]), []).append(rec)
print("\n== steps (rank 0's DSpark records): ms per step, tokens per request and step")
for (phase, arm, bs), sel in steps.items():
    n = len(sel)
    acc = [r["acc_len"] for rec in sel for r in rec["reqs"]]
    trim = sum(r["cap_trim"] for rec in sel for r in rec["reqs"])
    print(f"{phase:5s} {arm:6s} bs {bs} steps {n:5d} step {sum(r['step_gpu_ms'] for r in sel) / n:6.1f} ms "
          f"(draft {sum(r['draft_gpu_ms'] for r in sel) / n:4.1f}, verify {sum(r['target_verify_gpu_ms'] for r in sel) / n:5.1f}) "
          f"tokens/step {sum(acc) / len(acc):4.2f} cap-trimmed drafts {trim}", flush=True)
