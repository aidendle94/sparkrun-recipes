"""Sustained mixed load with exact answers: does the fleet stay up, and stay right, under an agent's traffic?

    python3 bench/stress.py <label> [--minutes 10] [--workers 8] [--long 0] [--long-tokens 150000] [--cancel 0.1]
                                    [--sizes 1000,4000,12000,30000,60000] [--port 8210] [--stop-file F]
Ordinary chat requests (safe on a serving fleet, but it is a load test: other users wait). Each worker runs
conversations one after another until the time is up. A conversation is a record context (bench/chatload.py: unique
`item | code` lines) of a random size followed by 3-6 turns that extend one history, as an agent does, so prefixes are
found in the cache and the KV pool churns as conversations end. The first `--long` workers use contexts of about
--long-tokens. Turns:
  lookup  codes of two items, greedy: must be exact (a wrong key/value read or another request's context shows here)
  tool    "call report_code for item X": one call, valid JSON arguments, the right item and code
  code    a coding question, 150-500 tokens, a third of them sampled at 0.7: finishes, no loop, no control tokens
  cancel  a code turn whose client closes the connection mid-answer (share --cancel of the code turns)
Half of the requests stream. Meanwhile /metrics (running, queued, tokens in the pool) is sampled every 2 s and /health
every 10 s. Lookups and tool turns that failed are asked once more on the quiet fleet afterwards: wrong again = the
model's own slip, right alone = wrong only under load. A lookup or tool turn that spends its whole token limit
reasoning is soft unless that reasoning loops. Writes logs/stress-<label>.json. Exit 1 on any failure that is neither
soft nor a repeated slip. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--minutes", type=float, default=10)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--long", type=int, default=0)
ap.add_argument("--long-tokens", type=int, default=150000)
ap.add_argument("--cancel", type=float, default=0.1)
ap.add_argument("--sizes", default="1000,4000,12000,30000,60000")
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--stop-file", default="")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
cl.PORT = args.port
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
SIZES = [int(x) for x in args.sizes.split(",")]
TOOL = {"name": "report_code", "description": "Report the code of one item from the records.",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}, "code": {"type": "string"}},
                       "required": ["item", "code"]}}
start, deadline = time.time(), time.time() + args.minutes * 60
lock, rows, retry = threading.Lock(), [], []
peak = {"running": 0, "queued": 0, "used_tokens": 0, "health_s": 0.0, "health_errors": 0}


def stopped():
    return time.time() >= deadline or (args.stop_file and os.path.exists(args.stop_file))


def judge(kind, a, want):
    """None if right, else what is wrong."""
    if a["error"]:
        return a["error"]
    if kind == "cancel":
        return None
    written = a["reasoning"] + a["text"]
    if cl.CONTROL.search(a["text"]):
        return f"control tokens in the text: {cl.CONTROL.findall(a['text'])[:3]}"
    if kind in ("lookup", "tool") and a["finish"] == "length" and not a["text"].strip() and not a["calls"]:
        looping = len(a["reasoning"].split()) >= 60 and cl.d4w(a["reasoning"]) < 0.5
        return (f"{'LOOPS' if looping else 'soft: out of tokens'} while reasoning (distinct 4-grams {cl.d4w(a['reasoning']):.2f}): "
                f"{a['reasoning'][-240:]!r}")
    if kind == "lookup":
        got = cl.codes_in(a["text"])
        return None if got == want else f"wanted {want} got {got} (finish {a['finish']}): {a['text'][-120:]!r}"
    if kind == "tool":
        if len(a["calls"]) != 1 or a["calls"][0][0] != "report_code":
            return f"calls {a['calls']!r:.200} (finish {a['finish']}): {a['text'][-120:]!r}"
        try:
            got = json.loads(a["calls"][0][1])
            got = [str(got.get("item")).replace("item ", ""), str(got.get("code"))]
        except (ValueError, AttributeError):
            return f"arguments are not a JSON object: {a['calls'][0][1]!r:.200}"
        return None if got == want else f"wanted {want} got {got}"
    if a["finish"] not in ("stop", "length"):
        return f"finish {a['finish']!r}"
    if not written.strip():
        return "nothing written"
    if len(written.split()) >= 60 and cl.d4w(written) < 0.5:
        return f"loops (distinct 4-grams {cl.d4w(written):.2f}): {written[-200:]!r}"
    return None


def worker(w):
    rng = random.Random(args.seed * 7919 + w)
    conv = 0
    while not stopped():
        conv += 1
        size = int(args.long_tokens * rng.uniform(0.8, 1.2)) if w < args.long else rng.choice(SIZES)
        rseed = rng.randrange(1 << 30)
        rec = cl.Records(cl.lines_for(size), rseed)
        history, context = [], rec.text() + "\n\n"
        for turn in range(rng.randint(3, 6)):
            if stopped():
                break
            kind = rng.choices(["lookup", "tool", "code"], [45, 25, 30])[0]
            stream, temperature, tools, cancel, want = rng.random() < 0.5, 0, None, None, None
            if kind == "lookup":
                q, want = rec.ask(2)
                limit = 1500
            elif kind == "tool":
                item, code = rec.rows[rng.randrange(len(rec.rows))]
                q, want, tools, limit = f"Call report_code for item {item} with its code from the records.", [str(item), str(code)], [TOOL], 1500
            else:
                q, limit = "Leave the records aside for this one. " + rng.choice(cl.Q), rng.randint(150, 500)
                temperature = 0.7 if rng.random() < 1 / 3 else 0
                if rng.random() < args.cancel:
                    kind, stream, cancel = "cancel", True, rng.randint(5, 40)
            messages = history + [{"role": "user", "content": (context if not history else "") + q}]
            a = cl.chat(messages, limit, temperature, stream, tools, cancel, timeout=1800)
            problem = judge(kind, a, want)
            if problem and kind == "lookup" and a["text"].strip():  # which record lines the wrong codes were read from
                problem += f" | read from: {rec.where(want, cl.codes_in(a['text']))}"
            u = a["usage"]
            row = {"t": round(time.time() - start, 1), "worker": w, "conv": conv, "turn": turn, "kind": kind, "stream": stream,
                   "temperature": temperature, "context": size, "prompt_tokens": u.get("prompt_tokens"),
                   "completion_tokens": u.get("completion_tokens"), "ttft": a["ttft"] and round(a["ttft"], 2),
                   "secs": round(a["secs"], 2), "finish": a["finish"], "problem": problem}
            if problem:
                row.update({"records_seed": rseed, "lines": len(rec.rows), "reasoning_tail": a["reasoning"][-1500:], "text": a["text"][-500:]})
            with lock:
                rows.append(row)
                if problem and kind in ("lookup", "tool") and not a["error"] and len(retry) < 24:
                    retry.append((row, messages, limit, tools, want, rec))
            if kind in ("lookup", "code") and a["text"].strip() and not a["error"]:
                history = messages + [{"role": "assistant", "content": a["text"]}]


def get(route, timeout):
    return urllib.request.urlopen(f"http://localhost:{args.port}/{route}", timeout=timeout).read().decode()


def monitor(done):
    n = 0
    while not done.is_set():
        try:
            for line in get("metrics", 30).splitlines():
                for name, key in (("sglang:num_running_reqs{", "running"), ("sglang:num_queue_reqs{", "queued"),
                                  ("sglang:num_used_tokens{", "used_tokens")):
                    if line.startswith(name):
                        peak[key] = max(peak[key], int(float(line.rsplit(" ", 1)[1])))
        except Exception:
            pass
        if n % 5 == 0:
            t = time.time()
            try:
                get("health", 60)
                peak["health_s"] = max(peak["health_s"], round(time.time() - t, 1))
            except Exception:
                peak["health_errors"] += 1
        if n % 30 == 29:
            with lock:
                print(f"  {int(time.time() - start):5d} s: {len(rows)} requests, {sum(1 for r in rows if r['problem'])} with a problem, "
                      f"peaks {peak}", flush=True)
        n += 1
        done.wait(2)


done = threading.Event()
mon = threading.Thread(target=monitor, args=(done,), daemon=True)
mon.start()
threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.workers)]
for th in threads:
    th.start()
for th in threads:
    th.join()
wall = time.time() - start
done.set()
mon.join()

slips = 0
for row, messages, limit, tools, want, rec in retry:  # the fleet is quiet now
    a = cl.chat(messages, limit, 0, False, tools, timeout=1800)
    row["alone"] = judge(row["kind"], a, want) or "right"
    if row["alone"] != "right" and row["kind"] == "lookup" and a["text"].strip():
        row["alone"] += f" | read from: {rec.where(want, cl.codes_in(a['text']))}"
    slips += row["alone"] != "right"


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


print(f"{len(rows)} requests in {wall:.0f} s from {args.workers} workers ({args.long} long): "
      f"{sum(r['completion_tokens'] or 0 for r in rows)} tokens out, {sum(r['completion_tokens'] or 0 for r in rows) / wall:.1f} tok/s over the run; peaks {peak}")
for kind in ("lookup", "tool", "code", "cancel"):
    k = [r for r in rows if r["kind"] == kind]
    ttft, secs = [r["ttft"] for r in k if r["ttft"]], [r["secs"] for r in k]
    print(f"  {kind:7s} {len(k):4d} requests, {sum(1 for r in k if r['problem']):3d} with a problem; first token p50 {pct(ttft, 0.5)} p95 {pct(ttft, 0.95)} s, "
          f"whole request p50 {pct(secs, 0.5)} p95 {pct(secs, 0.95)} max {max(secs, default=None)} s")
hard = 0
for r in rows:
    if r["problem"]:
        own, soft = r.get("alone") not in (None, "right"), r["problem"].startswith("soft:")
        hard += not (own or soft)
        print(f"  {'slip (wrong alone too)' if own else 'soft' if soft else 'FAIL'} t={r['t']} worker {r['worker']} {r['kind']} context {r['context']} "
              f"prompt {r['prompt_tokens']}: {r['problem'][:330]}" + (f" | alone: {r['alone'][:200]}" if "alone" in r else ""))
json.dump({"args": vars(args), "wall_s": wall, "peaks": peak, "rows": rows}, open(os.path.join(LOGS, f"stress-{args.label}.json"), "w"))
print("ALL OK" if not hard else f"{hard} FAILURE(S)", f"({slips} repeated slips)" if slips else "")
sys.exit(1 if hard else 0)
