"""Cold prefill of a boot at a few prompt lengths, comparable between boots: the same prompts, one answer token.

    python3 bench/prefill_ab.py <label> [--tokens 30000,60000,120000] [--rounds 2] [--port 8210]
Each round sends, per length, a record context (bench/chatload.py) that no earlier round of this run used, so every
request is a cold prefill; the contexts depend only on (length, round), so two boots get the same prompts. Before
each request it waits until the server runs nothing (up to 10 minutes) and during it samples /metrics every second:
a request that saw any running request (its own prefill does not count as running) is marked `shared`, and its number is not comparable. Time is the
server's own prefill latency (meta_info), tok/s = prompt tokens / that. Writes logs/prefill-<label>.json.
Ordinary requests (safe on a serving fleet, but a 120K-token prompt occupies it for about a minute).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--tokens", default="30000,60000,120000")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--port", type=int, default=8210)
args = ap.parse_args()
cl.PORT = args.port
BASE = f"http://127.0.0.1:{args.port}"
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


def running():
    try:
        text = urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode()
    except OSError:
        return -1
    return sum(float(line.split()[-1]) for line in text.splitlines() if line.startswith("sglang:num_running_reqs"))


def post(route, body):
    req = urllib.request.Request(BASE + route, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


rows = []
for rnd in range(args.rounds):
    for n in [int(x) for x in args.tokens.split(",")]:
        rec = cl.Records(cl.lines_for(n), 9100003 * (rnd + 1) + n)
        ids = post("/tokenize", {"messages": [{"role": "user", "content": rec.text() + "\n\nReply with OK."}]})["tokens"]
        for _ in range(300):
            if running() == 0:
                break
            time.sleep(2)
        peak, stop = [0.0], threading.Event()

        def sample():
            while not stop.wait(1):
                peak[0] = max(peak[0], running())
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        t0 = time.time()
        out = post("/generate", {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 1}})
        wall = time.time() - t0
        stop.set()
        th.join()
        m = out["meta_info"]
        pre = m.get("prefill_latency") or wall
        row = {"round": rnd, "target": n, "prompt_tokens": m["prompt_tokens"], "cached": m.get("cached_tokens", 0),
               "prefill_s": round(pre, 2), "wall_s": round(wall, 2), "tok_s": round(m["prompt_tokens"] / pre),
               "shared": peak[0] >= 1}  # a request being prefilled is not counted as running: any running one is another
        rows.append(row)
        print(f"round {rnd} {n:>7}: {row['prompt_tokens']} tokens ({row['cached']} cached) in {row['prefill_s']} s = "
              f"{row['tok_s']} tok/s{'  SHARED with another request' if row['shared'] else ''}", flush=True)
os.makedirs(LOGS, exist_ok=True)
json.dump({"label": args.label, "rows": rows}, open(os.path.join(LOGS, f"prefill-{args.label}.json"), "w"), indent=1)
for n in sorted({r["target"] for r in rows}):
    clean = [r["tok_s"] for r in rows if r["target"] == n and not r["shared"] and not r["cached"]]
    print(f"{args.label} {n}: {' / '.join(map(str, clean)) or 'no clean round'} tok/s")
