"""Does the fleet stay right when the windowed (SWA) KV pool runs full?

    python3 bench/swa_fill.py <label> [--convs 64] [--tokens 1500] [--workers 8] [--port 8210]
Ordinary greedy chat requests (safe on a serving fleet; it evicts other conversations' cached prefixes, as any load
does). A cached prompt keeps the keys of its last window in the windowed pool, which holds
`--swa-full-tokens-ratio` slots per token of the full pool (0.1 stock): many short conversations fill it long before
the full pool, and the engine then has to evict cached prefixes to serve a new request. This sends `--convs` distinct
record contexts (bench/chatload.py) of about --tokens each with one exact lookup of two items, --workers at a time,
then the first --workers of them again (old: evicted or not, the answer must be the same) and the last ones again
(recent). /metrics is sampled every second: peak share of the full pool, of the windowed pool and of the state
slots, and the tokens evicted. Writes logs/swafill-<label>.json. Exit 1 if a lookup is wrong or a request fails.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--convs", type=int, default=64)
ap.add_argument("--tokens", type=int, default=1500)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--port", type=int, default=8210)
args = ap.parse_args()
cl.PORT = args.port
GAUGES = {"sglang:full_token_usage{": "full", "sglang:swa_token_usage{": "swa", "sglang:mamba_usage{": "state",
          "sglang:evicted_tokens_total{": "evicted", "sglang:max_total_num_tokens{": "pool",
          "sglang:max_total_num_tokens_swa{": "swa_pool", "sglang:swa_used_tokens{": "swa_used",
          "sglang:swa_evictable_tokens{": "swa_cached", "sglang:swa_available_tokens{": "swa_free"}
peak, first, low = {}, {}, {}


def monitor(done):
    while not done.is_set():
        try:
            text = urllib.request.urlopen(f"http://localhost:{args.port}/metrics", timeout=30).read().decode()
            for line in text.splitlines():
                for name, key in GAUGES.items():
                    if line.startswith(name):
                        v = float(line.rsplit(" ", 1)[1])
                        first.setdefault(key, v)
                        peak[key] = max(peak.get(key, 0), v)
                        low[key] = min(low.get(key, v), v)
        except Exception:
            pass
        done.wait(1)


def ask(j, tag):
    rec = convs[j]
    q, want = questions[j]
    a = cl.chat([{"role": "user", "content": rec.text() + "\n\n" + q}], 4000, stream=False)
    got = cl.codes_in(a["text"])
    u = a["usage"] or {}
    row = {"conv": j, "pass": tag, "ok": got == want and not a["error"], "want": want, "got": got[:6], "error": a["error"],
           "finish": a["finish"], "prompt": u.get("prompt_tokens"), "secs": round(a["secs"], 1),
           "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
    if not row["ok"]:
        print(f"  NOT RIGHT {tag} conv {j}: want {want} got {got[:6]} finish {a['finish']} error {a['error']}", flush=True)
    return row


lines = cl.lines_for(args.tokens)
convs = [cl.Records(lines, 9000 + int(time.time()) % 100000 * 1000 + j) for j in range(args.convs)]
questions = [c.ask(2) for c in convs]
done = threading.Event()
threading.Thread(target=monitor, args=(done,), daemon=True).start()
start, rows = time.time(), []
with ThreadPoolExecutor(args.workers) as ex:
    rows += list(ex.map(lambda j: ask(j, "fill"), range(args.convs)))
    print(f"fill: {args.convs} conversations of {lines} lines in {time.time() - start:.0f} s, peaks {peak}", flush=True)
    rows += list(ex.map(lambda j: ask(j, "old"), range(args.workers)))
    rows += list(ex.map(lambda j: ask(j, "recent"), range(args.convs - args.workers, args.convs)))
done.set()
for tag in ("fill", "old", "recent"):
    part = [r for r in rows if r["pass"] == tag]
    cached = [r["cached"] or 0 for r in part]
    print(f"{tag:6s} {sum(r['ok'] for r in part)}/{len(part)} right, prompt tokens {part[0]['prompt']}, cached tokens "
          f"min {min(cached)} max {max(cached)}", flush=True)
evicted = peak.get("evicted", 0) - first.get("evicted", 0)
print(f"pool {peak.get('pool', 0):.0f} tokens, windowed pool {peak.get('swa_pool', 0):.0f}; peak share full "
      f"{peak.get('full', 0):.2f}, windowed {peak.get('swa', 0):.2f} ({peak.get('swa_used', 0):.0f} tokens), state slots "
      f"{peak.get('state', 0):.2f}; windowed slots held by cached prompts at most {peak.get('swa_cached', 0):.0f}, free at "
      f"least {low.get('swa_free', 0):.0f}; evicted {evicted:.0f} tokens; {time.time() - start:.0f} s")
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", f"swafill-{args.label}.json"), "w") as f:
    json.dump({"args": vars(args), "peak": peak, "low": low, "first": first, "rows": rows}, f, indent=1)
sys.exit(0 if all(r["ok"] for r in rows) else 1)
