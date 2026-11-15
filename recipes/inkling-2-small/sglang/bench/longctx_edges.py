"""The edges of a long-context deployment (ordinary chat requests; occupies the fleet for several minutes).

    python3 bench/longctx_edges.py <label> --context 524288 [--port 8210] [--only over,pair,abort,mixed]
  over   a prompt of context + 6K tokens: refused with HTTP 400, and a short request answers right afterwards
  pair   two cold prompts of 0.61 x context each at the same time (together more than a 600K pool holds): both
         answer with exact codes, whatever order the server runs them in
  abort  a cold prompt of 0.4 x context whose client disconnects after 20 s: a short request sent right then must
         not wait for the rest of that prefill
  mixed  a cold prompt of 0.75 x context while four streams ask short lookups in a loop: every code exact, and how
         long the short requests took meanwhile
Writes logs/longctx-edges-<label>.json. Exit 1 if any step fails. MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import http.client
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--context", type=int, required=True)
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--only", default="over,pair,abort,mixed")
args = ap.parse_args()
cl.PORT = args.port
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
out, bad = {}, 0


def lookup(tokens, seed, depths=(0.05, 0.5, 0.95)):
    rec = cl.Records(cl.lines_for(tokens), seed)
    q, want = rec.ask(depths=depths)
    return [{"role": "user", "content": rec.text() + "\n\n" + q}], want


def short():
    m, want = lookup(600, int(time.time() * 1000) % (1 << 30))
    a = cl.chat(m, 600, timeout=1800)
    return {"secs": round(a["secs"], 1), "right": cl.codes_in(a["text"]) == want, "error": a["error"]}


def report(name, ok, **facts):
    global bad
    bad += not ok
    out[name] = dict(facts, ok=bool(ok))
    print(f"{name:6s} {'ok' if ok else 'FAIL'} {json.dumps(facts)[:900]}", flush=True)
    json.dump(out, open(os.path.join(LOGS, f"longctx-edges-{args.label}.json"), "w"), indent=1)


steps = args.only.split(",")
if "over" in steps:
    m, _ = lookup(args.context + 6000, 11)
    a = cl.chat(m, 100, stream=False, timeout=1800)
    after = short()
    report("over", (a["error"] or "").startswith("HTTP 400") and after["right"], refused=(a["error"] or "not refused")[:200],
           refused_after_s=round(a["secs"], 1), short_after=after)

if "pair" in steps:
    res = [None, None]

    def one(i):
        m, want = lookup(int(args.context * 0.61), 21 + i)
        a = cl.chat(m, 4000, timeout=3600)
        res[i] = {"prompt_tokens": a["usage"].get("prompt_tokens"), "ttft_s": a["ttft"] and round(a["ttft"], 1), "secs": round(a["secs"], 1),
                  "right": cl.codes_in(a["text"]) == want, "error": a["error"], "finish": a["finish"],
                  "out_tokens": a["usage"].get("completion_tokens"), "tail": a["text"][-80:] or a["reasoning"][-200:]}
    th = [threading.Thread(target=one, args=(i,)) for i in range(2)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    report("pair", all(r["right"] and not r["error"] for r in res), first=res[0], second=res[1], short_after=short())

if "abort" in steps:
    m, _ = lookup(int(args.context * 0.4), 31)
    body = json.dumps({"model": cl.MODEL, "messages": m, "max_tokens": 100, "temperature": 0, "stream": True})
    c = http.client.HTTPConnection("localhost", args.port, timeout=600)
    c.request("POST", "/v1/chat/completions", body, {"content-type": "application/json"})
    time.sleep(20)
    c.sock.close()  # the client is gone 20 s into a prefill that takes over a minute
    first = short()
    time.sleep(5)
    second = short()
    report("abort", first["right"] and second["right"] and first["secs"] < 40, short_right_after=first, short_5s_later=second)

if "mixed" in steps:
    stop, small, big = threading.Event(), [], {}

    def loop(i):
        while not stop.is_set():
            small.append(short())

    def long_one():
        m, want = lookup(int(args.context * 0.75), 41)
        a = cl.chat(m, 4000, timeout=3600)
        big.update({"prompt_tokens": a["usage"].get("prompt_tokens"), "ttft_s": a["ttft"] and round(a["ttft"], 1),
                    "right": cl.codes_in(a["text"]) == want, "error": a["error"], "finish": a["finish"],
                    "out_tokens": a["usage"].get("completion_tokens"), "tail": a["text"][-80:] or a["reasoning"][-200:]})
    th = [threading.Thread(target=loop, args=(i,)) for i in range(4)]
    for t in th:
        t.start()
    time.sleep(10)
    lt = threading.Thread(target=long_one)
    lt.start()
    lt.join()
    stop.set()
    for t in th:
        t.join()
    secs = sorted(s["secs"] for s in small)
    report("mixed", big.get("right") and not big.get("error") and all(s["right"] and not s["error"] for s in small), long=big,
           short_requests=len(small), short_wrong=sum(1 for s in small if not s["right"]),
           short_p50_s=secs[len(secs) // 2] if secs else None, short_max_s=secs[-1] if secs else None)

print("ALL OK" if not bad else f"{bad} STEP(S) FAILED")
sys.exit(1 if bad else 0)
