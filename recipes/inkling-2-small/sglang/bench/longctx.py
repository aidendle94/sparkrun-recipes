"""Long contexts end to end: cold prefill time, exact retrieval, decode speed, and what a short request waits meanwhile.

    python3 bench/longctx.py <label> [--tokens 130000,260000,390000,500000] [--port 8210] [--seed 7]
Ordinary chat requests (safe on a serving fleet, but a cold 500K prefill occupies it for minutes). Per size:
  cold   a fresh record context (bench/chatload.py) + "give the codes of three items" at 3%, 50% and 97% of its depth:
         time to first token = the cold prefill, the three codes must be exact. A second thread sends a one-line chat
         request every 5 s meanwhile and keeps its latency (what the watchdog's probe or another user sees).
  warm   the same context with three other items (10%, 35%, 80%): time to first token on a cached prefix, codes exact.
  decode the same context with a coding question, 500 tokens: tokens/s after the first token (reasoning included),
         text not looping (distinct 4-grams of words >= 0.5), no control tokens.
Writes logs/longctx-<label>.json. Exit 1 if a code is wrong or a request fails.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--tokens", default="130000,260000,390000,500000")
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--limit", type=int, default=4000, help="max_tokens of a lookup (the model reasons longer over longer contexts)")
args = ap.parse_args()
cl.PORT = args.port
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
rows, bad = [], 0


def probes(stop, out):
    while not stop.is_set():
        a = cl.chat([{"role": "user", "content": "Reply with the word OK."}], 3, timeout=900)
        out.append(round(a["secs"], 1) if not a["error"] else a["error"])
        stop.wait(5)


for n in [int(x) for x in args.tokens.split(",")]:
    rec = cl.Records(cl.lines_for(n), args.seed * 1000003 + n)
    context = rec.text()
    q, want = rec.ask(depths=(0.03, 0.5, 0.97))
    stop, lat = threading.Event(), []
    th = threading.Thread(target=probes, args=(stop, lat), daemon=True)
    th.start()
    cold = cl.chat([{"role": "user", "content": context + "\n\n" + q}], args.limit)
    stop.set()
    th.join()
    got = cl.codes_in(cold["text"])
    u = cold["usage"]
    row = {"target": n, "prompt_tokens": u.get("prompt_tokens"),
           "cold_ttft_s": cold["ttft"] and round(cold["ttft"], 1), "cold_codes": got == want, "cold_error": cold["error"],
           "cold_finish": cold["finish"], "cold_out_tokens": u.get("completion_tokens"), "probe_s": lat}
    if u.get("prompt_tokens") and cold["ttft"]:
        row["cold_tok_s"] = round(u["prompt_tokens"] / cold["ttft"])
    q2, want2 = rec.ask(depths=(0.10, 0.35, 0.80))
    warm = cl.chat([{"role": "user", "content": context + "\n\n" + q2}], args.limit)
    code = cl.chat([{"role": "user", "content": context + "\n\nIgnore the records. " + cl.Q[1]}], 500)
    u3 = code["usage"]
    out_tokens, decode_s, written = u3.get("completion_tokens") or 0, code["secs"] - (code["ttft"] or 0), code["reasoning"] + code["text"]
    row.update({"warm_ttft_s": warm["ttft"] and round(warm["ttft"], 1), "warm_codes": cl.codes_in(warm["text"]) == want2,
                "warm_error": warm["error"], "warm_finish": warm["finish"], "warm_out_tokens": warm["usage"].get("completion_tokens"),
                "decode_tokens": out_tokens,
                "decode_tok_s": round(out_tokens / decode_s, 1) if decode_s > 0 else None, "d4w": round(cl.d4w(written), 2),
                "decode_error": code["error"]})
    ok = (row["cold_codes"] and row["warm_codes"] and not cold["error"] and not warm["error"] and not code["error"]
          and row["d4w"] >= 0.5 and not cl.CONTROL.search(written))
    bad += not ok
    print(json.dumps(row), flush=True)
    if not ok:
        print(f"  NOT OK. cold wanted {want} got {got}: {cold['text'][-200:]!r} (finish {cold['finish']}, reasoning ends {cold['reasoning'][-300:]!r})\n  warm wanted {want2}: {warm['text'][-200:]!r} (finish {warm['finish']}, reasoning ends {warm['reasoning'][-300:]!r})\n  code: {written[:300]!r}", flush=True)
    rows.append(dict(row, cold_text=cold["text"], warm_text=warm["text"], code_text=written, want=want, want2=want2,
                     cold_reasoning=cold["reasoning"][-3000:], warm_reasoning=warm["reasoning"][-3000:]))
    json.dump(rows, open(os.path.join(LOGS, f"longctx-{args.label}.json"), "w"), indent=1)
print("ALL OK" if not bad else f"{bad} SIZE(S) NOT OK")
sys.exit(1 if bad else 0)
