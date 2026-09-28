#!/usr/bin/env python3
"""multi_check.py — batched-decode correctness smoke test: the same short greedy chats alone and 4 at a time.
usage: multi_check.py BASE   Prints each answer alone and in the batch, the longest repeated-token run, and a verdict
(batched answers that collapse to 1-2 tokens or repeat one token 8+ times in a row count as broken).

MIT License, Copyright (c) 2026 Aiden Le.
"""
import json, sys, threading, urllib.request
base = sys.argv[1].rstrip("/")
P = ["Name the three primary colors and say one sentence about each.", "What is the capital of France? Answer in one short paragraph.",
     "Write a haiku about autumn rain.", "Explain in two sentences what a hash table is."]
def ask(p, out, k):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 80, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out[k] = json.load(urllib.request.urlopen(r, timeout=300))["choices"][0]["message"]["content"] or ""
def run(par):
    out = {}
    th = [threading.Thread(target=ask, args=(p, out, i)) for i, p in enumerate(P)]
    if par:
        [t.start() for t in th]; [t.join() for t in th]
    else:
        for t in th: t.start(); t.join()
    return [out[i] for i in range(len(P))]
def rep(s):
    w = s.split(); best = cur = 0
    for i in range(len(w)):
        cur = cur + 1 if i and w[i] == w[i - 1] else 1; best = max(best, cur)
    return best
alone = run(False); bad = 0
for rnd in range(2):
    batch = run(True)
    for i, (a, b) in enumerate(zip(alone, batch)):
        broken = len(b.split()) < 3 or rep(b) >= 8
        bad += broken
        print(f"round {rnd} q{i} {'BROKEN' if broken else 'ok':6s} alone {a[:70]!r}\n                 batch {b[:70]!r}", flush=True)
print(f"multi_check verdict: {'BROKEN' if bad else 'OK'} ({bad} broken of {2 * len(P)} batched answers)")
