"""Does a later lookup in a conversation get the earlier lookup's answer? (seen twice in bench/stress.py, 2026-11-14)

    python3 bench/stale_answer.py <label> [--tokens 60000] [--contexts 4] [--branches 6] [--parallel 1] [--port 8210] [--seed 5]
Ordinary chat requests, greedy. Per record context (bench/chatload.py), `branches` conversations that share it:
  turn 0  lookup of two items                       (answer kept in the history)
  turn 1  a coding question, 150-400 tokens         (answer kept, possibly cut at the limit, as an agent's would be)
  turn 2  lookup of two other items                 <- the turn that came back with turn 0's codes
  turn 3  lookup of two more items
`--parallel n` runs n conversations of a context at once (they share the cached records prefix); 1 = one at a time.
`--cold` puts a fresh first line on every request, so every turn is prefilled from scratch (no prefix-cache hit).
`--variant` changes how the conversation is sent, to see what the failure depends on (the picks stay the same):
  reasoning  the assistant turns of the history carry their reasoning (reasoning_content), not only the answer text
  stub       the assistant turns of the history carry one fixed sentence as their reasoning (is it the block or its text?)
  system     the records are a system message, every question a user message of its own
  nocode     no coding turn: the lookups follow each other directly
  fullcode   the coding answer gets 1500 tokens, so it is kept whole and not cut
  effort=<x> reasoning_effort sent with every request (the template's default is 0.9)
Each lookup is right, stale (the codes of an earlier turn of its conversation), wrong (says which record lines the codes
were read from), or empty (ran out of tokens while reasoning). Writes logs/stale-<label>.json with the reasoning of
every lookup that is not right. Exit 1 if a lookup is stale or wrong.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import random
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--tokens", type=int, default=60000)
ap.add_argument("--contexts", type=int, default=4)
ap.add_argument("--branches", type=int, default=6)
ap.add_argument("--parallel", type=int, default=1)
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--cold", action="store_true", help="a fresh first line on every request: no request finds a cached prefix")
ap.add_argument("--variant", default="", help="comma list of reasoning, stub, system, nocode, fullcode, effort=<x>")
args = ap.parse_args()
cl.PORT = args.port
VARIANT = set(filter(None, args.variant.split(",")))
EFFORT = next((v.split("=", 1)[1] for v in VARIANT if v.startswith("effort=")), None)
assert all(v in ("reasoning", "stub", "system", "nocode", "fullcode") or v.startswith("effort=") for v in VARIANT), VARIANT
STUB = "The user's message above has been answered."
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
lock, rows = threading.Lock(), []


def conversation(c, b, rec, context):
    rng = random.Random(args.seed * 104729 + c * 1009 + b)
    n, earlier = len(rec.rows), []
    history = [{"role": "system", "content": context.strip()}] if "system" in VARIANT else []
    for turn in range(4):
        if turn == 1:
            q, want, limit = "Leave the records aside for this one. " + rng.choice(cl.Q), None, rng.randint(150, 400)
            if "fullcode" in VARIANT:
                limit = 1500
            if "nocode" in VARIANT:
                continue
        else:
            picked = [rec.rows[j] for j in sorted(rng.sample(range(n), 2))]
            q = ("From the records, give the code of " + ", ".join(f"item {i}" for i, _ in picked)
                 + ". Reply with only the codes, in this order, separated by spaces.")
            want, limit = [str(code) for _, code in picked], 1500
        messages = history + [{"role": "user", "content": (context if not history else "") + q}]
        sent = messages
        if args.cold:  # same conversation, but nothing of it is in the prefix cache
            sent = [dict(messages[0], content=f"session {random.getrandbits(48)}\n" + messages[0]["content"])] + messages[1:]
        a = cl.chat(sent, limit, 0, stream=False, timeout=1800, extra={"reasoning_effort": EFFORT} if EFFORT else None)
        if want is not None:
            got = cl.codes_in(a["text"])
            verdict = ("error" if a["error"] else "right" if got == want else "empty" if not a["text"].strip()
                       else "stale" if got in earlier else "wrong")
            row = {"context": c, "branch": b, "turn": turn, "verdict": verdict, "secs": round(a["secs"], 1),
                   "out_tokens": a["usage"].get("completion_tokens"), "finish": a["finish"]}
            if verdict != "right":
                row.update({"want": want, "got": got, "read_from": rec.where(want, got), "error": a["error"],
                            "text": a["text"][-300:], "reasoning": a["reasoning"][-2500:]})
            with lock:
                rows.append(row)
                print(f"context {c} branch {b} turn {turn} {verdict:5s} {row['secs']:6.1f} s {row['out_tokens']} tokens"
                      + (f" wanted {want} got {got} read from {row['read_from']} | reasoning ends {a['reasoning'][-160:]!r}"
                         if verdict != "right" else ""), flush=True)
            earlier.append(want)
        if a["text"].strip() and not a["error"]:
            reply = {"role": "assistant", "content": a["text"]}
            if "reasoning" in VARIANT and a["reasoning"]:
                reply["reasoning_content"] = a["reasoning"]
            if "stub" in VARIANT:
                reply["reasoning_content"] = STUB
            history = messages + [reply]


for c in range(args.contexts):
    rec = cl.Records(cl.lines_for(args.tokens), args.seed * 1000003 + c)
    context = rec.text() + "\n\n"
    todo = list(range(args.branches))
    while todo:
        now, todo = todo[:args.parallel], todo[args.parallel:]
        th = [threading.Thread(target=conversation, args=(c, b, rec, context)) for b in now]
        for t in th:
            t.start()
        for t in th:
            t.join()
        json.dump({"args": vars(args), "rows": rows}, open(os.path.join(LOGS, f"stale-{args.label}.json"), "w"), indent=1)

count = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("right", "stale", "wrong", "empty", "error")}
later = [r for r in rows if r["turn"] >= 2]
print(f"{len(rows)} lookups at {args.tokens} tokens, {args.parallel} at a time: {count} | after a history: "
      f"{sum(1 for r in later if r['verdict'] == 'right')} right of {len(later)}, {sum(1 for r in later if r['verdict'] == 'stale')} stale")
bad = count["stale"] + count["wrong"] + count["error"]
print("ALL OK" if not bad else f"{bad} LOOKUP(S) NOT RIGHT")
sys.exit(1 if bad else 0)
