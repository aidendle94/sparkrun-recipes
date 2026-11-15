"""How often does a lookup over a long context end in a recital? (reasoning that copies the context until the limit)

    python3 bench/recital_rate.py <label> [--context records|code] [--tokens 130000] [--n 40] [--limit 1500]
                                  [--temp 0] [--effort x] [--hint "..."] [--qseed 1] [--only 3,7] [--seed 7] [--port 8210]
                                  [--hosted any|<provider>]
Ordinary chat requests, one at a time, all over ONE context: the first request prefills it, the rest reuse it.
  records  lines `item <6 digits> | code <6 digits>` (bench/chatload.py); each lookup asks the codes of 3 random items
  code     Python standard-library files of this machine up to `tokens`; each lookup asks the file and the parameters
           of 3 functions that are defined exactly once in them
`--qseed` picks the questions (the same label-independent list for every run of a context, so runs can be compared
question by question), `--only` runs some of them, `--hint` is appended to every question.
`--hosted` asks the hosted model on OpenRouter instead of the fleet (key in the environment variable
OPENROUTER_API_KEY; `any` lets OpenRouter route, a provider slug such as `modal` pins one without fallbacks): the
same context and questions, so the fleet and a hosted endpoint can be compared question by question. The context is still sized with
the fleet's tokenizer. It costs the prompt once per lookup unless the provider caches it.
Verdicts: right | recital (ran out of tokens, and the reasoning holds a verbatim copy of at least 8 consecutive record
lines / 15 consecutive code lines) | empty (ran out of tokens otherwise) | wrong | error.
Writes logs/recital-<label>.json with the whole reasoning of every lookup. Exit 1 if a lookup is wrong or an error.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--context", choices=("records", "code"), default="records")
ap.add_argument("--tokens", type=int, default=130000)
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--limit", type=int, default=1500)
ap.add_argument("--temp", type=float, default=0)
ap.add_argument("--effort", default=None)
ap.add_argument("--hint", default="")
ap.add_argument("--qseed", type=int, default=1)
ap.add_argument("--only", default="", help="comma list of lookup numbers to run (the questions stay the same)")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--hosted", default="", help="any | an OpenRouter provider slug")
args = ap.parse_args()
cl.PORT = args.port
if args.hosted:
    route = {} if args.hosted == "any" else {"provider": {"only": [args.hosted], "allow_fallbacks": False}}
    cl.HOSTED = ("openrouter.ai", "/api/v1/chat/completions", os.environ["OPENROUTER_API_KEY"],
                 "thinkingmachines/inkling-2-small", route)
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
ONLY = {int(x) for x in args.only.split(",") if x}
rng = random.Random(args.qseed * 7919 + args.seed)


def parameters(signature):
    """Parameter names of a one-line signature, without self / cls and the bare markers."""
    out, depth, cur = [], 0, ""
    for ch in signature + ",":
        depth += (ch in "([{") - (ch in ")]}")
        if ch == "," and depth == 0:
            p = re.split(r"[:=]", cur.strip())[0].strip().lstrip("*")
            if p and p not in ("self", "cls", "/"):
                out.append(p)
            cur = ""
        else:
            cur += ch
    return out


if args.context == "records":
    rec = cl.Records(cl.lines_for(args.tokens), args.seed * 1000003 + args.tokens)
    context, MIN_COPY = rec.text(), 8

    def question():
        picked = [rec.rows[i] for i in sorted(rng.sample(range(len(rec.rows)), 3))]
        q = ("From the records, give the code of " + ", ".join(f"item {i}" for i, _ in picked)
             + ". Reply with only the codes, in this order, separated by spaces.")
        want = [str(c) for _, c in picked]
        return q, want, lambda text: cl.codes_in(text) == want
else:
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))
    random.Random(args.seed).shuffle(files)
    parts, total = [], 0
    for f in files:
        src = open(f, errors="replace").read()
        if not 2000 < len(src) < 120000:
            continue
        part = f"# ==== file: {os.path.basename(f)} ====\n{src.rstrip()}\n"
        n = cl.count_tokens(part)
        if total + n <= args.tokens:
            parts.append((os.path.basename(f), part))
            total += n
        if total > args.tokens * 0.97:
            break
    context, MIN_COPY = "\n".join(p for _, p in parts), 15
    defined = {}  # every def of a name counts, also the ones with a signature over several lines
    for m in re.finditer(r"\bdef (\w+)\s*\(", context):
        defined[m.group(1)] = defined.get(m.group(1), 0) + 1
    pool = []
    for name, part in parts:
        for m in re.finditer(r"^\s*def (\w+)\((.*)\):\s*$", part, re.M):
            ps = parameters(m.group(2))
            if defined[m.group(1)] == 1 and len(m.group(1)) > 4 and not m.group(1).startswith("__") and ps \
                    and all(re.fullmatch(r"\w+", p) for p in ps):  # not the `def f(...)` of a docstring
                pool.append((m.group(1), name, ps))
    print(f"code context: {len(parts)} files, {total} tokens, {len(pool)} functions to ask for", flush=True)

    def question():
        picked = rng.sample(pool, 3)
        q = ("For each of these functions defined in the code above, give the file it is defined in and its parameters "
             "(without self): " + ", ".join(f"`{n}`" for n, _, _ in picked)
             + ". Reply with one line per function, in this order: name | file | parameters.")

        def right(text):
            rows = [l for l in text.split("\n") if l.strip()]
            line = lambda name: next((l for l in rows if name in l), "")
            return all(f in line(n) and all(re.search(r"\b" + re.escape(p) + r"\b", line(n)) for p in ps) for n, f, ps in picked)
        return q, [list(p) for p in picked], right

BODY = [l.strip() for l in context.split("\n") if l.strip()]
AT = {}
for i, l in enumerate(BODY):
    AT.setdefault(l, []).append(i)


def longest_copy(reasoning):
    """(lines, first context line) of the longest run of consecutive context lines copied in order; blank lines and
    quote marks around a line do not count."""
    best, runs = (0, None), {}
    for l in (x.strip() for x in reasoning.split("\n")):
        if not l:
            continue
        runs = {i: runs.get(i - 1, 0) + 1 for i in AT.get(l, AT.get(l.strip('"'), ()))}
        for i, n in runs.items():
            if n > best[0]:
                best = (n, i - n + 1)
    return best


rows, t_first = [], None
for j in range(args.n):
    q, want, right = question()  # drawn for every j, so `--only` keeps the questions of the full run
    if ONLY and j not in ONLY:
        continue
    a = cl.chat([{"role": "user", "content": context + "\n\n" + q + args.hint}], args.limit, args.temp,
                extra={"reasoning_effort": args.effort} if args.effort else None)
    copy = longest_copy(a["reasoning"])
    ran_out = a["finish"] == "length"
    verdict = ("error" if a["error"] else "right" if right(a["text"]) else "recital" if ran_out and copy[0] >= MIN_COPY
               else "empty" if ran_out or not a["text"].strip() else "wrong")
    rows.append({"j": j, "q": q, "want": want, "verdict": verdict, "finish": a["finish"], "secs": round(a["secs"], 1),
                 "out_tokens": a["usage"].get("completion_tokens"), "prompt_tokens": a["usage"].get("prompt_tokens"),
                 "longest_copy": copy, "error": a["error"], "provider": a.get("provider"), "text": a["text"],
                 "reasoning": a["reasoning"]})
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {j:3d} {verdict:7s} finish {a['finish']} {rows[-1]['out_tokens']} tokens "
          f"{rows[-1]['secs']:.1f} s, longest copy {copy[0]} lines" + (f" from context line {copy[1]}" if copy[0] >= MIN_COPY else "")
          + (f" | answer {a['text'][:160]!r} wanted {want}" if verdict == "wrong" else "")
          + (f" | {a['error']}" if a["error"] else "") + (f" | {a['provider']}" if a.get("provider") else ""), flush=True)
    json.dump({"args": vars(args), "rows": rows}, open(os.path.join(LOGS, f"recital-{args.label}.json"), "w"))

count = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("right", "recital", "empty", "wrong", "error")}
used = [r["out_tokens"] or 0 for r in rows if r["verdict"] == "right"]
print(f"{len(rows)} {args.context} lookups at {rows[0]['prompt_tokens'] if rows else 0} prompt tokens, temperature {args.temp}, "
      f"effort {args.effort or 'default'}, limit {args.limit}: {count} | ran out of tokens: {count['recital'] + count['empty']} "
      f"| right ones used {sum(used) / max(1, len(used)):.0f} tokens on average, at most {max(used, default=0)}")
bad = count["wrong"] + count["error"]
print("NO WRONG ANSWER" if not bad else f"{bad} LOOKUP(S) WRONG OR FAILED")
sys.exit(1 if bad else 0)
