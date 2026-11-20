"""Load for a confidence-head capture window (patches/conf_capture.py): a few thousand chat requests shaped like the
fleet's traffic, sent in short phases that follow the head's temperature.

    python3 bench/conf_capture_run.py --out <dir> [--scale 1.0] [--minutes 105] [--concurrency 4] [--plan-only]
Requests (ids "gate-k8-cc-<category>-<n>", so the recorder takes them and the gate verifies 8 rows):
  short    general questions, short coding tasks, agent turns with tool definitions            (about 60%)
  medium   a source file of 1-8K tokens with a task: explain, edit, rename, copy lines, tests  (about 20%)
  long     shared contexts of 15-100K tokens asked many questions each, one stream at a time,
           so that all but the first question meet a cached prefix: record lookups, a
           many-file code context (edit, copy, retrieve), agent transcripts with tool results  (about 20%)
Sampling as clients send it: 60% no temperature (the chat and messages endpoints then sample at 1.0, what Claude
Code gets), 30% temperature 0, 10% temperature 0.7. Reasoning effort: default for 55%, "none" for 30%, "low" for 15%.
Heat: requests start only while the head's hottest zone (and the recording node's, --watch) is under --soft and a
phase is younger than --phase seconds; then the running ones finish and nothing starts until the head is back
under --cool. At --abort every open request is cut and queued again. The first question of a long context waits
for a cool head and runs alone. The fleet-level guard (stop at 97 C) belongs to the window script, not to this file.
Writes <out>/manifest.jsonl (a line per finished request), <out>/temps.tsv (a line a second), <out>/plan.json.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import glob
import hashlib
import http.client
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402
from toolcall_cases import SYSTEM, TOOLS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--seed", type=int, default=1119)
ap.add_argument("--scale", type=float, default=1.0, help="fraction of the full plan (about 2,400 requests)")
ap.add_argument("--minutes", type=float, default=120, help="wall-clock limit")
ap.add_argument("--concurrency", type=int, default=4)
ap.add_argument("--soft", type=int, default=92)
ap.add_argument("--abort", type=int, default=95)
ap.add_argument("--cool", type=int, default=76)
ap.add_argument("--phase", type=float, default=60)
ap.add_argument("--watch", default="", help="user@host of the recording node: its hottest zone counts too")
ap.add_argument("--corpus", default=os.path.expanduser("~/inkling2-corpus"), help="a directory of source files (.py .cuh .cu .h, 3-120 KB each) for the file tasks; the author's captures read the serving image's own sources")
ap.add_argument("--plan-only", action="store_true")
ap.add_argument("--no-long", action="store_true", help="short and medium requests only (a smoke run)")
ap.add_argument("--tag", default="m", help="part of every request id: keeps two runs of one capture apart")
args = ap.parse_args()
PREFIX = f"gate-k8-cc-{args.tag}-"
MODEL = "inkling-2-small"
rng = random.Random(args.seed)
os.makedirs(args.out, exist_ok=True)

TOPICS = """photosynthesis; the French Revolution; how vaccines train the immune system; compound interest; the water cycle;
plate tectonics; how a transistor works; the causes of the First World War; supply and demand; the Doppler effect;
how airplanes generate lift; the rules of chess openings; black holes; the printing press; inflation; DNA replication;
the Roman aqueducts; how GPS finds a position; sourdough fermentation; the Silk Road; tides; public-key cryptography;
the scientific method; how a refrigerator works; the Industrial Revolution; rainbow formation; garbage collection in
programming languages; the placebo effect; how noise-cancelling headphones work; the Pythagorean theorem; volcanoes;
the stock market; the fall of the Berlin Wall; how batteries store energy; the human circadian rhythm; TCP congestion
control; composting; the Renaissance; how a bill becomes law; antibiotics and resistance; the greenhouse effect;
binary search trees; the history of the bicycle; how touchscreens sense a finger; ocean currents; game theory;
the Apollo program; how espresso extraction works; earthquakes; database indexes; the life cycle of a star;
perspective in painting; how vaccines are tested; the metric system; machine translation; honeybee communication;
the Byzantine Empire; how elevators stay safe; prime numbers; coral reefs; version control with git; jazz harmony;
the electoral college; how a microwave oven heats food; the Great Depression; Bayesian reasoning; glaciers;
the invention of the telephone; how HTTPS protects a connection; crop rotation; the Maya calendar; fiber optics;
the periodic table; how a thermostat controls temperature; migration of birds; the Magna Carta; hash tables;
sleep and memory; the Panama Canal; how solar panels make electricity; monetary policy; the Marshall Plan""".replace("\n", " ").split("; ")
ASKS = ["Explain {t} in two short paragraphs for a curious teenager.", "Give five bullet points on {t}.",
        "What are the three most common misconceptions about {t}? Correct each in a sentence.",
        "Write a short quiz of four questions about {t}, with the answers.", "Summarise {t} in exactly three sentences.",
        "Explain {t} with an everyday analogy.", "What should a beginner read or try first to understand {t}? Be specific.",
        "Write a 100-word encyclopedia entry on {t}.", "Compare {t} with {u}: what do they have in common, if anything?",
        "Draft a polite email to a colleague asking for a one-page briefing on {t} by Friday.",
        "Translate into French, then into German: 'I would like to learn more about {t}.'",
        "Return a JSON object with the keys title, summary and three_keywords for a note about {t}. JSON only.",
        "A friend claims that {t} is overrated. Reply in four sentences, friendly but firm."]
LANGS = ["Python", "Rust", "Go", "TypeScript", "C", "C++", "Java", "Bash", "SQL", "Kotlin"]
TASKS = """checks whether a string is a palindrome, ignoring case and punctuation; merges overlapping intervals;
parses a semantic version string and compares two versions; implements an LRU cache with get and put in O(1);
reverses a singly linked list in place; finds the k largest elements of a stream with a heap; debounces a function call;
computes the Levenshtein distance between two strings; flattens a nested list of arbitrary depth; implements binary
search returning the insertion point; validates balanced brackets of three kinds; converts a Roman numeral to an integer;
groups anagrams from a list of words; rotates a matrix by 90 degrees in place; computes a running median; parses one CSV
line with quoted fields; implements a token-bucket rate limiter; walks a directory tree and sums file sizes by extension;
retries a failing operation with exponential backoff and jitter; deduplicates a list while keeping order; implements a
trie with insert and prefix search; finds the longest common prefix of a list of strings; topologically sorts a
dependency graph and reports a cycle; computes the nth Fibonacci number with memoisation and iteratively; formats a
duration in seconds as a human-readable string; implements union-find with path compression; reads a large file line by
line and counts word frequencies; converts snake_case identifiers to camelCase; implements a ring buffer; finds the
first non-repeated character of a string; checks whether two rectangles overlap; implements quickselect; chunks a list
into batches of n; parses an ISO 8601 date without a library; implements a simple publish-subscribe event bus;
computes the intersection of two sorted arrays; serialises a binary tree to a string and back""".replace("\n", " ").split("; ")
CODE_ASKS = ["Write a {l} function that {k}.", "Write a {l} function that {k}. Add a short docstring and two example calls.",
             "In {l}, write code that {k}, then state its time and space complexity in one line each.",
             "Write a {l} function that {k}, with three unit tests.",
             "Show an idiomatic {l} solution that {k}. Code only, no explanation."]
AGENT_ASKS = ["Read {p} and tell me what its main entry point does.", "Run the test suite and summarise any failures.",
              "Find every place where `{i}` is used in the repository.", "Create the file notes/{w}.md with three todo items about {t}.",
              "Show me the git status, then list the Python files under src/.", "Search for TODO comments under {d} and count them.",
              "Open {p}, lines 1 to 60, and check whether it imports anything unused.",
              "Add an issue to acme/widgets titled 'Flaky test in {w}' with a two-line description.",
              "Rename the function `{i}` to `{i}_v2` in {p}.", "Write a todo list for adding retries to the client in {p}.",
              "What does the Makefile build by default? Check it first.", "Run `python -m pytest -q {d}` with a 120 second timeout.",
              "Replace every occurrence of 'colour' with 'color' in {p}.", "List the files that changed in the last commit."]
PATHS = ["src/app.py", "src/utils/config.py", "server/handlers.go", "lib/parser.rs", "web/src/api/client.ts", "tests/test_cache.py",
         "cmd/tool/main.go", "src/models/user.py", "scripts/deploy.sh", "core/scheduler.cpp", "src/db/migrations.py", "README.md"]
IDENTS = ["parse_config", "load_user", "handle_request", "retry_with_backoff", "build_index", "validate_token", "flush_cache"]
WORDS = ["parser", "cache", "scheduler", "uploader", "auth", "billing", "search", "export"]
FILE_ASKS = [
    ("explain", "Explain briefly what this code does and name its two most important functions or classes.", 128),
    ("docstrings", "Add a one-line comment above every function or method that lacks one and output the complete updated code.", 256),
    ("rename", "Rename the identifier `{i}` to `{i}_renamed` everywhere and output the complete updated code.", 256),
    ("tests", "Write unit tests for the first function or method defined here.", 192),
    ("copy", "Copy lines {a} through {b} of the file verbatim, with their line numbers removed, and nothing else.", 256),
    ("outline", "List every function and class defined here, each with a one-line description.", 160),
    ("bug", "Name the most likely bug or unhandled edge case in this code and show the fix as a small diff.", 192),
    ("first", "Quote the first function or class definition line of this file exactly, then say in one sentence what it is for.", 96)]


def temperature():
    r = rng.random()
    return None if r < 0.60 else 0 if r < 0.90 else 0.7


def effort():
    r = rng.random()
    return None if r < 0.55 else "none" if r < 0.85 else "low"


def budget(base=None):
    if base:
        return base
    r = rng.random()
    return 96 if r < 0.60 else 160 if r < 0.85 else 256 if r < 0.97 else 512


def corpus():
    files = []
    for ext in ("py", "cuh", "cu", "h"):
        files += glob.glob(os.path.join(args.corpus, "**", f"*.{ext}"), recursive=True)
    files = sorted(f for f in files if 3000 <= os.path.getsize(f) <= 120000)
    rng.shuffle(files)
    return files


def read(path, limit=None):
    try:
        text = open(path, encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        return None
    if "\x00" in text or max((len(line) for line in text.splitlines()), default=0) > 400:
        return None
    return text if limit is None else text[:limit]


def rel(path):
    return os.path.relpath(path, args.corpus)


def numbered(text):
    return "\n".join(f"{n:4d}  {line}" for n, line in enumerate(text.splitlines(), 1))


def ident(text):
    names = [n for n in re.findall(r"\b(?:def|class|struct|void|int|auto)\s+([A-Za-z_][A-Za-z0-9_]{3,})", text)
             if not n.startswith("__")]
    return rng.choice(names) if names else "result"


def file_question(text, path, where=""):
    kind, ask, tokens = rng.choice(FILE_ASKS)
    lines = text.count("\n") + 1
    a = rng.randrange(1, max(2, lines - 12))
    ask = ask.format(i=ident(text), a=a, b=min(lines, a + rng.randrange(4, 12)))
    return kind, (f"In the file `{rel(path)}` {where}: " + ask[0].lower() + ask[1:]) if where else ask, tokens


def tool_defs():
    return [{"type": "function", "function": t} for t in TOOLS]


def read_turns(pairs):
    """An agent that read these (path, text) files: one assistant turn of read_file calls, then the tool results."""
    calls = [{"id": f"call_{n}", "type": "function",
              "function": {"name": "read_file", "arguments": json.dumps({"path": p})}} for n, (p, _) in enumerate(pairs)]
    return [{"role": "assistant", "content": None, "tool_calls": calls}] + \
           [{"role": "tool", "tool_call_id": f"call_{n}", "content": numbered(t)} for n, (_, t) in enumerate(pairs)]


def build():
    jobs, groups = [], []
    n = lambda count: max(1, round(count * args.scale))
    files = corpus()
    used = iter(files)

    def job(cat, messages, tokens=None, tools=None, group=None, temp="draw", eff="draw"):
        jobs.append({"rid": f"{PREFIX}{cat}-{len(jobs):05d}", "cat": cat, "messages": messages, "tools": tools,
                     "max_tokens": budget(tokens), "temperature": temperature() if temp == "draw" else temp,
                     "effort": effort() if eff == "draw" else eff, "group": group})
        return jobs[-1]

    for _ in range(n(650)):
        t, u = rng.sample(TOPICS, 2)
        job("general", [{"role": "user", "content": rng.choice(ASKS).format(t=t, u=u)}])
    for _ in range(n(500)):
        job("coding", [{"role": "user", "content": rng.choice(CODE_ASKS).format(l=rng.choice(LANGS), k=rng.choice(TASKS))}])
    for _ in range(n(330)):
        ask = rng.choice(AGENT_ASKS).format(p=rng.choice(PATHS), i=rng.choice(IDENTS), w=rng.choice(WORDS),
                                            t=rng.choice(TOPICS), d=rng.choice(["src/", "tests/", "lib/", "server/"]))
        job("agent", [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}], tools=tool_defs())
    made = 0
    while made < n(500):
        path = next(used)
        text = read(path, rng.choice([3000, 6000, 10000, 16000, 22000]))
        if not text:
            continue
        made += 1
        if rng.random() < 0.25:  # the same file as an agent sees it: a read_file result, then an edit request
            ask = rng.choice(["Rename `{i}` to `{i}_renamed` where it is defined, with edit_file.",
                              "Add a short comment above the definition of `{i}`, with edit_file.",
                              "The first function needs an early return for an empty input; make that edit."]).format(i=ident(text))
            job("file-agent", [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Look at {rel(path)}."},
                               *read_turns([(rel(path), text)]), {"role": "user", "content": ask}], 256, tools=tool_defs())
        else:
            kind, ask, tokens = file_question(text, path)
            job(f"file-{kind}", [{"role": "user", "content": f"File `{rel(path)}`:\n\n```\n{numbered(text)}\n```\n\n{ask}"}], tokens)

    def group(name, kind, head, asks):
        """A shared context: `head` messages, then one question per request, sent one after another."""
        gid = len(groups)
        members = [job(f"long-{kind}", head + [{"role": "user", "content": a}], tok, tools=tools, group=gid)
                   for a, tok, tools in asks]
        groups.append({"id": gid, "name": name, "kind": kind, "jobs": [m["rid"] for m in members],
                       "chars": sum(len(m.get("content") or "") for m in head)})

    per = n(24)
    for size in (20000, 40000, 60000):  # record lookups: exact answers, copied digits
        rec = cl.Records(max(8, int(size / 10.2)), rng.randrange(1 << 30))
        asks = []
        for _ in range(per):
            q, _codes = rec.ask(k=rng.choice([1, 2, 3, 5]))
            asks.append((q, 64, None))
        group(f"records-{size // 1000}k", "records", [{"role": "user", "content": "Records:\n" + rec.text()},
                                                      {"role": "assistant", "content": "I have the records."}], asks)
    for size in (15000, 15000, 15000, 30000, 30000, 60000, 60000, 100000):  # many files, as pasted into a chat
        parts, total = [], 0
        while total < size * 2.4:  # 2.77 characters a token once numbered (measured), numbering adds about 15%
            path = next(used)
            text = read(path, 24000)
            if text:
                parts.append((path, text))
                total += len(text)
        context = "\n\n".join(f"=== {rel(p)} ===\n{numbered(t)}" for p, t in parts)
        asks = []
        for _ in range(per):
            path, text = rng.choice(parts)
            _, ask, tokens = file_question(text, path, "above")
            asks.append((ask, tokens, None))
        group(f"files-{size // 1000}k", "files", [{"role": "user", "content": "These are files of a repository:\n\n" + context},
                                                  {"role": "assistant", "content": "I have read the files."}], asks)
    for size in (12000, 20000, 30000):  # an agent's transcript: tool definitions, read_file results, then a request
        parts, total = [], 0
        while total < size * 2.4:
            path = next(used)
            text = read(path, 16000)
            if text:
                parts.append((rel(path), text))
                total += len(text)
        head = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Read " + ", ".join(p for p, _ in parts) + " before we change anything."},
                *read_turns(parts), {"role": "assistant", "content": "I have read the files."}]
        asks = []
        for _ in range(per):
            path, text = rng.choice(parts)
            ask = rng.choice(["In {p}, rename `{i}` to `{i}_renamed` where it is defined, with edit_file.",
                              "In {p}, add a short comment above the definition of `{i}`, with edit_file.",
                              "Which of the files defines `{i}`? Answer with the path only.",
                              "Write the first 12 lines of {p} into notes/excerpt.txt with write_file, unchanged.",
                              "Explain in three sentences what `{i}` in {p} is for."]).format(p=path, i=ident(text))
            asks.append((ask, 256, tool_defs()))
        group(f"agent-{size // 1000}k", "agent", head, asks)
    rng.shuffle(groups)  # kinds and sizes mixed in time; the largest context last (its cold prefill is the hottest)
    groups.sort(key=lambda g: g["name"] == "files-100k")
    return jobs, groups


jobs, groups = build()
if args.no_long:
    jobs, groups = [j for j in jobs if j["group"] is None], []
by_rid = {j["rid"]: j for j in jobs}
short = [j for j in jobs if j["group"] is None]
rng.shuffle(short)
plan = {"seed": args.seed, "scale": args.scale, "requests": len(jobs), "short": len(short),
        "categories": {c: sum(1 for j in jobs if j["cat"] == c) for c in sorted({j["cat"] for j in jobs})},
        "temperature": {str(t): sum(1 for j in jobs if j["temperature"] == t) for t in (None, 0, 0.7)},
        "effort": {str(e): sum(1 for j in jobs if j["effort"] == e) for e in (None, "none", "low")},
        "max_tokens": {str(m): sum(1 for j in jobs if j["max_tokens"] == m) for m in sorted({j["max_tokens"] for j in jobs})},
        "groups": groups,
        "jobs": [{"rid": j["rid"], "cat": j["cat"], "group": j["group"], "max_tokens": j["max_tokens"],
                  "temperature": j["temperature"], "effort": j["effort"], "tools": bool(j["tools"]),
                  "chars": sum(len(m.get("content") or "") for m in j["messages"]),
                  "sha1": hashlib.sha1(json.dumps(j["messages"], sort_keys=True).encode()).hexdigest()[:12]} for j in jobs]}
json.dump(plan, open(os.path.join(args.out, "plan.json"), "w"))
print(json.dumps({k: v for k, v in plan.items() if k not in ("jobs", "groups")}), flush=True)
print("groups:", " ".join(f"{g['name']}({len(g['jobs'])}, {g['chars'] // 1000}K chars)" for g in groups), flush=True)
if args.plan_only:
    sys.exit(0)

# ---- the run -----------------------------------------------------------------------------------------------------
lock = threading.Lock()
gate, solo_free, stop = threading.Event(), threading.Event(), threading.Event()
solo_free.set()
state = {"mode": "cool", "open": {}, "done": 0, "errors": 0, "streak": 0, "aborted": 0, "tokens": 0, "watch": 0,
         "want_solo": False, "solo_go": False, "peak": 0, "peak_watch": 0, "retries": {}}
manifest = open(os.path.join(args.out, "manifest.jsonl"), "a")
temps = open(os.path.join(args.out, "temps.tsv"), "a")


def hot():
    best = 0
    for zone in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            best = max(best, int(open(zone).read()) // 1000)
        except (OSError, ValueError):
            pass
    return best


def watcher():
    """The recording node's hottest zone, a line every 3 s over one ssh connection."""
    cmd = "while true; do m=0; for z in /sys/class/thermal/thermal_zone*/temp; do t=$(( $(cat $z) / 1000 )); " \
          "[ $t -gt $m ] && m=$t; done; echo $m; sleep 3; done"
    while not stop.is_set():
        p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.watch, cmd],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in p.stdout:
            if stop.is_set():
                break
            if line.strip().isdigit():
                state["watch"] = int(line)
        p.kill()
        time.sleep(2)


def send(job):
    body = {"model": MODEL, "messages": job["messages"], "max_tokens": job["max_tokens"], "stream": True,
            "stream_options": {"include_usage": True}, "rid": job["rid"] + f"-r{state['retries'].get(job['rid'], 0)}"}
    if job["temperature"] is not None:
        body["temperature"] = job["temperature"]
    if job["effort"] is not None:
        body["reasoning_effort"] = job["effort"]
    if job["tools"]:
        body["tools"] = job["tools"]
    a = {"rid": body["rid"], "cat": job["cat"], "group": job["group"], "temperature": job["temperature"],
         "effort": job["effort"], "max_tokens": job["max_tokens"], "start": round(time.time(), 3), "ttft": None,
         "finish": None, "usage": {}, "error": None, "aborted": False, "chars": 0, "reasoning_chars": 0, "calls": 0}
    c = http.client.HTTPConnection("127.0.0.1", args.port, timeout=1800)
    with lock:
        state["open"][job["rid"]] = c
    try:
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            a["error"] = f"HTTP {r.status}: {r.read()[:200].decode(errors='replace')}"
        else:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    a["usage"] = d["usage"]
                for ch in d.get("choices") or ():
                    delta = ch.get("delta") or {}
                    text, think = delta.get("content") or "", delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if (text or think or delta.get("tool_calls")) and a["ttft"] is None:
                        a["ttft"] = round(time.time() - a["start"], 3)
                    a["chars"] += len(text)
                    a["reasoning_chars"] += len(think)
                    a["calls"] += sum(1 for t in delta.get("tool_calls") or () if (t.get("function") or {}).get("name"))
                    a["finish"] = ch.get("finish_reason") or a["finish"]
    except Exception as e:  # a cut connection is a result of the run
        a["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        with lock:
            cut = state["open"].pop(job["rid"], None) is None
        try:
            c.close()
        except OSError:
            pass
    a["aborted"] = cut
    a["secs"] = round(time.time() - a["start"], 3)
    return a


def record(job, a, requeue):
    with lock:
        manifest.write(json.dumps(a) + "\n")
        manifest.flush()
        if a["aborted"]:
            state["aborted"] += 1
            tries = state["retries"][job["rid"]] = state["retries"].get(job["rid"], 0) + 1
            if tries <= 2:
                requeue(job)
        elif a["error"]:
            state["errors"] += 1
            state["streak"] += 1
        else:
            state["done"] += 1
            state["streak"] = 0
            state["tokens"] += (a["usage"] or {}).get("completion_tokens") or 0


def short_worker():
    while not stop.is_set():
        if not (gate.wait(1) and solo_free.wait(1)):
            continue
        with lock:
            job = short.pop() if short else None
        if job is None:
            return
        if not gate.is_set() or not solo_free.is_set():
            with lock:
                short.append(job)
            continue
        record(job, send(job), short.append)


def long_worker():
    for g in groups:
        pending = [by_rid[r] for r in g["jobs"]][::-1]
        first = True
        while pending and not stop.is_set():
            job = pending.pop()
            if first:  # the cold prefill of a context: from a cool head and alone
                state["want_solo"] = True
                while not state["solo_go"] and not stop.is_set():
                    time.sleep(0.5)
                a = send(job)
                state["want_solo"] = state["solo_go"] = False
                solo_free.set()
                first = a["aborted"] or bool(a["error"])
                if a["aborted"] and state["retries"].get(job["rid"], 0) >= 1:
                    print(f"{time.strftime('%T')} context {g['name']}: the cold prefill was cut twice by the "
                          f"temperature limit, the context is skipped", flush=True)
                    record(job, a, lambda j: None)
                    break
                record(job, a, pending.append)
                continue
            if not gate.wait(1):
                pending.append(job)
                continue
            record(job, send(job), pending.append)
    short_worker()


def cut_all():
    with lock:
        conns, state["open"] = list(state["open"].values()), {}
    for c in conns:
        try:
            c.sock.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass


threads = [threading.Thread(target=long_worker, daemon=True)] + \
          [threading.Thread(target=short_worker, daemon=True) for _ in range(args.concurrency - 1)]
if args.watch:
    threading.Thread(target=watcher, daemon=True).start()
began = phase_start = time.time()
for t in threads:
    t.start()
last_print, reason, smooth = 0.0, "all requests done", []
while any(t.is_alive() for t in threads):
    time.sleep(1)
    now, h = time.time(), hot()
    smooth = (smooth + [h])[-3:]
    level = max(sum(smooth) / len(smooth), state["watch"])
    state["peak"], state["peak_watch"] = max(state["peak"], h), max(state["peak_watch"], state["watch"])
    with lock:
        busy = len(state["open"])
    if max(h, state["watch"]) >= args.abort and busy:
        cut_all()
        gate.clear()
        state["mode"] = "cool"
        print(f"{time.strftime('%T')} {max(h, state['watch'])} C: {busy} open requests cut and queued again", flush=True)
    elif state["mode"] == "load" and (level >= args.soft or (now - phase_start >= args.phase and not state["solo_go"])
                                      or (state["want_solo"] and not state["solo_go"])):
        gate.clear()
        state["mode"] = "drain"
    elif state["mode"] == "drain" and busy == 0:
        state["mode"] = "cool"
    elif state["mode"] == "cool" and busy == 0 and level <= args.cool:
        phase_start = now
        if state["want_solo"]:
            solo_free.clear()
            state["solo_go"] = True
            state["mode"] = "load"       # the gate stays closed until the solo request is back
        else:
            state["mode"] = "load"
            gate.set()
    if state["mode"] == "load" and not state["solo_go"] and not state["want_solo"] and not gate.is_set():
        gate.set()
    temps.write(f"{now:.1f}\t{h}\t{state['watch']}\t{state['mode']}\t{busy}\t{state['done']}\t{state['tokens']}\n")
    temps.flush()
    if now - last_print >= 60:
        last_print = now
        print(f"{time.strftime('%T')} {state['mode']:5s} head {h} C (peak {state['peak']}), watch {state['watch']} C; "
              f"done {state['done']} of {len(jobs)}, {state['tokens']} tokens, {state['errors']} errors, "
              f"{state['aborted']} cut; {(now - began) / 60:.0f} min", flush=True)
    if state["streak"] >= 8:
        reason = "eight requests in a row failed (the fleet is gone?)"
    elif now - began > args.minutes * 60:
        reason = f"the {args.minutes:.0f} minute limit"
    else:
        continue
    stop.set()
    gate.clear()
    for _ in range(90):  # let the running requests finish, at most 90 s
        with lock:
            if not state["open"]:
                break
        time.sleep(1)
    cut_all()
    break
stop.set()
print(f"{time.strftime('%T')} end: {reason}; done {state['done']} of {len(jobs)}, {state['tokens']} completion tokens, "
      f"{state['errors']} errors, {state['aborted']} cut; {(time.time() - began) / 60:.1f} min; head peak "
      f"{state['peak']} C, watched node peak {state['peak_watch']} C", flush=True)
sys.exit(0 if reason == "all requests done" else 3 if "failed" in reason else 2)
