"""How many thinking tokens does each reasoning effort spend, and are the answers still right?

    python3 bench/effort_tokens.py <label> [--levels none,minimal,low,0.5,medium,high,max] [--passes 2] [--parallel 4]
                                   [--limit 8000] [--temp 0] [--only kind,..] [--port 8210] [--model inkling-2-small]
Ordinary short chat requests (safe on a serving fleet). Every task is asked at every level, `passes` times (the order
is shuffled per pass, so a task meets other batch neighbours). Task kinds:
  trivial     a one-word fact, an echo
  reasoning   small word problems with one right number or name
  code-read   predict the output of a snippet, name the line of a bug
  code-write  write a small function; the answer's code block is run against tests in a subprocess
  tool        agent turns with tools declared: the first look at the repository, the next one after a file was
              read, the edit
  hard        tasks that need real thought: an expression evaluator and an LRU cache to write (tested), a counting
              problem, an ordering puzzle, a recursive program to trace, and an agent turn that has to find a
              missing line in a cache module from a failing test
The server turns a level name into the number the prompt carries (`Thinking effort level: x`): none 0.0, minimal 0.1,
low 0.2, medium 0.7, high 0.9 (also the default when a request names none), xhigh and max 0.99; a float is taken as
it is. Thinking tokens are counted by tokenizing the returned reasoning.
Prints per level and kind the median and the range of thinking tokens and how many answers were right; writes
logs/effort-<label>.json. Exit 1 only on request errors.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chatload as cl  # noqa: E402
import toolcall_cases as tc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--levels", default="none,minimal,low,0.5,medium,high,max")
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--parallel", type=int, default=4)
ap.add_argument("--limit", type=int, default=8000)
ap.add_argument("--temp", type=float, default=0)
ap.add_argument("--only", default="", help="comma list of task kinds")
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--model", default="inkling-2-small")
args = ap.parse_args()
cl.PORT, cl.MODEL = args.port, args.model
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
LEVELS = [x for x in args.levels.split(",") if x]

SNIP_SUM = '''def f(xs):
    out = []
    for i in range(1, len(xs)):
        if xs[i] > xs[i - 1]:
            out.append(xs[i] - xs[i - 1])
    return sum(out) * len(out)

print(f([3, 1, 4, 1, 5, 9, 2, 6]))'''
SNIP_DICT = '''d = {}
for w in "the quick brown fox jumps over the lazy dog the end".split():
    d[len(w)] = d.get(len(w), 0) + 1
print(sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[0])'''
SNIP_BUG = '''1  def find(xs, x):
2      lo, hi = 0, len(xs) - 1
3      while lo <= hi:
4          mid = (lo + hi) // 2
5          if xs[mid] == x:
6              return mid
7          if xs[mid] < x:
8              lo = mid
9          else:
10             hi = mid - 1
11     return -1'''
TEST_FILE = '''from dates import parse_date


def test_parse_date():
    assert parse_date("2026-03-09") == (2026, 3, 9)
    assert parse_date(" 1999-12-31 ") == (1999, 12, 31)
'''
SRC_FILE = '''from datetime import datetime


def parse_date(text):
    """Parse an ISO date (YYYY-MM-DD) into a (year, month, day) tuple."""
    d = datetime.strptime(text.strip(), "%d/%m/%Y")
    return (d.year, d.month, d.day)
'''
ASK_FIX = "tests/test_dates.py::test_parse_date fails. Find the cause and fix it."
SNIP_WALK = '''def walk(n, seen=None):
    seen = {} if seen is None else seen
    if n in seen:
        return seen[n]
    if n < 3:
        seen[n] = n + 1
    else:
        seen[n] = walk(n - 1, seen) + 2 * walk(n - 3, seen) - len(seen) % 2
    return seen[n]


def pairs(k):
    for i in range(k):
        if walk(i) % 3 == 0:
            yield i, walk(i)


print(list(pairs(9))[-2:], walk(7))'''
CACHE_FILE = '''import time
from collections import OrderedDict


class TTLCache:
    """Least-recently-used cache with a time to live per entry.

    `get` and `put` both count as a use. When the cache is full, `put` drops the entry
    that was used longest ago. Entries older than `ttl` seconds are dropped on access.
    """

    def __init__(self, capacity, ttl, clock=time.monotonic):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._data = OrderedDict()  # key -> (value, stored_at), least recently used first
        self.hits = 0
        self.misses = 0

    def _expired(self, stored_at):
        return self.clock() - stored_at >= self.ttl

    def get(self, key, default=None):
        entry = self._data.get(key)
        if entry is None:
            self.misses += 1
            return default
        value, stored_at = entry
        if self._expired(stored_at):
            del self._data[key]
            self.misses += 1
            return default
        self.hits += 1
        return value

    def put(self, key, value):
        if key in self._data:
            del self._data[key]
        elif len(self._data) >= self.capacity:
            self._purge()
            if len(self._data) >= self.capacity:
                self._data.popitem(last=False)
        self._data[key] = (value, self.clock())

    def _purge(self):
        for key in [k for k, (_, stored_at) in self._data.items() if self._expired(stored_at)]:
            del self._data[key]

    def __len__(self):
        return len(self._data)
'''
CACHE_TEST = '''from cache import TTLCache


def test_get_counts_as_use():
    now = [0.0]
    c = TTLCache(2, ttl=60, clock=lambda: now[0])
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)
    assert c.get("b") is None
    assert c.get("a") == 1
'''
ASK_CACHE = ("tests/test_cache.py::test_get_counts_as_use fails:\n\n    >       assert c.get(\"b\") is None\n"
             "    E       assert 2 is None\n\nFind the cause and fix it.")


def number(want):
    return lambda a: re.sub(r"[,\s]", "", a["text"]).strip(".") == str(want)


def word(want):
    return lambda a: a["text"].strip().strip(".").lower() == want


def runs(tests):
    """The last code block of the answer, then `tests`, in a subprocess with limits."""
    def check(a):
        blocks = re.findall(r"```(?:python)?\n(.*?)```", a["text"], re.S)
        if not blocks:
            return False
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "answer.py")
            open(path, "w").write(blocks[-1] + "\n\n" + tests + "\n")
            limit = lambda: (resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30)),
                             resource.setrlimit(resource.RLIMIT_CPU, (10, 10)))
            try:
                return subprocess.run([sys.executable, "-I", path], cwd=d, timeout=20, preexec_fn=limit,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            except subprocess.TimeoutExpired:
                return False
    return check


def calls(name, **needs):
    """The answer's first tool call is `name` and each named argument contains the given text."""
    def check(a):
        if not a["calls"] or a["calls"][0][0] != name:
            return False
        try:
            got = json.loads(a["calls"][0][1])
        except ValueError:
            return False
        return all(v in str(got.get(k, "")) for k, v in needs.items())
    return check


def looks(text):
    """The answer's first tool call inspects the repository, with valid JSON arguments that contain `text`."""
    def check(a):
        if not a["calls"] or a["calls"][0][0] not in ("read_file", "grep", "glob", "bash", "git_status"):
            return False
        try:
            return isinstance(json.loads(a["calls"][0][1] or "{}"), dict) and text in a["calls"][0][1]
        except ValueError:
            return False
    return check


def fixes_cache(a):
    """An edit (or rewrite) of cache.py that makes `get` move the entry to the recent end."""
    if not a["calls"] or a["calls"][0][0] not in ("edit_file", "write_file") or "cache.py" not in a["calls"][0][1]:
        return False
    return "move_to_end" in a["calls"][0][1] or "self._data[key] = " in a["calls"][0][1]


def user(text):
    return [{"role": "user", "content": text}]


def read_turns(n, path, content):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": f"call_read_{n}", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": path})}}]},
            {"role": "tool", "tool_call_id": f"call_read_{n}", "content": tc.numbered(content)}]


AGENT = [{"role": "system", "content": tc.SYSTEM}]
TASKS = [  # (kind, name, messages, tools, check)
    ("trivial", "capital", user("What is the capital of Australia? Reply with one word."), None, word("canberra")),
    ("trivial", "echo", user("Reply with exactly the word: ready"), None, word("ready")),
    ("reasoning", "screws", user("A shelf holds 17 boxes of 23 screws each, and 9 loose screws lie beside it. How many "
                                 "screws is that in total? Reply with only the number."), None, number(400)),
    ("reasoning", "trains", user("Two trains start 390 km apart and drive towards each other at 70 km/h and 60 km/h. "
                                 "After how many hours do they meet? Reply with only the number."), None, number(3)),
    ("reasoning", "order", user("Anna is taller than Ben. Ben is taller than Cy. Dee is shorter than Cy. Eve is taller "
                                "than Anna. Who is the second shortest? Reply with only the name."), None, word("cy")),
    ("reasoning", "lcm", user("What is the smallest positive integer that is divisible by every integer from 1 to 12? "
                              "Reply with only the number."), None, number(27720)),
    ("reasoning", "weekday", user("If the 3rd of a month is a Tuesday, which weekday is the 27th? Reply with one word."),
     None, word("friday")),
    ("code-read", "sum", user(f"What does this Python program print? Reply with only the output.\n\n```python\n{SNIP_SUM}\n```"),
     None, number(60)),
    ("code-read", "dict", user(f"What does this Python program print? Reply with only the output.\n\n```python\n{SNIP_DICT}\n```"),
     None, lambda a: a["text"].replace(" ", "").strip() == "(3,6)"),
    ("code-read", "bug", user("This binary search sometimes never returns. Which line has the bug? Reply with only the "
                              f"line number.\n\n```\n{SNIP_BUG}\n```"), None, number(8)),
    ("code-write", "intervals", user("Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] "
                                     "lists in any order and returns the merged, sorted list; intervals that touch "
                                     "([1, 2] and [2, 3]) merge. Reply with one Python code block and nothing else."), None,
     runs("assert merge_intervals([[8, 10], [1, 3], [2, 6], [15, 18], [18, 20]]) == [[1, 6], [8, 10], [15, 20]]\n"
          "assert merge_intervals([]) == []\nassert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]")),
    ("code-write", "roman", user("Write a Python function `roman_to_int(s)` for Roman numerals up to 3999. Reply with one "
                                 "Python code block and nothing else."), None,
     runs("assert [roman_to_int(x) for x in ('III', 'IV', 'IX', 'LVIII', 'MCMXCIV', 'MMMCMXCIX')] == [3, 4, 9, 58, 1994, 3999]")),
    ("code-write", "words", user("Write a Python function `top_k_words(text, k)` that returns the k most frequent words as a "
                                 "list of (word, count) tuples: words are runs of letters, compared in lower case; most "
                                 "frequent first, ties in alphabetical order. Reply with one Python code block and nothing else."),
     None, runs("assert top_k_words('The cat and the dog. The end, and THE cat!', 3) == [('the', 4), ('and', 2), ('cat', 2)]\n"
                "assert top_k_words('', 2) == []")),
    ("tool", "first-call", AGENT + user(ASK_FIX), tc.TOOLS, looks("")),
    ("tool", "next-call", AGENT + user(ASK_FIX) + read_turns(1, "tests/test_dates.py", TEST_FILE), tc.TOOLS, looks("dates")),
    ("tool", "edit", AGENT + user(ASK_FIX) + read_turns(1, "tests/test_dates.py", TEST_FILE)
     + read_turns(2, "dates.py", SRC_FILE), tc.TOOLS, calls("edit_file", path="dates.py", new_string="%Y-%m-%d")),
    ("hard", "evaluator", user("Write a Python function `evaluate(expr)` that returns the value of an arithmetic expression "
                               "given as a string, as a float: numbers with an optional decimal part, + - * /, parentheses, "
                               "unary minus, spaces anywhere, the usual precedence. Do not use eval, exec or ast. Reply "
                               "with one Python code block and nothing else."), None,
     runs("for e, v in (('2+3*4', 14), ('-(2+3)*4', -20), ('2*(3+4)-5/2', 11.5), (' 10 / 4 ', 2.5), ('2--3', 5),\n"
          "             ('((1.5))*-2', -3), ('8/2/2', 2), ('1-2-3', -4)):\n    assert abs(evaluate(e) - v) < 1e-9, e")),
    ("hard", "lru", user("Write a Python class `LRUCache(capacity)` with `get(key)` (returns -1 if absent) and "
                         "`put(key, value)`, both O(1) and both counting as a use; when full, `put` of a new key drops "
                         "the least recently used one. Do not use OrderedDict or functools. Reply with one Python code "
                         "block and nothing else."), None,
     runs("c = LRUCache(2)\nc.put(1, 1); c.put(2, 2)\nassert c.get(1) == 1\nc.put(3, 3)\nassert c.get(2) == -1\n"
          "c.put(4, 4)\nassert (c.get(1), c.get(3), c.get(4)) == (-1, 3, 4)\nc.put(3, 30); c.put(5, 5)\n"
          "assert (c.get(4), c.get(3), c.get(5)) == (-1, 30, 5)")),
    ("hard", "count", user("How many integers from 1 to 1000 are divisible by 3 or by 5, but not by 7? Reply with only "
                           "the number."), None, number(401)),
    ("hard", "puzzle", user("Ada, Ben, Cho, Dev and Eli ran a race with no ties. Ada finished before Ben. Cho finished "
                            "immediately after Dev. Eli was neither first nor last. Ben finished exactly two places "
                            "after Eli. Give the finishing order, first to last: only the five names, separated by spaces."),
     None, lambda a: re.findall(r"[A-Za-z]+", a["text"]) == ["Dev", "Cho", "Eli", "Ada", "Ben"]),
    ("hard", "trace", user(f"What does this Python program print? Reply with only the output.\n\n```python\n{SNIP_WALK}\n```"),
     None, lambda a: a["text"].replace(" ", "").strip() == "[(6,24),(8,69)]41"),
    ("hard", "cache-fix", AGENT + user(ASK_CACHE) + read_turns(1, "tests/test_cache.py", CACHE_TEST)
     + read_turns(2, "cache.py", CACHE_FILE), tc.TOOLS, fixes_cache),
]
KINDS = [k for k in dict.fromkeys(t[0] for t in TASKS) if not args.only or k in args.only.split(",")]
TASKS = [t for t in TASKS if t[0] in KINDS]


def thinking_tokens(text):
    if not text:
        return 0
    status, data = cl.post("tokenize", {"prompt": text})
    return len(json.loads(data)["tokens"]) if status == 200 else None


rows, lock = [], threading.Lock()
jobs = []
for p in range(args.passes):
    one = [(p, lv, t) for lv in LEVELS for t in TASKS]
    random.Random(p).shuffle(one)
    jobs += one
todo = iter(jobs)


def worker():
    while True:
        with lock:
            job = next(todo, None)
        if job is None:
            return
        p, lv, (kind, name, messages, tools, check) = job
        a = cl.chat(messages, args.limit, args.temp, tools=tools, extra={"reasoning_effort": lv})
        row = {"pass": p, "level": lv, "kind": kind, "task": name, "error": a["error"], "finish": a["finish"],
               "thinking": thinking_tokens(a["reasoning"]), "out_tokens": a["usage"].get("completion_tokens"),
               "secs": round(a["secs"], 1), "right": bool(not a["error"] and a["finish"] != "length" and check(a)),
               "text": a["text"], "calls": a["calls"], "reasoning": a["reasoning"]}
        with lock:
            rows.append(row)
            print(f"{time.strftime('%H:%M:%S', time.gmtime())} pass {p} {lv:8s} {kind:10s} {name:10s} thinking {row['thinking']} "
                  f"of {row['out_tokens']} tokens, {row['secs']:.1f} s, {'right' if row['right'] else 'NOT RIGHT'}"
                  + (f" finish {a['finish']}" if a["finish"] == "length" else "") + (f" | {a['error']}" if a["error"] else ""),
                  flush=True)
            json.dump({"args": vars(args), "rows": rows}, open(os.path.join(LOGS, f"effort-{args.label}.json"), "w"))


threads = [threading.Thread(target=worker) for _ in range(args.parallel)]
[t.start() for t in threads]
[t.join() for t in threads]


def cell(sel):
    th = sorted(r["thinking"] or 0 for r in sel)
    if not th:
        return "-"
    return f"{th[len(th) // 2]:5d} ({th[0]}-{th[-1]}) {sum(r['right'] for r in sel)}/{len(sel)}"


print(f"\nthinking tokens: median (least-most) right/asked; temperature {args.temp}, {args.passes} pass(es), limit {args.limit}")
print(f"{'level':8s} | " + " | ".join(f"{k:24s}" for k in KINDS + ["all"]))
for lv in LEVELS:
    at = [r for r in rows if r["level"] == lv and not r["error"]]
    print(f"{lv:8s} | " + " | ".join(f"{cell([r for r in at if r['kind'] == k]):24s}" for k in KINDS)
          + f" | {cell(at)}  mean {sum(r['thinking'] or 0 for r in at) / max(1, len(at)):.0f}"
          + f", answer+thinking {sum(r['out_tokens'] or 0 for r in at) / max(1, len(at)):.0f}"
          + f", {sum(r['secs'] for r in at) / max(1, len(at)):.1f} s")
bad = sum(1 for r in rows if r["error"])
cut = sum(1 for r in rows if r["finish"] == "length")
print(f"{len(rows)} requests, {bad} errors, {cut} stopped at the limit")
sys.exit(1 if bad else 0)
