"""Tool-call correctness of the serving fleet for a coding agent (ordinary chat requests: safe on production).

    python3 bench/toolcall_check.py <label> [--trials 2] [--parallel 4] [--apis chat,chat-stream,...] [--only read,edit]
                                    [--skip-cases] [--skip-agent] [--skip-effort] [--port 8210] [--model inkling-2-small]
1. cases   every case of bench/toolcall_cases.py over every wire format (bench/toolcall_api.py), `trials` times:
           trial 0 greedy, the others with the server's default sampling. `parallel` requests at a time.
2. agent   three small repositories with a failing test each, driven as an agent loop (list_dir, read_file,
           edit_file, write_file, run_tests) until the tests pass, on chat-stream and anthropic-stream, all at once.
           run_tests runs the model's files with python3 -I under CPU/memory limits in a temporary directory.
3. effort  one task at each reasoning effort (none ... max, and a float on chat): reasoning length, tokens, time.
Prints a case x api table (ok / soft / FAIL counts), every non-ok note, and writes logs/toolcall-<label>.json.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import concurrent.futures
import json
import os
import random
import resource
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toolcall_api  # noqa: E402
import toolcall_cases as tc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--trials", type=int, default=2)
ap.add_argument("--parallel", type=int, default=4)
ap.add_argument("--apis", default=",".join(toolcall_api.APIS))
ap.add_argument("--only", default="")
ap.add_argument("--skip-cases", action="store_true")
ap.add_argument("--skip-agent", action="store_true")
ap.add_argument("--skip-effort", action="store_true")
ap.add_argument("--drop-reasoning", action="store_true", help="send the history without the assistant's reasoning")
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--model", default="inkling-2-small")
args = ap.parse_args()
toolcall_api.PORT, toolcall_api.MODEL = args.port, args.model
toolcall_api.DROP_REASONING = args.drop_reasoning
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
apis = args.apis.split(",")
results = []


def stamp():
    return time.strftime("%T")


def run_case(case, api, trial):
    name = case.__name__[2:]
    rng = random.Random(f"{name}-{trial}")
    start = time.time()
    try:
        verdict, note, answers = case(api, rng)
    except Exception as e:  # a bug in a check must not take the run down
        verdict, note, answers = "FAIL", f"check raised {type(e).__name__}: {e}", []
    row = {"part": "cases", "case": name, "api": api, "trial": trial, "verdict": verdict, "note": note,
           "secs": round(time.time() - start, 1), "out_tokens": sum(a["out_tokens"] or 0 for a in answers),
           "answers": answers}
    print(stamp(), f"{name:16s} {api:16s} t{trial} {verdict:4s} {row['secs']:6.1f}s {note[:150]}", flush=True)
    return row

# ---- agent loop ----


REPOS = {
    "median": {"stats.py": "def median(values):\n    ordered = sorted(values)\n    mid = len(ordered) // 2\n    return ordered[mid]\n",
               "test_stats.py": "import unittest\nfrom stats import median\n\n\nclass T(unittest.TestCase):\n"
                                "    def test_odd(self):\n        self.assertEqual(median([3, 1, 2]), 2)\n\n"
                                "    def test_even(self):\n        self.assertEqual(median([4, 1, 3, 2]), 2.5)\n\n"
                                "    def test_empty(self):\n        with self.assertRaises(ValueError):\n            median([])\n"},
    "chunks": {"README.md": "chunks(seq, n) splits seq into consecutive pieces of length n; the last may be shorter.\n",
               "util/__init__.py": "",
               "util/seq.py": "def chunks(seq, n):\n    return [seq[i:i + n] for i in range(0, len(seq) - n, n)]\n\n\n"
                              "def flatten(nested):\n    out = []\n    for item in nested:\n        out.extend(item)\n    return out\n",
               "test_seq.py": "import unittest\nfrom util.seq import chunks, flatten\n\n\nclass T(unittest.TestCase):\n"
                              "    def test_exact(self):\n        self.assertEqual(chunks([1, 2, 3, 4], 2), [[1, 2], [3, 4]])\n\n"
                              "    def test_tail(self):\n        self.assertEqual(chunks('abcde', 2), ['ab', 'cd', 'e'])\n\n"
                              "    def test_round_trip(self):\n        self.assertEqual(flatten(chunks(list(range(7)), 3)), list(range(7)))\n"},
    "slugify": {"text.py": "import re\n\n\ndef slugify(title):\n    return re.sub(r'\\s+', '-', title.lower())\n",
                "test_text.py": "import unittest\nfrom text import slugify, truncate\n\n\nclass T(unittest.TestCase):\n"
                                "    def test_punctuation(self):\n        self.assertEqual(slugify(\"Hello, World! It's 5 o'clock\"), 'hello-world-its-5-oclock')\n\n"
                                "    def test_edges(self):\n        self.assertEqual(slugify('  --Already--slugged--  '), 'already-slugged')\n\n"
                                "    def test_truncate(self):\n        self.assertEqual(truncate('abcdefghij', 6), 'abc...')\n"
                                "        self.assertEqual(truncate('abc', 6), 'abc')\n"},
}
AGENT_TOOLS = [t for t in tc.TOOLS if t["name"] in ("read_file", "edit_file", "write_file")] + [
    tc._tool("list_dir", "List the files of the repository (recursive).", []),
    tc._tool("run_tests", "Run the repository's unit tests (python3 -m unittest) and return the output.", [])]


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))


def run_tests(files):
    with tempfile.TemporaryDirectory() as tmp:
        for path, text in files.items():
            full = os.path.realpath(os.path.join(tmp, path))
            if not full.startswith(os.path.realpath(tmp) + os.sep):
                return False, f"refused path {path}"
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "w").write(text)
        try:
            p = subprocess.run([sys.executable, "-I", "-m", "unittest", "-q"], cwd=tmp, capture_output=True, text=True,
                               timeout=30, preexec_fn=_limits, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return p.returncode == 0, (p.stdout + p.stderr)[-3000:]


def run_agent(repo, api, trial):
    files, temperature = dict(REPOS[repo]), 0 if trial == 0 else None
    turns = [{"role": "user", "text": "The unit tests of this repository fail. Find out why, fix the code (not the "
                                      "tests) and confirm with run_tests. Say DONE when they pass."}]
    stats = {"turns": 0, "calls": 0, "invalid": [], "edit_misses": [], "passed": False, "out_tokens": 0}
    start = time.time()
    for _ in range(20):
        a = toolcall_api.ask(api, tc.SYSTEM, turns, AGENT_TOOLS, temperature=temperature, max_tokens=8192)
        stats["turns"] += 1
        stats["out_tokens"] += a["out_tokens"] or 0
        stats["invalid"] += tc.problems(a, AGENT_TOOLS)
        if a["finish"] == "error" or not a["calls"]:
            break
        turns.append(tc._assistant(a))
        for c in a["calls"]:
            stats["calls"] += 1
            call_args = c["args"] or {}
            if c["args"] is None or c["name"] not in {t["name"] for t in AGENT_TOOLS}:
                out = "Error: invalid tool call"
            elif c["name"] == "list_dir":
                out = "\n".join(sorted(files))
            elif c["name"] == "read_file":
                out = tc.numbered(files[call_args["path"]]) if call_args.get("path") in files else f"Error: no such file {call_args.get('path')}"
            elif c["name"] == "run_tests":
                stats["passed"], out = run_tests(files)
            else:
                one_call = dict(a, calls=[c])
                try:
                    bad = tc.apply_edits(one_call, files)
                except KeyError as e:
                    bad = [f"missing argument {e}"]
                stats["edit_misses"] += bad
                out = "Error: " + bad[0] if bad else "OK"
            turns.append({"role": "tool", "id": c["id"], "name": c["name"], "text": out})
    stats["passed"] = run_tests(files)[0]
    verdict = "FAIL" if stats["invalid"] or not stats["passed"] else "soft" if stats["edit_misses"] else "ok"
    note = (f"passed {stats['passed']} in {stats['turns']} turns, {stats['calls']} calls, invalid {len(stats['invalid'])}, "
            f"edit misses {len(stats['edit_misses'])} {(stats['invalid'] + stats['edit_misses'])[:2]}")
    print(stamp(), f"agent {repo:8s} {api:16s} t{trial} {verdict:4s} {time.time() - start:6.1f}s {note[:220]}", flush=True)
    return {"part": "agent", "case": f"agent-{repo}", "api": api, "trial": trial, "verdict": verdict, "note": note,
            "secs": round(time.time() - start, 1), "out_tokens": stats["out_tokens"], "turns": turns}

# ---- reasoning effort ----


def run_effort(api, effort):
    user = ("tests/test_billing.py::test_proration fails with `AssertionError: 33.33 != 33.34`. The code is in "
            "billing/proration.py. Start investigating.")
    a = toolcall_api.ask(api, tc.SYSTEM, [{"role": "user", "text": user}], tc.TOOLS, effort=effort, temperature=0)
    bad = tc.problems(a, tc.TOOLS)
    note = (f"reasoning {len(a['reasoning']):5d} chars, {a['out_tokens']:4d} tokens, {a['secs']:5.1f}s, "
            f"calls {[c['name'] for c in a['calls']]} {bad[:2] if bad else ''}")
    print(stamp(), f"effort {str(effort):8s} {api:16s} {note}", flush=True)
    return {"part": "effort", "case": f"effort-{effort}", "api": api, "trial": 0, "note": note,
            "verdict": "FAIL" if bad or not a["calls"] else "ok", "secs": a["secs"], "out_tokens": a["out_tokens"],
            "reasoning_chars": len(a["reasoning"]), "answers": [a]}


cases = [c for c in tc.CASES if not args.only or c.__name__[2:] in args.only.split(",")
         or c.__name__[2:].split("_")[0] in args.only.split(",")]
if args.skip_cases:
    cases = []
# TEMPERATURE is a module global: greedy and sampled trials run as separate passes so no thread sees the other's.
for trial in range(args.trials):
    tc.TEMPERATURE = 0 if trial == 0 else None
    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        jobs = [pool.submit(run_case, case, api, trial) for case in cases for api in apis]
        results += [j.result() for j in jobs]
if not args.skip_agent and not args.only:
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        jobs = [pool.submit(run_agent, repo, api, trial) for trial in range(args.trials) for repo in REPOS
                for api in ("chat-stream", "anthropic-stream") if api in apis]
        results += [j.result() for j in jobs]
if not args.skip_effort and not args.only:
    for api in apis:
        tiers = ["none", "minimal", "low", "medium", "high", "max"] + ([0.5] if api == "chat" else [])
        results += [run_effort(api, e) for e in tiers]
json.dump(results, open(os.path.join(LOGS, f"toolcall-{args.label}.json"), "w"))

print("\n== verdicts per case and wire format: ok/soft/FAIL")
names = list(dict.fromkeys(r["case"] for r in results))
print(f"{'':18s}" + "".join(f"{a:>18s}" for a in apis))
for name in names:
    cells = []
    for api in apis:
        sel = [r["verdict"] for r in results if r["case"] == name and r["api"] == api]
        cells.append("/".join(str(sel.count(v)) for v in ("ok", "soft", "FAIL")) if sel else "-")
    print(f"{name:18s}" + "".join(f"{c:>18s}" for c in cells))
total = [r["verdict"] for r in results]
print(f"\ntotal: {total.count('ok')} ok, {total.count('soft')} soft, {total.count('FAIL')} FAIL of {len(total)}")
print("\n== everything that was not ok")
for r in results:
    if r["verdict"] != "ok":
        print(f"{r['verdict']:4s} {r['case']:16s} {r['api']:16s} t{r['trial']} {r['note'][:400]}")
