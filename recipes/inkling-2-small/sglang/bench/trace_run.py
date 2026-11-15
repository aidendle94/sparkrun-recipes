"""Workload for a trace window: a boot with NCCL_EXTRA="-e SPARK_TRACE=/root/.cache/spark-trace/<label>
-e SGLANG_DSPARK_DEBUG_DUMP=all" (patches/spark_trace.py; planned windows only, never the serving fleet's default).

    python3 bench/trace_run.py <label> [--tokens 400] [--port 8210] [--phases g1,s1,g4,g8,copy,long]
Ordinary completions over 24 prompts of different kinds, every request with rid "<phase>-<prompt>" so the trace can be
split by phase and by prompt (bench/trace_sim.py fits on even prompts and scores on odd ones):
  g1    each prompt alone, greedy          s1   each prompt alone, temperature 0.7
  g4    four at a time, greedy             g8   eight at a time, greedy
  copy  alone, greedy: edit a given file (most of the answer copies the prompt)
  long  alone, greedy, behind a 60K-token prompt of its own
Then a pause and a one-token request (the recorder writes its tail when a step arrives after 2 s of silence), and the
server's DSpark step records (/server_info) go to logs/dspark-info-<label>.json, the requests to logs/trace-<label>.json.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--tokens", type=int, default=400)
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--phases", default="g1,s1,g4,g8,copy,long")
args = ap.parse_args()
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
Q = ["Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
     "Explain how a bill becomes law in the United States, step by step, for a high-school student.",
     "Implement an LRU cache class in Python with get and put in O(1), without functools.",
     "Write a short story about a lighthouse keeper who finds a message in a bottle.",
     "Write a C function that reverses a singly linked list in place, then one that detects a cycle.",
     "Translate into French, German and Spanish: 'The meeting has been moved to Thursday at ten; please bring the "
     "signed contracts and last quarter's figures.' Then explain the differences in word order.",
     "Write a Rust function that parses a semantic version string into a struct, with error handling.",
     "Prove that the square root of 2 is irrational, then that there are infinitely many primes.",
     "Implement binary search over a sorted array in Go, returning the insertion point when the key is missing.",
     "Compare the causes of the First and the Second World War in about 300 words.",
     "Write a SQL query that finds the top three customers by revenue per month, with a window function.",
     "Give a week's vegetarian dinner plan with a shopping list, as a JSON object with one key per day.",
     "Write a JavaScript debounce function with leading and trailing options, and explain each branch.",
     "Explain what happens, physically, when ice melts in a glass of water, and why the level does not change.",
     "Implement Dijkstra's algorithm in Python with a binary heap and reconstruct the shortest path.",
     "Write a polite but firm email declining a vendor's proposal and asking for a revised quote by Friday.",
     "Write a bash script that rotates the logs in a directory: gzip files older than a day, delete those older than 30.",
     "Describe the symptoms, causes and usual treatment of iron-deficiency anaemia for a patient leaflet.",
     "Write a TypeScript generic function that groups an array of objects by a key, with full type inference.",
     "Solve step by step: a train leaves at 9:40 at 72 km/h, another leaves the same station at 10:10 at 96 km/h. "
     "When and where does the second catch the first?",
     "Write a Dockerfile and a docker-compose.yml for a Flask app with a Postgres database and a healthcheck.",
     "Write a sonnet about autumn rain, then a limerick about a cat who distrusts umbrellas.",
     "Write a Java class for a thread-safe bounded blocking queue using ReentrantLock and conditions.",
     "Summarize the plot of Romeo and Juliet in ten bullet points, then name its three main themes."]
SOURCE = '''import csv
import sys
from collections import defaultdict


def load_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def revenue_by_customer(rows):
    totals = defaultdict(float)
    for row in rows:
        totals[row["customer"]] += float(row["quantity"]) * float(row["unit_price"])
    return totals


def top_customers(totals, count):
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return ranked[:count]


def format_report(ranked):
    lines = ["customer,revenue"]
    for customer, revenue in ranked:
        lines.append(f"{customer},{revenue:.2f}")
    return "\\n".join(lines)


def main(argv):
    if len(argv) != 3:
        print("usage: report.py <orders.csv> <count>")
        return 2
    rows = load_rows(argv[1])
    totals = revenue_by_customer(rows)
    print(format_report(top_customers(totals, int(argv[2]))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''
COPY = ["Here is report.py:\n\n```python\n" + SOURCE + "```\n\nRename `revenue_by_customer` to `customer_revenue` everywhere "
        "and print the whole file again. Only the code.",
        "Here is report.py:\n\n```python\n" + SOURCE + "```\n\nAdd a `--min-revenue` threshold that drops customers below it, "
        "and print the whole file again. Only the code."]


def post(route, **body):
    req = urllib.request.Request(f"http://localhost:{args.port}/{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def tokenize(text):
    return post("tokenize", messages=[{"role": "user", "content": text}])["tokens"]


def d4(o):
    grams = [tuple(o[i:i + 4]) for i in range(len(o) - 3)]
    return len(set(grams)) / max(len(grams), 1)


rows = []


def run(phase, prompts, width, tokens=None, **sampling):
    """prompts: {name: ids}; `width` of them decode together."""
    names = list(prompts)
    for at in range(0, len(names), width):
        group, outs = names[at:at + width], {}

        def one(name):
            outs[name] = post("generate", input_ids=prompts[name], rid=f"{phase}-{name}",
                              sampling_params=dict({"temperature": 0, "max_new_tokens": tokens or args.tokens}, **sampling))
        threads = [threading.Thread(target=one, args=(n,)) for n in group]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        for name in group:
            m, o = outs[name]["meta_info"], outs[name]["output_ids"]
            decode_s = m["e2e_latency"] - m["prefill_latency"]
            row = {"phase": phase, "prompt": name, "width": len(group), "prompt_tokens": m["prompt_tokens"],
                   "tokens": m["completion_tokens"], "decode_s": round(decode_s, 2),
                   "tok_s": round(m["completion_tokens"] / decode_s, 1), "accept": round(m.get("spec_accept_length") or 0, 2),
                   "d4": round(d4(o), 2)}
            print(time.strftime("%T"), row, repr(outs[name]["text"][:60]), flush=True)
            rows.append(dict(row, ids=o, text=outs[name]["text"], prompt_ids=prompts[name]))


ids = {f"q{i:02d}": tokenize(q) for i, q in enumerate(Q)}
phases = args.phases.split(",")
if "g1" in phases:
    run("g1", ids, 1)
if "s1" in phases:
    run("s1", ids, 1, temperature=0.7)
if "g4" in phases:
    run("g4", ids, 4)
if "g8" in phases:
    run("g8", ids, 8)
if "copy" in phases:
    run("copy", {f"c{i}": tokenize(c) for i, c in enumerate(COPY)}, 1, tokens=600)
if "long" in phases:
    rng = random.Random(7)
    text = " ".join(rng.choice(WORDS) for _ in range(int(60000 / 1.3))) + "\n\nIgnore the words above. " + Q[2]
    run("long", {"q02": tokenize(text)}, 1)
time.sleep(3)
post("generate", input_ids=ids["q00"], rid="flush", sampling_params={"temperature": 0, "max_new_tokens": 24})
json.dump(rows, open(os.path.join(LOGS, f"trace-{args.label}.json"), "w"))
info = json.load(urllib.request.urlopen(f"http://localhost:{args.port}/server_info", timeout=600))
records = [s.get("dspark_info_record") for s in info.get("internal_states", [])]
json.dump(records, open(os.path.join(LOGS, f"dspark-info-{args.label}.json"), "w"))
print("dspark records:", [len(r.get("records", r)) if r else 0 for r in records], flush=True)
for phase in dict.fromkeys(r["phase"] for r in rows):
    sel = [r for r in rows if r["phase"] == phase]
    print(f"{phase}: {len(sel)} requests, accept {sum(r['accept'] for r in sel) / len(sel):.2f}, "
          f"per-stream tok/s {sum(r['tok_s'] for r in sel) / len(sel):.1f}, min d4 {min(r['d4'] for r in sel):.2f}", flush=True)
