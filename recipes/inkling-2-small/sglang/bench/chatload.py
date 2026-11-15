"""Shared pieces of bench/longctx.py and bench/stress.py: record contexts with exact answers, one chat request.

A context is a list of lines `item <6 digits> | code <6 digits>`, every item unique, so any line can be asked for and
the answer is checked digit for digit: a wrong key/value read, a wrong position or another request's context shows as
a wrong code, where fluent text would hide it. MIT License, Copyright (c) 2026 Aiden Le.
"""
import http.client
import json
import random
import re
import time

PORT, MODEL = 8210, "inkling-2-small"
# (host, path, key, model, extra body) of a hosted OpenAI-style endpoint: chat() then asks it over HTTPS instead of
# the fleet (tokens are still counted on the fleet: the same model, the same tokenizer). None: the fleet.
HOSTED = None
CONTROL = re.compile(r"<\|[a-z_]+\|>|</?(?:tool_call|think)>")
Q = ["Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.",
     "Implement an LRU cache class in Python with get and put in O(1), without functools.",
     "Write a C function that reverses a singly linked list in place, then one that detects a cycle.",
     "Write a Rust function that parses a semantic version string into a struct, with error handling.",
     "Implement binary search over a sorted array in Go, returning the insertion point when the key is missing.",
     "Write a SQL query that finds the top three customers by revenue per month, with a window function.",
     "Write a JavaScript debounce function with leading and trailing options, and explain each branch.",
     "Implement Dijkstra's algorithm in Python with a binary heap and reconstruct the shortest path."]


def post(route, body, timeout=3600):
    c = http.client.HTTPConnection("localhost", PORT, timeout=timeout)
    c.request("POST", "/" + route.lstrip("/"), json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def count_tokens(text):
    status, data = post("tokenize", {"messages": [{"role": "user", "content": text}]})
    return len(json.loads(data)["tokens"])


class Records:
    """`lines` records from a seed. text() is the context, ask(k) picks k items and returns (question, codes)."""

    def __init__(self, lines, seed):
        self.rng = random.Random(seed)
        items = self.rng.sample(range(100000, 1000000), lines)
        self.rows = [(i, self.rng.randrange(100000, 1000000)) for i in items]

    def text(self):
        return "\n".join(f"item {i} | code {c}" for i, c in self.rows)

    def where(self, want, got):
        """Where wrong codes come from: for each one, how many lines from the wanted item's line it stands in the records
        (+ later, - earlier), or 'absent' if no record has that code."""
        line = {}
        for j, (_, c) in enumerate(self.rows):
            line.setdefault(str(c), j)
        out = []
        for i, g in enumerate(got):
            if i < len(want) and g == want[i]:
                out.append("right")
            elif g not in line:
                out.append("absent")
            else:
                out.append(f"{line[g] - line[want[i]]:+d} lines" if i < len(want) and want[i] in line else f"line {line[g]}")
        return out

    def ask(self, k=3, depths=None):
        n = len(self.rows)
        at = [min(n - 1, int(d * n)) for d in depths] if depths else sorted(self.rng.sample(range(n), k))
        picked = [self.rows[j] for j in at]
        q = ("From the records, give the code of " + ", ".join(f"item {i}" for i, _ in picked)
             + ". Reply with only the codes, in this order, separated by spaces.")
        return q, [str(c) for _, c in picked]


def lines_for(tokens, per_line=[None]):
    if per_line[0] is None:  # tokens per record line, measured once on 400 lines
        per_line[0] = count_tokens(Records(400, 1).text()) / 400
    return max(8, int(tokens / per_line[0]))


def chat(messages, max_tokens, temperature=0, stream=True, tools=None, cancel_after=None, timeout=3600, extra=None):
    """One chat completion. Returns text, reasoning, calls [(name, raw arguments)], finish, usage, ttft, secs, error.
    cancel_after=n closes the connection after n content chunks (a client that gives up mid-answer)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": stream}
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
    if stream:
        body["stream_options"] = {"include_usage": True}
    body.update(extra or {})
    a = {"text": "", "reasoning": "", "calls": [], "finish": None, "usage": {}, "ttft": None, "error": None, "cancelled": False,
         "chunks": 0}
    start = time.time()
    try:
        if HOSTED:
            host, path, key, model, more = HOSTED
            body.update(more, model=model)
            c = http.client.HTTPSConnection(host, timeout=timeout)
            c.request("POST", path, json.dumps(body), {"content-type": "application/json", "authorization": "Bearer " + key})
        else:
            c = http.client.HTTPConnection("localhost", PORT, timeout=timeout)
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            a["error"] = f"HTTP {r.status}: {r.read()[:300].decode(errors='replace')}"
        elif not stream:
            d = json.loads(r.read())
            m = d["choices"][0]["message"]
            a["text"], a["reasoning"] = m.get("content") or "", m.get("reasoning_content") or m.get("reasoning") or ""
            a["calls"] = [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or ()]
            a["finish"], a["usage"] = d["choices"][0]["finish_reason"], d.get("usage") or {}
        else:
            calls = {}
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("error"):  # a hosted endpoint reports a failure inside the stream
                    a["error"] = f"stream error: {json.dumps(d['error'])[:300]}"
                    a["provider"] = d.get("provider") or a.get("provider")
                if d.get("provider"):
                    a["provider"] = d["provider"]
                if d.get("usage"):
                    a["usage"] = d["usage"]
                for ch in d.get("choices") or ():
                    delta = ch.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls"):
                        a["ttft"] = a["ttft"] or time.time() - start
                        a["chunks"] += 1
                    a["text"] += delta.get("content") or ""
                    a["reasoning"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
                    for t in delta.get("tool_calls") or ():
                        slot = calls.setdefault(t.get("index", 0), ["", ""])
                        slot[0] += (t.get("function") or {}).get("name") or ""
                        slot[1] += (t.get("function") or {}).get("arguments") or ""
                    a["finish"] = ch.get("finish_reason") or a["finish"]
                if cancel_after and a["chunks"] >= cancel_after:
                    a["cancelled"] = True
                    break
            a["calls"] = [tuple(v) for _, v in sorted(calls.items())]
        c.close()
    except Exception as e:  # a transport error is a result of the run, not a crash of it
        a["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    a["secs"] = time.time() - start
    return a


def codes_in(text):
    return re.findall(r"(?<!\d)\d{6}(?!\d)", text)


def d4w(text):
    """Distinct 4-grams of words / 4-grams (text that loops falls well below 0.5)."""
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / max(len(grams), 1)
