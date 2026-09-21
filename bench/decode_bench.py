#!/usr/bin/env python3
"""Single-stream decode benchmark against an OpenAI-compatible chat endpoint (DeepSeek-V4.1-Flash on SGLang).

Three regimes, each a fixed prompt, each run --runs times; the best run counts (the first run warms the prefix
cache and any JIT kernels, so a single run understates steady-state speed):
  counting  count from 1 to 300 separated by spaces, numbers only   (short tokens; the draft model's easy case)
  code      a Python function followed by unit tests                (identifiers and punctuation)
  prose     a ~350-word essay without headings                      (ordinary language)
Greedy decoding (temperature 0), thinking off (chat_template_kwargs {"thinking": false}), streamed with
stream_options.include_usage so the final chunk carries usage.completion_tokens: the server's own token count,
which is the only reliable one under speculative decoding (one streamed chunk may carry several tokens).

Per regime: tok/s = completion_tokens / (time of the last content chunk - time of the first content chunk),
time to first token (request sent -> first content chunk) and the token count.

usage: decode_bench.py [BASE] [--model deepseek-v4.1-flash] [--max-tokens 400] [--runs 2]
                       [--regimes counting,code,prose] [--timeout 600]
  BASE  http://127.0.0.1:8210 (default); with or without a trailing /v1.
Prints one progress line per run, then the summary
  decode sample (single stream): counting X tok/s, code Y tok/s, prose Z tok/s
and a JSON summary line. Exit status 1 on any request error (connection, HTTP status, malformed stream, missing
usage), 2 on bad arguments. Standard library only.

The request body and the streaming loop follow tonyd2wild's bench/v41needle.py (Copyright (c) 2026
Tech2wild, MIT License; see NOTICE). MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.request

PROMPTS = {
    "counting": "Count from 1 to 300, separated by single spaces. Output the numbers only, nothing else.",
    "code": ("Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns "
             "the merged list of non-overlapping intervals sorted by start. Follow it with a unittest.TestCase class "
             "with at least six test methods (empty input, one interval, touching intervals, nested intervals, "
             "unsorted input, no overlaps) and the usual __main__ guard. Output only the code, no explanation."),
    "prose": ("Write an essay of about 350 words on why tidal power has not been adopted as widely as wind power. "
              "Plain paragraphs only: no title, no headings, no lists."),
}
SUMMARY_ORDER = ("counting", "code", "prose")


class RequestError(Exception):
    """A request that did not yield a usable measurement (transport, HTTP status, stream shape, no usage)."""


def stream_once(base: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    """One streamed chat completion; returns tokens, ttft_s, span_s, chunks, tok_s (None if a single chunk)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_send = time.perf_counter()
    t_first = t_last = None
    usage = None
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if (choice.get("delta") or {}).get("content"):
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        chunks += 1
    except urllib.error.HTTPError as e:
        raise RequestError(f"HTTP {e.code}: {e.read(300).decode('utf-8', 'ignore').strip()}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as e:
        raise RequestError(f"{type(e).__name__}: {e}") from None
    if t_first is None:
        raise RequestError("the stream carried no content chunk")
    tokens = (usage or {}).get("completion_tokens")
    if not isinstance(tokens, int):
        raise RequestError("the final chunk carried no usage.completion_tokens (stream_options.include_usage unsupported?)")
    span = t_last - t_first
    return {"tokens": tokens, "ttft_s": t_first - t_send, "span_s": span, "chunks": chunks,
            "tok_s": tokens / span if span > 0 else None}


def fmt_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="single-stream decode benchmark (three fixed prompts, best of --runs)")
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8210", help="server base URL, with or without /v1")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--regimes", default=",".join(SUMMARY_ORDER), help="comma-separated subset of counting,code,prose")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds allowed per request")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    bad = [r for r in regimes if r not in PROMPTS]
    if bad or not regimes or args.runs < 1:
        print(f"error: --regimes must name some of {','.join(SUMMARY_ORDER)} and --runs must be >= 1", file=sys.stderr)
        return 2

    best: dict[str, dict] = {}
    for regime in regimes:
        for run in range(1, args.runs + 1):
            try:
                r = stream_once(base, args.model, PROMPTS[regime], args.max_tokens, args.timeout)
            except RequestError as e:
                print(f"error: {regime} run {run}: {e}", file=sys.stderr)
                return 1
            print(f"  {regime} run {run}: {r['tokens']} tokens in {r['chunks']} chunks, ttft {r['ttft_s']:.2f} s, "
                  f"{fmt_rate(r['tok_s'])} tok/s", flush=True)
            if regime not in best or (r["tok_s"] or -1.0) > (best[regime]["tok_s"] or -1.0):
                best[regime] = r

    ordered = [r for r in SUMMARY_ORDER if r in best]
    print("decode sample (single stream): " + ", ".join(f"{r} {fmt_rate(best[r]['tok_s'])} tok/s" for r in ordered))
    summary = {"base": base, "model": args.model, "max_tokens": args.max_tokens, "runs": args.runs,
               "regimes": {r: {"tok_s": None if best[r]["tok_s"] is None else round(best[r]["tok_s"], 2),
                               "ttft_s": round(best[r]["ttft_s"], 3), "completion_tokens": best[r]["tokens"],
                               "chunks": best[r]["chunks"], "decode_span_s": round(best[r]["span_s"], 3)}
                           for r in ordered}}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
