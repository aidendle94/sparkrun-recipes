#!/usr/bin/env python3
"""Prefill on DEGENERATE filler (one word repeated), the kind of input behind published 3-4K tok/s prefill tables.
A unique tag at the start defeats the prefix cache; the needle sits at 50% depth; TTFT = prefill time.

usage: prefill_repetitive.py http://127.0.0.1:8210 32768,131072
The streaming request loop is from tonyd2wild's bench/v41needle.py (Copyright (c) 2026 Tech2wild, MIT License;
see NOTICE). MIT License, Copyright (c) 2026 Aiden Le."""
import json, sys, time, urllib.request
base = sys.argv[1]; targets = [int(x) for x in sys.argv[2].split(",")]
NEEDLE = "Note for the record: the vault passphrase is COPPER-LANTERN-8315."
for tgt in targets:
    words = ["the"] * int(tgt * 0.98)  # 'the ' ~1 token each
    half = len(words) // 2
    prompt = f"[session {time.time_ns()}] " + " ".join(words[:half]) + " " + NEEDLE + " " + " ".join(words[half:]) + \
             "\n\nWhat is the vault passphrase mentioned in the text above? Reply with the passphrase only."
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 24,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; usage = None; text = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
            ev = json.loads(line[5:].strip())
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and ttft is None: ttft = time.time() - t0
                text += piece
    pt = usage["prompt_tokens"] if usage else 0
    print(json.dumps({"target": tgt, "prompt_tokens": pt, "ttft_s": round(ttft, 1), "prefill_tok_s": round(pt / ttft), "pass": "COPPER-LANTERN-8315" in text}))
