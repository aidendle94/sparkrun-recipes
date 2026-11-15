"""Does the server put an answer's reasoning back when the client's history leaves it out? (patches/reasoning_cache.py)

    python3 bench/reasoning_cache_check.py [--expect on|off] [--port 8210] [--model inkling-2-small]
Ordinary small requests, greedy (safe on a serving fleet). Per wire format, one first turn that reasons, then the next
request of that conversation twice with a 1-token answer, and the prompt tokens the server counted for each:
  kept     the assistant turn is sent back with its reasoning (reasoning_content, or a thinking block on /v1/messages)
  dropped  the assistant turn is sent back with its answer only, as most OpenAI-style clients do
With the cache on, `dropped` must count exactly as many prompt tokens as `kept` (the server restored the reasoning);
with it off (unset, or the flag file absent) `dropped` is shorter by the reasoning.
Formats: chat, chat-stream, chat-tool (a tool call and its result), messages, messages-stream, messages-stream-tool.
Exit 1 if a format does not match --expect or its first turn did not reason.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import argparse
import http.client
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--expect", choices=("on", "off"), default="on")
ap.add_argument("--port", type=int, default=8210)
ap.add_argument("--model", default="inkling-2-small")
args = ap.parse_args()
Q1 = "A shelf holds 17 boxes of 23 screws. How many screws is that? Reply with only the number."
Q2 = "And with 9 more screws? Only the number."
QT = "What is the weather in Lisbon right now? Use the tool."
TOOL = {"name": "get_weather", "description": "Current weather of a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
RESULT = '{"city": "Lisbon", "temp_c": 19, "sky": "clear"}'


def post(route, body):
    c = http.client.HTTPConnection("localhost", args.port, timeout=600)
    c.request("POST", route, json.dumps(dict(body, model=args.model, temperature=0)),
              {"content-type": "application/json", "x-api-key": "x", "anthropic-version": "2023-06-01"})
    r = c.getresponse()
    data = r.read().decode(errors="replace")
    c.close()
    assert r.status == 200, f"{route}: HTTP {r.status} {data[:300]}"
    if not body.get("stream"):
        return json.loads(data)
    return [json.loads(line[5:]) for line in data.splitlines() if line.startswith("data:") and line.strip() != "data: [DONE]"]


def chat_turn(messages, stream, tools=None):
    """First turn on /v1/chat/completions -> (reasoning, assistant message without it)."""
    body = {"messages": messages, "max_tokens": 800, "stream": stream}
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
    d = post("/v1/chat/completions", body)
    if not stream:
        m = d["choices"][0]["message"]
        text, reasoning, calls = m.get("content") or "", m.get("reasoning_content") or "", m.get("tool_calls") or []
    else:
        text, reasoning, slots = "", "", {}
        for ev in d:
            for ch in ev.get("choices") or ():
                delta = ch.get("delta") or {}
                text += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                for t in delta.get("tool_calls") or ():
                    s = slots.setdefault(t.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    s["id"] += t.get("id") or ""
                    s["function"]["name"] += (t.get("function") or {}).get("name") or ""
                    s["function"]["arguments"] += (t.get("function") or {}).get("arguments") or ""
        calls = [slots[i] for i in sorted(slots)]
    reply = {"role": "assistant", "content": text}
    if calls:
        reply["tool_calls"] = [{"id": t["id"], "type": "function", "function": {"name": t["function"]["name"],
                                                                                   "arguments": t["function"]["arguments"]}} for t in calls]
    return reasoning, reply


def chat_count(messages, tools=None):
    body = {"messages": messages, "max_tokens": 1, "stream": False}
    if tools:
        body["tools"] = [{"type": "function", "function": t} for t in tools]
    return post("/v1/chat/completions", body)["usage"]["prompt_tokens"]


def chat_case(stream, tool):
    first = [{"role": "user", "content": QT if tool else Q1}]
    tools = [TOOL] if tool else None
    reasoning, reply = chat_turn(first, stream, tools)
    if tool and not reply.get("tool_calls"):
        return None, None, "the first turn made no tool call"
    after = ([{"role": "tool", "tool_call_id": reply["tool_calls"][0]["id"], "content": RESULT}] if tool
             else [{"role": "user", "content": Q2}])
    kept = chat_count(first + [dict(reply, reasoning_content=reasoning)] + after, tools)
    dropped = chat_count(first + [reply] + after, tools)
    return kept, dropped, "" if reasoning.strip() else "the first turn did not reason"


def messages_turn(messages, stream, tools=None):
    """First turn on /v1/messages -> the content blocks of the answer."""
    body = {"messages": messages, "max_tokens": 800, "stream": stream}
    if tools:
        body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
    d = post("/v1/messages", body)
    if not stream:
        return d["content"]
    blocks, partial = {}, {}
    for ev in d:
        if ev.get("type") == "content_block_start":
            blocks[ev["index"]] = dict(ev["content_block"])
        elif ev.get("type") == "content_block_delta":
            b, delta = blocks[ev["index"]], ev["delta"]
            if delta.get("type") == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + delta["thinking"]
            elif delta.get("type") == "text_delta":
                b["text"] = b.get("text", "") + delta["text"]
            elif delta.get("type") == "input_json_delta":
                partial[ev["index"]] = partial.get(ev["index"], "") + delta["partial_json"]
    for i, raw in partial.items():
        blocks[i]["input"] = json.loads(raw) if raw.strip() else {}
    return [blocks[i] for i in sorted(blocks)]


def messages_case(stream, tool):
    first = [{"role": "user", "content": QT if tool else Q1}]
    tools = [TOOL] if tool else None
    blocks = messages_turn(first, stream, tools)
    thinking = [b for b in blocks if b.get("type") == "thinking" and (b.get("thinking") or "").strip()]
    uses = [b for b in blocks if b.get("type") == "tool_use"]
    if tool and not uses:
        return None, None, "the first turn made no tool call"
    after = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"], "content": RESULT}] if tool else Q2}]
    body = {"max_tokens": 1, "stream": False}
    if tools:
        body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
    count = lambda content: post("/v1/messages", dict(body, messages=first + [{"role": "assistant", "content": content}] + after))["usage"]["input_tokens"]
    kept = count(blocks)
    dropped = count([b for b in blocks if b.get("type") != "thinking"])
    return kept, dropped, "" if thinking else "the first turn did not reason"


CASES = [("chat", lambda: chat_case(False, False)), ("chat-stream", lambda: chat_case(True, False)),
         ("chat-tool", lambda: chat_case(False, True)), ("messages", lambda: messages_case(False, False)),
         ("messages-stream", lambda: messages_case(True, False)), ("messages-stream-tool", lambda: messages_case(True, True))]
bad = 0
for name, run in CASES:
    try:
        kept, dropped, note = run()
    except Exception as e:
        kept, dropped, note = None, None, f"{type(e).__name__}: {str(e)[:200]}"
    if note:
        verdict = "NOT CHECKED: " + note
    elif args.expect == "on":
        verdict = "restored" if dropped == kept else "NOT RESTORED"
    else:
        verdict = "left out, as sent" if dropped < kept else "UNEXPECTED: the same count with the cache off"
    ok = verdict in ("restored", "left out, as sent")
    bad += not ok
    print(f"{name:21s} prompt tokens: kept {kept}, dropped {dropped} -> {verdict}", flush=True)
print(f"cache expected {args.expect}: " + ("ALL OK" if not bad else f"{bad} FORMAT(S) NOT AS EXPECTED"))
sys.exit(1 if bad else 0)
