"""One conversation, five wire formats: the transport half of bench/toolcall_check.py.

ask(api, system, turns, tools, ...) sends a neutral conversation to the fleet and returns a neutral answer, so every
case runs unchanged over  chat | chat-stream | anthropic | anthropic-stream | responses  (/v1/chat/completions,
/v1/messages, /v1/responses). Streams are reassembled from their deltas, and what a client would trip over is
collected in answer["problems"]: arguments that are not a JSON object, calls without id or name, duplicate ids, a
finish reason that disagrees with the calls, control tokens in text, stream events out of order.
  turns   {"role": "user", "text"} | {"role": "assistant", "text", "reasoning", "calls": [{"id", "name", "args"}]}
          | {"role": "tool", "id", "name", "text"}
  answer  {"text", "reasoning", "calls": [{"id", "name", "raw", "args"}], "finish": tool_calls|stop|length|...,
           "problems": [...], "secs", "out_tokens"}
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import re
import time
import urllib.error
import urllib.request

APIS = ("chat", "chat-stream", "anthropic", "anthropic-stream", "responses")
PORT, MODEL = 8210, "inkling-2-small"
CONTROL = re.compile(r"<\|[a-z_]+\|>")


def _post(route, body, stream):
    req = urllib.request.Request(f"http://localhost:{PORT}{route}", json.dumps(body).encode(),
                                 {"content-type": "application/json", "anthropic-version": "2023-06-01",
                                  "x-api-key": "none", "authorization": "Bearer none"})
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}") from None
    if not stream:
        return json.load(resp)
    events, name = [], None
    for line in resp:
        line = line.decode().rstrip("\n")
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            events.append((name, data if data == "[DONE]" else json.loads(data)))
            name = None
    return events


DROP_REASONING = False  # True: history turns go back without their reasoning, as most OpenAI-style clients send them


def _chat_messages(system, turns):
    out = [{"role": "system", "content": system}] if system else []
    for t in turns:
        if t["role"] == "user":
            out.append({"role": "user", "content": t["text"]})
        elif t["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": t["id"], "content": t["text"]})
        else:
            m = {"role": "assistant", "content": t.get("text") or None}
            if t.get("reasoning") and not DROP_REASONING:
                m["reasoning_content"] = t["reasoning"]
            if t.get("calls"):
                m["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                    "name": c["name"], "arguments": json.dumps(c["args"])}} for c in t["calls"]]
            out.append(m)
    return out


def _anthropic_messages(turns):
    out = []
    for t in turns:
        if t["role"] == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": t["text"]}]})
        elif t["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": t["id"], "content": t["text"]}
            if out and out[-1]["role"] == "user" and out[-1]["content"][-1]["type"] == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            blocks = []
            if t.get("reasoning") and not DROP_REASONING:
                blocks.append({"type": "thinking", "thinking": t["reasoning"], "signature": t.get("signature") or ""})
            if t.get("text"):
                blocks.append({"type": "text", "text": t["text"]})
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                       for c in t.get("calls") or ()]
            out.append({"role": "assistant", "content": blocks})
    return out


def _responses_input(turns):
    out = []
    for t in turns:
        if t["role"] == "user":
            out.append({"role": "user", "content": t["text"]})
        elif t["role"] == "tool":
            out.append({"type": "function_call_output", "call_id": t["id"], "output": t["text"]})
        else:
            if t.get("text"):
                out.append({"role": "assistant", "content": t["text"]})
            out += [{"type": "function_call", "call_id": c["id"], "name": c["name"],
                     "arguments": json.dumps(c["args"])} for c in t.get("calls") or ()]
    return out


def _answer():
    return {"text": "", "reasoning": "", "calls": [], "finish": None, "problems": [], "out_tokens": 0}


def _from_chat(resp, a):
    choice = resp["choices"][0]
    m = choice["message"]
    a["text"], a["reasoning"] = m.get("content") or "", m.get("reasoning_content") or ""
    a["calls"] = [{"id": c.get("id"), "name": c["function"].get("name"), "raw": c["function"].get("arguments")}
                  for c in m.get("tool_calls") or ()]
    a["finish"] = choice.get("finish_reason")
    a["out_tokens"] = (resp.get("usage") or {}).get("completion_tokens", 0)


def _from_chat_stream(events, a):
    calls = {}
    if not events or events[-1][1] != "[DONE]":
        a["problems"].append("stream did not end with [DONE]")
    for _, e in events:
        if e == "[DONE]":
            continue
        if e.get("usage"):
            a["out_tokens"] = e["usage"].get("completion_tokens", 0)
        for choice in e.get("choices") or ():
            d = choice.get("delta") or {}
            a["text"] += d.get("content") or ""
            a["reasoning"] += d.get("reasoning_content") or ""
            for c in d.get("tool_calls") or ():
                slot = calls.setdefault(c.get("index"), {"id": None, "name": None, "raw": ""})
                f = c.get("function") or {}
                for key, new in (("id", c.get("id")), ("name", f.get("name"))):
                    if new:
                        if slot[key] and slot[key] != new:
                            a["problems"].append(f"stream: call {c.get('index')} {key} changed {slot[key]!r} -> {new!r}")
                        slot[key] = new
                if f.get("arguments"):
                    if slot["name"] is None:
                        a["problems"].append(f"stream: arguments of call {c.get('index')} before its name")
                    slot["raw"] += f["arguments"]
            if choice.get("finish_reason"):
                a["finish"] = choice["finish_reason"]
    if sorted(calls, key=str) != list(range(len(calls))):
        a["problems"].append(f"stream: call indexes {sorted(calls, key=str)}")
    a["calls"] = [calls[i] for i in sorted(calls, key=str)]


_STOP = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}


def _from_anthropic(resp, a):
    for b in resp.get("content") or ():
        if b["type"] == "text":
            a["text"] += b["text"]
        elif b["type"] == "thinking":
            a["reasoning"] += b.get("thinking") or ""
            a["signature"] = b.get("signature")
        elif b["type"] == "tool_use":
            if not isinstance(b.get("input"), dict):
                a["problems"].append(f"tool_use input is {type(b.get('input')).__name__}")
            a["calls"].append({"id": b.get("id"), "name": b.get("name"), "raw": json.dumps(b.get("input"))})
    a["finish"] = _STOP.get(resp.get("stop_reason"), resp.get("stop_reason"))
    a["out_tokens"] = (resp.get("usage") or {}).get("output_tokens", 0)


def _from_anthropic_stream(events, a):
    blocks, open_ = {}, set()
    names = [n for n, _ in events]
    if names[:1] != ["message_start"] or names[-1:] != ["message_stop"]:
        a["problems"].append(f"stream: starts {names[:1]} ends {names[-1:]}")
    for name, e in events:
        kind = e.get("type")
        if name and name != kind:
            a["problems"].append(f"stream: event {name} carries type {kind}")
        if kind == "content_block_start":
            b = dict(e["content_block"])
            b["raw"] = ""
            blocks[e["index"]] = b
            open_.add(e["index"])
        elif kind == "content_block_delta":
            b, d = blocks.get(e["index"]), e["delta"]
            if b is None or e["index"] not in open_:
                a["problems"].append(f"stream: delta for block {e['index']} that is not open")
                continue
            if d["type"] == "text_delta":
                b["text"] = (b.get("text") or "") + d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] = (b.get("thinking") or "") + d["thinking"]
            elif d["type"] == "signature_delta":
                b["signature"] = d.get("signature")
            elif d["type"] == "input_json_delta":
                b["raw"] += d["partial_json"]
        elif kind == "content_block_stop":
            open_.discard(e["index"])
        elif kind == "message_delta":
            a["finish"] = _STOP.get(e["delta"].get("stop_reason"), e["delta"].get("stop_reason"))
            a["out_tokens"] = (e.get("usage") or {}).get("output_tokens", 0)
        elif kind == "error":
            a["problems"].append(f"stream error event: {json.dumps(e)[:300]}")
    if open_:
        a["problems"].append(f"stream: blocks {sorted(open_)} never stopped")
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "text":
            a["text"] += b.get("text") or ""
        elif b["type"] == "thinking":
            a["reasoning"] += b.get("thinking") or ""
            a["signature"] = b.get("signature")
        elif b["type"] == "tool_use":
            a["calls"].append({"id": b.get("id"), "name": b.get("name"), "raw": b["raw"] or json.dumps(b.get("input") or {})})


def _from_responses(resp, a):
    for item in resp.get("output") or ():
        if item["type"] == "function_call":
            a["calls"].append({"id": item.get("call_id"), "name": item.get("name"), "raw": item.get("arguments")})
        elif item["type"] == "message":
            a["text"] += "".join(p.get("text") or "" for p in item.get("content") or ())
        elif item["type"] == "reasoning":
            a["reasoning"] += "".join(p.get("text") or "" for p in (item.get("content") or []) + (item.get("summary") or []))
    incomplete = resp.get("status") == "incomplete"
    a["finish"] = "length" if incomplete else "tool_calls" if a["calls"] else "stop" if resp.get("status") == "completed" \
        else resp.get("status")
    a["out_tokens"] = (resp.get("usage") or {}).get("output_tokens", 0)


def ask(api, system, turns, tools, choice=None, effort=None, temperature=None, max_tokens=4096):
    """choice: None (auto) | "none" | "required" | a tool name. effort: a tier name (or a float on chat)."""
    stream = api.endswith("-stream")
    if api.startswith("chat"):
        body = {"model": MODEL, "messages": _chat_messages(system, turns), "max_tokens": max_tokens, "stream": stream}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        if choice:
            body["tool_choice"] = choice if choice in ("none", "required") else {"type": "function", "function": {"name": choice}}
        if effort is not None:
            body["reasoning_effort"] = effort
        if stream:
            body["stream_options"] = {"include_usage": True}
        route, read = "/v1/chat/completions", _from_chat_stream if stream else _from_chat
    elif api.startswith("anthropic"):
        body = {"model": MODEL, "messages": _anthropic_messages(turns), "max_tokens": max_tokens, "stream": stream}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        if choice:
            body["tool_choice"] = {"type": {"none": "none", "required": "any"}.get(choice, "tool"),
                                   **({} if choice in ("none", "required") else {"name": choice})}
        if effort == "none":
            body["thinking"] = {"type": "disabled"}
        elif effort is not None:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}
        route, read = "/v1/messages", _from_anthropic_stream if stream else _from_anthropic
    else:
        body = {"model": MODEL, "input": _responses_input(turns), "max_output_tokens": max_tokens}
        if system:
            body["instructions"] = system
        if tools:
            body["tools"] = [dict(t, type="function") for t in tools]
        if choice:
            body["tool_choice"] = choice if choice in ("none", "required") else {"type": "function", "name": choice}
        if effort is not None:
            body["reasoning"] = {"effort": effort}
        route, read = "/v1/responses", _from_responses
    if temperature is not None:
        body["temperature"] = temperature
    a, start = _answer(), time.time()
    try:
        read(_post(route, body, stream), a)
    except Exception as e:  # a transport or server error is a result, not a crash of the run
        a["problems"].append(f"request failed: {type(e).__name__}: {str(e)[:500]}")
        a["finish"] = "error"
    a["secs"] = round(time.time() - start, 1)
    names, seen = {t["name"] for t in tools or ()}, set()
    for c in a["calls"]:
        c["args"] = None
        try:
            c["args"] = json.loads(c["raw"])
        except (TypeError, ValueError):
            if a["finish"] != "length":
                a["problems"].append(f"arguments of {c['name']} are not JSON: {str(c['raw'])[:120]!r}")
        if c["args"] is not None and not isinstance(c["args"], dict):
            a["problems"].append(f"arguments of {c['name']} are {type(c['args']).__name__}, not an object")
            c["args"] = None
        if not c["id"] or c["id"] in seen:
            a["problems"].append(f"call id {c['id']!r} missing or repeated")
        seen.add(c["id"])
        if c["name"] not in names:
            a["problems"].append(f"unknown tool {c['name']!r}")
    if a["calls"] and a["finish"] not in ("tool_calls", "length"):
        a["problems"].append(f"finish {a['finish']!r} with {len(a['calls'])} calls")
    if not a["calls"] and a["finish"] == "tool_calls":
        a["problems"].append("finish tool_calls without calls")
    for field in ("text", "reasoning"):
        found = CONTROL.findall(a[field])
        if found:
            a["problems"].append(f"control tokens in {field}: {sorted(set(found))}")
    return a
