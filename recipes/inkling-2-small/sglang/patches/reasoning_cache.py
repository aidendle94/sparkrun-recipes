"""Reasoning kept for clients that drop it: SPARK_REASONING_CACHE=1, or a file path (off by default).

Inkling's chat template renders an assistant turn of the history as thinking + answer only when the client sends the
turn's reasoning back (reasoning_content on /v1/chat/completions, thinking blocks on /v1/messages). Many OpenAI-style
clients send back only the answer text and the tool calls. The model then reads a conversation in which no earlier
question was ever thought about, and over a long context it answers an earlier question instead of the last one
(bench/stale_answer.py at 60K tokens: 9 of 24 later lookups not right without the reasoning, 0 of 24 with it).

With this on, the API server remembers the reasoning of each answer it gives and, when a later request brings that
answer back as history without reasoning, puts it back before the prompt is rendered. A turn that arrives with
reasoning is left alone. The model's arithmetic is untouched: the prompt becomes the one a client that keeps reasoning
would have sent.

A remembered answer is found by, in this order:
  chain  the conversation before it (role, text and tool calls of every non-system message) plus the answer itself
  call   the id of one of its tool calls (the server makes them unique)
  text   the answer alone, when its text and tool calls come to 64 characters or more
An answer that is not found stays as the client sent it (after a restart, or when the client rewrote its history).
The store lives in the API server process: least recently used out first, SPARK_REASONING_CACHE_MB (32) of text.
SPARK_REASONING_CACHE=<absolute path>: the hooks are installed but act only while that file exists in the container.
A failure in here is logged once and switches it off: it never fails a request.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict

LOG = logging.getLogger("spark_reasoning_cache")
MODE = os.environ.get("SPARK_REASONING_CACHE", "")
LIMIT = int(float(os.environ.get("SPARK_REASONING_CACHE_MB", "32")) * 2 ** 20)
MIN_TEXT = 64
_lock = threading.Lock()
_store = OrderedDict()    # key -> reasoning
_streams = OrderedDict()  # (response id, choice) -> [reasoning pieces, text pieces, tool calls] of a running stream
_local = threading.local()
_state = {"on": True, "size": 0, "lines": 0}
counts = {"remembered": 0, "chain": 0, "call": 0, "text": 0, "plain": 0, "missed": 0}


def active():
    return _state["on"] and (MODE == "1" or (MODE.startswith("/") and os.path.exists(MODE)))


def _text(content):
    """The text of a message, whether it came as a string or as parts."""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for part in content or ():
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            parts.append(part["text"] if isinstance(part.get("text"), str) else f"<{part.get('type')}>")
    return "\n".join(parts).strip()


def _field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _calls(tool_calls):
    """Names and arguments of an answer's tool calls, the same string for a response's calls (arguments as JSON text)
    and for the calls of a history message (arguments parsed)."""
    out = []
    for call in tool_calls or ():
        fn = _field(call, "function")
        args = _field(fn, "arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        out.append([_field(fn, "name"), args if args not in (None, "") else {}])
    return json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=False) if out else ""


def _update(h, *parts):
    for part in parts:
        data = part if isinstance(part, bytes) else part.encode("utf-8", "surrogatepass")
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)


def _keys(prefix, text, calls, tool_calls):
    h = hashlib.blake2b(digest_size=16)
    _update(h, prefix, text, calls)
    keys = [("chain", h.digest())]
    keys += [("call", str(_field(call, "id"))) for call in tool_calls or () if _field(call, "id")]
    if len(text) + len(calls) >= MIN_TEXT:
        h = hashlib.blake2b(digest_size=16)
        _update(h, text, calls)
        keys.append(("text", h.digest()))
    return keys


def fill(messages):
    """Put the remembered reasoning back into the assistant turns that came without any. Returns the digest of the
    conversation (the next answer is remembered under it) and how many turns were filled and not found."""
    run = hashlib.blake2b(digest_size=16)
    filled = missed = 0
    for message in messages:
        role = message.get("role") or ""
        if role == "system":
            continue
        text, calls = _text(message.get("content")), _calls(message.get("tool_calls"))
        if role == "assistant" and (text or calls) and not (message.get("reasoning_content") or message.get("reasoning")):
            keys = _keys(run.digest(), text, calls, message.get("tool_calls"))
            with _lock:
                found = next((key for key in keys if key in _store), None)
                if found is None:
                    counts["missed"] += 1
                    missed += 1
                else:
                    _store.move_to_end(found)
                    if _store[found]:
                        message["reasoning_content"] = _store[found]
                        counts[found[0]] += 1
                        filled += 1
                    else:  # an answer this server gave without reasoning: nothing to put back
                        counts["plain"] += 1
        _update(run, role, text, calls)
    return run.digest(), filled, missed


def remember(prefix, reasoning, text, tool_calls):
    """Keep the reasoning of an answer that followed the conversation with digest `prefix`."""
    text = (text or "").strip()
    calls = _calls(tool_calls)
    reasoning = reasoning or ""
    if prefix is None or not (text or calls):
        return
    with _lock:
        for key in _keys(prefix, text, calls, tool_calls):
            old = _store.pop(key, None)
            _state["size"] += len(reasoning) + 64 - (len(old) + 64 if old is not None else 0)
            _store[key] = reasoning
        while _state["size"] > LIMIT and _store:
            _state["size"] -= len(_store.popitem(last=False)[1]) + 64
        counts["remembered"] += 1


def _failed(e):
    _state["on"] = False
    LOG.warning("[reasoning-cache] DISABLED by %s: %s", type(e).__name__, str(e)[:400])


def install(mod):
    """mod: sglang.srt.entrypoints.openai.serving_inkling (chat completions; /v1/messages goes through it too)."""
    try:
        cls = mod.InklingChatEncodingAdapter
        encode, parse, stream, sse = (cls.encode_messages, cls.parse_chat_response, cls.stream_chat_chunks,
                                      mod.token_output_sse_chunks)
    except AttributeError as e:  # another SGLang: leave it as it is
        return _failed(e)

    def encode_messages(self, *, messages, request, **kwargs):
        if active():
            try:
                prefix, filled, missed = fill(messages)
                object.__setattr__(request, "_spark_reasoning_prefix", prefix)
                if (filled or missed) and (_state["lines"] < 40 or (missed and _state["lines"] < 400)):
                    _state["lines"] += 1
                    LOG.warning("[reasoning-cache] request: %d turn(s) got their reasoning back, %d not found | so far %s, "
                                "%d entries, %.1f MB", filled, missed, counts, len(_store), _state["size"] / 2 ** 20)
            except Exception as e:
                _failed(e)
        return encode(self, messages=messages, request=request, **kwargs)

    def parse_chat_response(self, *, request, **kwargs):
        result = parse(self, request=request, **kwargs)
        if active():
            try:
                remember(getattr(request, "_spark_reasoning_prefix", None), result[0], result[1], result[2])
            except Exception as e:
                _failed(e)
        return result

    def stream_chat_chunks(self, *, content, index, request, finish_reason_type, **kwargs):
        if not active():
            return stream(self, content=content, index=index, request=request, finish_reason_type=finish_reason_type,
                          **kwargs)
        key = (content["meta_info"]["id"], index)
        pieces = _streams.get(key)
        if pieces is None:
            pieces = _streams[key] = ([], [], [])
            while len(_streams) > 512:  # streams whose client went away never finish
                _streams.popitem(last=False)
        _local.pieces = pieces
        try:
            chunks = stream(self, content=content, index=index, request=request, finish_reason_type=finish_reason_type,
                            **kwargs)
        finally:
            _local.pieces = None
        if finish_reason_type is not None:
            _streams.pop(key, None)
            try:
                remember(getattr(request, "_spark_reasoning_prefix", None), "".join(pieces[0]), "".join(pieces[1]),
                         pieces[2])
            except Exception as e:
                _failed(e)
        return chunks

    def token_output_sse_chunks(*, reasoning, content, tool_calls, **kwargs):
        pieces = getattr(_local, "pieces", None)
        if pieces is not None:
            pieces[0].append(reasoning or "")
            pieces[1].append(content or "")
            pieces[2].extend(tool_calls or ())
        return sse(reasoning=reasoning, content=content, tool_calls=tool_calls, **kwargs)

    for wrapper, stock in ((encode_messages, encode), (parse_chat_response, parse), (stream_chat_chunks, stream),
                           (token_output_sse_chunks, sse)):
        wrapper.__wrapped__ = stock
    cls.encode_messages, cls.parse_chat_response, cls.stream_chat_chunks = encode_messages, parse_chat_response, stream_chat_chunks
    mod.token_output_sse_chunks = token_output_sse_chunks
    LOG.warning("[reasoning-cache] installed (%s, %d MB)", "on" if MODE == "1" else f"on while {MODE} exists", LIMIT >> 20)
