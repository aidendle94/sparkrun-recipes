"""Gated transient system-prompt reinforcement for Inkling-2-Small (SPARK_SYSTEM_REINFORCE=user|system; unset = off).

Why: 40 of Inkling-2-Small's 48 layers use 1,024-token sliding-window attention (SWA). Once prompt length passes
~1,024 tokens (which already happens on Turn 1 when tool schemas are declared, and throughout long multi-turn
agent sessions), those 40 layers cannot attend to the leading system prompt.

This hook is 100% general (no task-specific rules) and preserves model weights and intelligence:
  - Single-turn short prompts (assistant_turns == 0 and total prompt chars < MIN_CHARS, default 3600 ~= 1K tokens)
    are completely untouched (bit-for-bit identical to stock).
  - Requests without a system/developer message or ending in an assistant prefill are untouched.
  - Once context exceeds the SWA window (or on any multi-turn request with assistant_turns >= 1), a concise
    transient copy of the active system prompt's imperative directives is appended right before tokenization.
  - Because the reminder is transient (never saved in client conversation history and applied after
    patches/reasoning_cache.py hashes/fills history), prior turns keep full SGLang radix prefix-cache hits.
  - Any unexpected error logs once and falls back cleanly to stock rendering without failing a request.
MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Mapping, Sequence

LOG = logging.getLogger("spark.system_reinforce")

_RAW_MODE = (os.environ.get("SPARK_SYSTEM_REINFORCE") or "").strip().lower()
MODE = "user" if _RAW_MODE in ("1", "true", "yes", "user") else ("system" if _RAW_MODE == "system" else "")
MAX_CHARS = int(os.environ.get("SPARK_SYSTEM_REINFORCE_MAX_CHARS") or "1200")
MIN_CHARS = int(os.environ.get("SPARK_SYSTEM_REINFORCE_MIN_CHARS") or "3600")

IMPERATIVE_RE = re.compile(
    r"\b(must|never|always|mandatory|policy|rules?|do not|don't|prefer|avoid|forbid|strictly|required|prohibit|guideline|important)\b",
    re.I,
)

_state = {"on": bool(MODE), "logged_install": False, "logged_first": False}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str) and b.strip():
                parts.append(b.strip())
            elif isinstance(b, dict) and b.get("type") in ("text", "input_text") and isinstance(b.get("text"), str):
                t = b["text"].strip()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _extract_system_text(messages: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for m in messages:
        if m.get("role") not in ("system", "developer"):
            break
        t = _content_text(m.get("content"))
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _estimate_chars(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None) -> int:
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, str):
                    total += len(b)
                elif isinstance(b, dict) and isinstance(b.get("text"), str):
                    total += len(b["text"])
        tc = m.get("tool_calls")
        if tc:
            total += len(str(tc))
        if total >= MIN_CHARS:
            return total
    if tools:
        try:
            total += len(json.dumps(tools, ensure_ascii=False))
        except Exception:
            total += len(str(tools))
    return total


def condense_system_prompt(sys_text: str, max_chars: int = MAX_CHARS) -> str:
    if len(sys_text) <= max_chars:
        return sys_text

    sections = re.split(r"\n(?=#{1,3}\s)", sys_text)
    if len(sections) == 1:
        sections = [p for p in sys_text.split("\n\n") if p.strip()]

    scored = []
    for idx, sec in enumerate(sections):
        hits = len(IMPERATIVE_RE.findall(sec))
        # Mild recency bonus for appended system-prompt sections
        bonus = 3 if idx == len(sections) - 1 else 0
        scored.append((hits + bonus, idx, sec.strip()))

    chosen = set()
    budget = max_chars
    for score, idx, sec in sorted(scored, key=lambda x: (-x[0], -x[1])):
        if score <= 0 and chosen:
            continue
        if len(sec) + 2 <= budget:
            chosen.add(idx)
            budget -= len(sec) + 2
        elif not chosen:
            return sys_text[-max_chars:]

    if not chosen:
        return sys_text[-max_chars:]
    return "\n\n".join(sections[i].strip() for i in sorted(chosen))


def reinforce_messages(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> Sequence[Mapping[str, Any]]:
    if not _state["on"] or not messages or len(messages) < 2:
        return messages

    last = messages[-1]
    if last.get("role") == "assistant":
        return messages

    sys_text = _extract_system_text(messages)
    if not sys_text:
        return messages

    assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
    if assistant_turns == 0 and _estimate_chars(messages, tools) < MIN_CHARS:
        return messages

    # Guard against double-injection if caller already attached a reminder
    if "<system-reminder>" in _content_text(last.get("content")):
        return messages

    snippet = condense_system_prompt(sys_text, MAX_CHARS)
    reminder = (
        "ACTIVE SYSTEM INSTRUCTIONS — follow these strictly on this turn "
        "(prioritize these rules over habits from earlier turns):\n\n"
        f"{snippet}"
    )

    out = [dict(m) for m in messages]
    if MODE == "system":
        out.append({"role": "system", "content": reminder})
    else:
        block = f"<system-reminder>\n{reminder}\n</system-reminder>"
        tail = out[-1]
        if tail.get("role") == "user":
            c = tail.get("content")
            if isinstance(c, str):
                tail["content"] = c + "\n\n" + block
            elif isinstance(c, list):
                tail["content"] = list(c) + [{"type": "text", "text": "\n\n" + block}]
            else:
                out.append({"role": "user", "content": block})
        else:
            out.append({"role": "user", "content": block})

    if not _state["logged_first"]:
        _state["logged_first"] = True
        LOG.warning(
            "[system-reinforce] active (mode=%s, max_chars=%d, min_chars=%d): first transient reinforcement applied "
            "(assistant_turns=%d, snippet=%d chars)",
            MODE, MAX_CHARS, MIN_CHARS, assistant_turns, len(snippet),
        )
    return out


def _wrap_render(stock):
    if getattr(stock, "_spark_sys_reinforce", False):
        return stock

    def render_inkling_messages(messages, *, tools=None, reasoning_effort=None, spec="inkling"):
        try:
            messages = reinforce_messages(messages, tools=tools)
        except Exception as exc:
            LOG.warning("[system-reinforce] disabled after error: %s", exc)
            _state["on"] = False
        return stock(messages, tools=tools, reasoning_effort=reasoning_effort, spec=spec)

    render_inkling_messages.__wrapped__ = stock
    render_inkling_messages._spark_sys_reinforce = True
    return render_inkling_messages


def install_renderer(mod) -> None:
    if not MODE:
        return
    mod.render_inkling_messages = _wrap_render(mod.render_inkling_messages)
    if not _state["logged_install"]:
        _state["logged_install"] = True
        LOG.warning("[system-reinforce] installed (mode=%s, max_chars=%d, min_chars=%d)", MODE, MAX_CHARS, MIN_CHARS)


def install_serving(mod) -> None:
    if not MODE:
        return
    if hasattr(mod, "render_inkling_messages"):
        mod.render_inkling_messages = _wrap_render(mod.render_inkling_messages)
