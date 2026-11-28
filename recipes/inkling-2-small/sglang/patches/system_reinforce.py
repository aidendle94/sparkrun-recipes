"""Gated transient system-prompt reinforcement for Inkling-2-Small (SPARK_SYSTEM_REINFORCE=user|system; unset = off).

Why: 40 of Inkling-2-Small's 48 layers use 1,024-token sliding-window attention (SWA). Once prompt length passes
~1,024 tokens (which already happens on Turn 1 when tool schemas are declared, and throughout long multi-turn
agent sessions), those 40 layers cannot attend to the leading system prompt.

This hook is 100% general (no task-specific rules) and preserves model weights and intelligence:
  - Single-turn short prompts (assistant_turns == 0 and total prompt chars < MIN_CHARS, default 3600 ~= 1K tokens)
    are completely untouched (bit-for-bit identical to stock).
  - Requests without a system/developer message or ending in an assistant prefill are untouched.
  - Before extracting directives, reference/catalog sections that agent harnesses inject for on-demand lookup
    (<available_skills>, <skills>, <tools>, <docs>, <mcp_servers>, <cwd>, etc.) are stripped so optional skill
    or tool catalogs are never misclassified as mandatory per-turn rules.
  - Once context exceeds the SWA window (or on any multi-turn request with assistant_turns >= 1), a concise
    transient copy of the active system prompt's behavioral rules is appended right before tokenization.
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

# XML-tagged reference/catalog blocks used by coding harnesses (Pi, Claude Code, Codex, MCP, etc.)
# that describe optional tools, skills, docs, or environment metadata rather than unconditional turn rules.
_CATALOG_TAGS = (
    "available_skills",
    "skills",
    "skill",
    "available_tools",
    "tools",
    "tool_definitions",
    "tool_list",
    "functions",
    "mcp_servers",
    "mcp_tools",
    "available_mcp_tools",
    "docs",
    "documentation",
    "cwd",
    "environment",
    "env",
    "system_info",
    "user_information",
    "workspace_information",
    "slash_commands",
    "file_tree",
    "directory_structure",
)
_CATALOG_XML_RE = re.compile(
    r"<(?P<tag>" + "|".join(_CATALOG_TAGS) + r")(?:\s[^>]*)?>[\s\S]*?</(?P=tag)>",
    re.I,
)

# Markdown catalog headings (optional skill/tool listings)
_CATALOG_MD_HEAD_RE = re.compile(
    r"^\s*#{1,3}\s*(?:available\s+(?:skills|tools|commands)|skills\s+catalog|tool\s+(?:definitions|list)|documentation)\b",
    re.I,
)

# Plain-text "Available skills:" / "Available tools:" trailing catalog blocks
_CATALOG_PLAIN_RE = re.compile(
    r"(?:^|\n)Available\s+(?:skills|tools|slash\s+commands)\s*:\s*\n(?:[ \t]*[-*]\s+.*\n?)+",
    re.I,
)

# Top-level XML wrapper tags that often contain actual directives (unwrap into separate scorable sections)
_RULE_XML_RE = re.compile(
    r"<(?P<tag>rules|addendum|guidelines|instructions|policy|constraints|project_context|project_instructions)(?:\s[^>]*)?>([\s\S]*?)</(?P=tag)>",
    re.I,
)

_RULE_HEAD_RE = re.compile(
    r"^\s*(?:#{1,3}\s*.*?\b(policy|rules?|guidelines?|instructions?|mandatory|constraints?)\b|(?:rules|guidelines|instructions)\s*:)",
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


def _strip_catalogs(text: str) -> str:
    # Remove nested/top-level XML catalog blocks cleanly (two passes handle nested <skills><available_skills>...)
    cleaned = text
    for _ in range(2):
        prev = cleaned
        cleaned = _CATALOG_XML_RE.sub("\n", cleaned)
        if cleaned == prev:
            break
    cleaned = _CATALOG_PLAIN_RE.sub("\n", cleaned)
    return cleaned.strip()


def _split_oversized(sec: str, max_chars: int) -> list[str]:
    if len(sec) <= max_chars:
        return [sec]
    # Try splitting by blank lines first, then by lines if a single paragraph is huge
    paras = [p.strip() for p in re.split(r"\n\s*\n", sec) if p.strip()]
    if len(paras) > 1:
        out = []
        for p in paras:
            if len(p) <= max_chars:
                out.append(p)
            else:
                out.extend(_split_oversized(p, max_chars))
        return out
    # Single huge block: group lines up to max_chars
    lines = sec.splitlines()
    chunks, cur = [], []
    cur_len = 0
    for ln in lines:
        add = len(ln) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            chunks.append("\n".join(cur).strip())
            cur, cur_len = [ln], len(ln)
        elif len(ln) <= max_chars:
            cur.append(ln)
            cur_len += add
    if cur:
        chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c]


def _extract_candidate_sections(cleaned: str, max_chars: int) -> list[tuple[str, float]]:
    """Return [(section_text, structural_bonus)] in original order."""
    raw_units: list[tuple[str, float]] = []

    # 1. Split out XML directive sections (<rules>, <addendum>, <guidelines>, <project_instructions>, ...)
    pos = 0
    for m in _RULE_XML_RE.finditer(cleaned):
        before = cleaned[pos:m.start()].strip()
        if before:
            raw_units.append((before, 0.0))
        tag = m.group("tag").lower()
        body = m.group(2).strip()
        # Unwrap nested <project_instructions> inside <project_context>
        if tag == "project_context" and _RULE_XML_RE.search(body):
            for sub_text, sub_bonus in _extract_candidate_sections(body, max_chars):
                raw_units.append((sub_text, max(sub_bonus, 6.0)))
        elif body:
            bonus = 12.0 if tag in ("rules", "addendum", "guidelines", "instructions", "policy", "constraints") else 6.0
            raw_units.append((body, bonus))
        pos = m.end()
    tail = cleaned[pos:].strip()
    if tail:
        raw_units.append((tail, 0.0))

    # 2. Further split each unit on markdown headings (# / ## / ###), dropping catalog sections
    sections: list[tuple[str, float]] = []
    for unit_text, base_bonus in raw_units:
        md_parts = [p.strip() for p in re.split(r"\n(?=#{1,3}\s)", unit_text) if p.strip()]
        for part in md_parts:
            if _CATALOG_MD_HEAD_RE.match(part):
                continue
            head_bonus = 8.0 if _RULE_HEAD_RE.match(part) else 0.0
            for piece in _split_oversized(part, max_chars):
                sections.append((piece, base_bonus + head_bonus))
    return sections


def condense_system_prompt(sys_text: str, max_chars: int = MAX_CHARS) -> str:
    cleaned = _strip_catalogs(sys_text)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        # Unwrap harmless XML section tags when returning the whole cleaned prompt
        compact = re.sub(
            r"</?(?:rules|addendum|guidelines|instructions|policy|constraints|project_context|project_instructions)(?:\s[^>]*)?>",
            "",
            cleaned,
            flags=re.I,
        )
        return re.sub(r"\n{3,}", "\n\n", compact).strip()

    candidates = _extract_candidate_sections(cleaned, max_chars)
    if not candidates:
        return ""

    scored = []
    for idx, (sec, struct_bonus) in enumerate(candidates):
        hits = len(IMPERATIVE_RE.findall(sec))
        # Density-based imperative score so compact rule blocks beat long prose dumps
        density = (hits * 1000.0) / max(len(sec), 200)
        total = density + struct_bonus
        scored.append((total, idx, sec))

    chosen: set[int] = set()
    budget = max_chars
    for score, idx, sec in sorted(scored, key=lambda x: (-x[0], x[1])):
        if score <= 0.0:
            continue
        if len(sec) + 2 <= budget:
            chosen.add(idx)
            budget -= len(sec) + 2

    if not chosen:
        return ""

    return "\n\n".join(candidates[i][0] for i in sorted(chosen))


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
    if not snippet:
        return messages

    reminder = (
        "ACTIVE SYSTEM RULES — continue following the applicable rules below on this turn "
        "(prioritize these over habits from earlier turns):\n\n"
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
