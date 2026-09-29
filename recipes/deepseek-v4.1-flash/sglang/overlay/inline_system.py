"""Keep the prompt prefix stable for clients that send system notes in the middle of a conversation (Claude Code).

SGLang's Anthropic endpoint (/v1/messages) hoists every `role: system` message found inside `messages` into the one
system prompt at the top when the chat template has no mid-conversation system turn, which is DeepSeek-V4.1's case.
Claude Code adds such a note on every turn (`<total_tokens>N tokens left</total_tokens>`), so each request's system
prompt grew by one line at about token 4,400 and every later token moved: the prefix cache matched ~4K of a 150K+
token prompt and each turn re-prefilled the rest, about a minute before the first token (2026-09-29).

This hook lets the stock converter emit those messages in place and then folds each one into the next user message
(prepended as text), or into a trailing user message when no user message follows; a tool call and its results are
never split. The system prompt at the top holds only the request's `system` field, and the rendered prompt of the
next turn extends this one instead of rewriting it.

  SPARK_INLINE_SYSTEM_IN_PLACE   1 (default): fold inline system notes in place; 0: stock hoisting

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _text(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return "\n".join(getattr(p, "text", "") or "" for p in content or () if getattr(p, "type", None) == "text")


def fold_inline_system(messages, text_part_cls, user_cls):
    """Messages after the first whose role is system are folded into the next user message (or a trailing one)."""
    out, pending = [], []
    for i, m in enumerate(messages):
        if m.role == "system" and i > 0:
            t = _text(m).strip()
            if t:
                pending.append(t)
            continue
        if m.role == "user" and pending:
            note = "\n".join(pending)
            if isinstance(m.content, str):
                m.content = note + "\n\n" + m.content
            else:
                m.content = [text_part_cls(type="text", text=note)] + list(m.content)
            pending = []
        out.append(m)
    if pending:
        out.append(user_cls(role="user", content="\n".join(pending)))
    return out


def install(module) -> None:
    """entrypoints.anthropic.serving: AnthropicServing._convert_to_chat_completion_request."""
    if os.environ.get("SPARK_INLINE_SYSTEM_IN_PLACE", "1") != "1":
        return
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionMessageContentTextPart,
        ChatCompletionMessageUserParam,
    )
    cls = module.AnthropicServing
    stock = cls._convert_to_chat_completion_request

    def _convert_to_chat_completion_request(self, anthropic_request):
        if not self._merge_inline_system or not any(
                getattr(m, "role", None) == "system" for m in anthropic_request.messages or ()):
            return stock(self, anthropic_request)
        self._merge_inline_system = False          # the converter runs synchronously on the event loop
        try:
            request = stock(self, anthropic_request)
        finally:
            self._merge_inline_system = True
        request.messages = fold_inline_system(request.messages, ChatCompletionMessageContentTextPart,
                                              ChatCompletionMessageUserParam)
        return request

    cls._convert_to_chat_completion_request = _convert_to_chat_completion_request
    logger.info("inline system notes of /v1/messages requests are folded in place (prefix-cache friendly)")
