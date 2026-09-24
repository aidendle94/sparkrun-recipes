"""Reject requests the engine cannot serve before they reach the scheduler, where they would stop the server.

With `--enable-decoder-swa-bounded-replay` (which this deployment needs for long contexts) DeepSeek-V4's model forward
cannot return log-probabilities of prompt tokens. SGLang checks this only for the encoder variant of bounded replay; for
the decoder variant the request reaches the model, the forward raises inside the scheduler, and the scheduler's
failure stops the whole server (SIGQUIT to every process). One client call is enough, e.g. `/v1/completions` with
`echo: true` and `logprobs`, or a native `/generate` with `logprob_start_len` inside the prompt. This hook applies the
same rule SGLang applies to the encoder variant at request validation, so such a request gets an HTTP 400 and the
server keeps serving. Always on when the decoder variant is enabled.

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install(module) -> None:
    """Wrap sglang.srt.managers.tokenizer_manager.TokenizerManager._validate_one_request."""
    cls = module.TokenizerManager
    stock = cls._validate_one_request
    if getattr(stock, "_spark_guard", False):
        return

    def _validate_one_request(self, obj, input_ids):
        stock(self, obj, input_ids)
        get_exec = getattr(module, "get_exec", None)   # the accessor tokenizer_manager itself imported
        feats = get_exec().features if get_exec is not None else None
        if not getattr(feats, "enable_decoder_swa_bounded_replay", False):
            return
        start = getattr(obj, "logprob_start_len", None)
        n = len(input_ids) if input_ids is not None else 0
        if getattr(obj, "return_logprob", False) and start not in (None, -1) and start < n:
            raise ValueError("this server runs decoder SWA bounded replay and cannot return log-probabilities of "
                             "prompt tokens; request output logprobs only (no echo, or logprob_start_len equal to "
                             "the prompt length)")

    _validate_one_request._spark_guard = True
    cls._validate_one_request = _validate_one_request
    logger.info("request guard: prompt-logprob requests are rejected under decoder SWA bounded replay")
