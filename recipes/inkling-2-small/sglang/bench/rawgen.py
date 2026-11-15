"""Raw-prompt requests on the server's native API, shared by the recital and path tools.

A prompt is built as token ids (the chat template's rendering of one user message at the default effort), so that a
generation can be continued from any point of its own output: `generate(prompt + output[:t], 1)` reuses the cached
prefix and returns the next-token distribution at position t as the PREFILL path computes it. Only output logprobs are
asked for (safe on a serving fleet; input logprobs over a long span are not, see the README).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json

import chatload as cl

HEAD = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>"
END = "<|end_message|>"
ANSWER = "<|content_text|>"  # in a generation's text, the answer follows the last one of these


def tokens(text):
    status, data = cl.post("tokenize", {"prompt": text})
    assert status == 200, f"tokenize: HTTP {status} {data[:300]}"
    return json.loads(data)["tokens"]


def prompt_ids(content):
    """Token ids of one user message with `content`, up to where the model's turn begins."""
    return tokens(HEAD + content + END)


def generate(ids, n=1, top=10, temperature=0):
    """n new tokens after `ids`. Returns text, cached (prefix tokens reused), finish ('stop' or 'length'), and per new
    token `chosen` [logprob, id, text] and `top` (the `top` most likely, same triples)."""
    status, data = cl.post("generate", {"input_ids": ids, "return_logprob": True, "top_logprobs_num": top,
                                        "return_text_in_logprobs": True,
                                        "sampling_params": {"temperature": temperature, "max_new_tokens": n,
                                                            "skip_special_tokens": False}})
    assert status == 200, f"generate: HTTP {status} {data[:300]}"
    d = json.loads(data)
    m = d["meta_info"]
    return {"text": d["text"], "cached": m.get("cached_tokens"), "finish": (m.get("finish_reason") or {}).get("type"),
            "chosen": m["output_token_logprobs"], "top": m.get("output_top_logprobs") or []}


def show(top, k=4):
    return "  ".join(f"{t[2]!r} {t[0]:.3f}" for t in top[:k])
