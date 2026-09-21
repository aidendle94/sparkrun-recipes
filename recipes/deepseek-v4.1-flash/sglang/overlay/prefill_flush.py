"""Return the caching allocator's reserve after long prefill chunks.

The incident behind this hook: a 128K-token prompt took all four Sparks to zero MemAvailable within a
minute of prefill and the OOM guard killed two ranks. The cause is on the allocator side, not the
model's: on a Spark the GPU pool is the host's 128 GB, and PyTorch's caching allocator never returns
a freed block to the driver on its own. Two facts make that grow without bound during a DeepSeek-V4
prefill:
  - the indexer scores every chunk against the entire prefix, so its scratch tensors get larger with
    each chunk;
  - a request for a larger block is not served by the smaller cached ones, it is a fresh allocation,
    so the reserve adds up over chunks instead of being reused.
The usual remedy, `expandable_segments`, is off on this stack because the sparse prefill kernels
break under it. The hook does the next-best thing: after every prefill forward whose longest
sequence has reached SPARK_PREFILL_FLUSH_TOKENS it calls `torch.cuda.empty_cache()`, which is one
device sync per long chunk; short prompts never pay it.

The wrapper only acts on plain prefill batches (EXTEND, MIXED, SPLIT_PREFILL); a DSpark verify
step is an "extend" too but runs once per decode token and must stay untouched. It never acts
while a CUDA graph is being captured (cudaFree inside a capture invalidates it) and reads the
longest length from the batch's CPU copies so that no device sync is added on the common path.
Anything that goes wrong inside the wrapper is logged and swallowed: the forward has already
produced its output and a failed flush must not turn into a failed request.

Environment:
  SPARK_PREFILL_FLUSH_TOKENS   longest-sequence threshold in tokens (default 8192; 0 disables)

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

DEFAULT_TOKENS = 8192


def threshold_tokens() -> int:
    """SPARK_PREFILL_FLUSH_TOKENS as an int; an unparsable value falls back to the default."""
    raw = os.environ.get("SPARK_PREFILL_FLUSH_TOKENS", "").strip()
    if not raw:
        return DEFAULT_TOKENS
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("SPARK_PREFILL_FLUSH_TOKENS=%r is not an integer; using %d", raw, DEFAULT_TOKENS)
        return DEFAULT_TOKENS


def is_prefill(forward_mode) -> bool:
    """Plain prefill: extend-like but not the speculative verify step."""
    return bool(forward_mode.is_extend()) and not bool(forward_mode.is_target_verify())


def longest_sequence(forward_batch) -> int:
    """Longest (prefix + new) length in the batch, from CPU-side copies when they exist."""
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.device.type == "cpu" and seq_lens_cpu.numel():
        return int(seq_lens_cpu.max())
    extend = getattr(forward_batch, "extend_seq_lens_cpu", None)
    prefix = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if extend and prefix and len(extend) == len(prefix):
        return max(int(e) + int(p) for e, p in zip(extend, prefix))
    total = getattr(forward_batch, "seq_lens_sum", None)
    if total is not None and getattr(forward_batch, "batch_size", 0) == 1:
        return int(total)
    return int(forward_batch.seq_lens.max().item())  # last resort: one device sync


def flush_after(forward_batch, tokens: int) -> bool:
    """Empty the allocator cache when the batch qualifies; returns whether it did."""
    if not is_prefill(forward_batch.forward_mode):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    longest = longest_sequence(forward_batch)
    if longest < tokens:
        return False
    torch.cuda.empty_cache()
    logger.debug("prefill flush: empty_cache after chunk with longest sequence %d", longest)
    return True


def install(module) -> None:
    """Wrap sglang.srt.model_executor.model_runner.ModelRunner.forward."""
    tokens = threshold_tokens()
    if tokens <= 0:
        logger.info("prefill flush disabled (SPARK_PREFILL_FLUSH_TOKENS=0)")
        return
    cls = module.ModelRunner
    stock = cls.forward
    state = {"warned": False}

    def forward(self, forward_batch, *args, **kwargs):
        out = stock(self, forward_batch, *args, **kwargs)
        try:
            flush_after(forward_batch, tokens)
        except Exception as exc:  # the forward succeeded; a flush problem must not fail the step
            if not state["warned"]:
                state["warned"] = True
                logger.warning("prefill flush skipped: %r (further failures logged at DEBUG)", exc, exc_info=True)
            else:
                logger.debug("prefill flush skipped: %r", exc)
        return out

    forward.__wrapped__ = stock
    forward.__doc__ = stock.__doc__
    cls.forward = forward
    logger.info("prefill flush armed: empty_cache after prefill chunks with a sequence >= %d tokens", tokens)
