"""Build the DeepGEMM schedule for DeepSeek-V4.1's ratio-1 and ratio-2 indexers on SM120.

Why: `PagedIndexerMetadata.__post_init__` (sglang.srt.layers.attention.dsv4.metadata) decides
whether to run DeepGEMM's `get_paged_mqa_logits_metadata` planner for an indexer source. On
SM120/SM121 SGLang switches the classic compress-ratio-4 indexer to a torch implementation
(SGLANG_FP8_PAGED_MQA_LOGITS_TORCH is set for consumer Blackwell) and, unless the metadata is
constructed with `force_deep_gemm_metadata=True`, the planner is skipped and `deep_gemm_metadata`
is None. That is right for the ratio-4 source but not for V4.1's ratio-1 and ratio-2 sources: their
logits go through DeepGEMM's `fp8_fp4_paged_mqa_logits`, which reads the plan, and without one the
CUDA-graph capture of the decode step fails with a `schedule_meta` mismatch. Stock SGLang only
forces the plan when the optional FP4 c4 indexer is enabled, which this deployment does not use.

The hook wraps `__post_init__` so that, on SM120 only, a metadata object for compress ratio 1 or 2
sets `force_deep_gemm_metadata` before the stock code runs; the stock code then honours the field
and plans as it would for the FP4 path (the flag also selects DeepGEMM's own planner over the JIT
one, whose split_kv encoding the SM120 FP4 kernel does not accept). Ratio-4 metadata and every
other GPU are untouched. The captured and per-step copies of the metadata are both constructed
this way, so `copy_`'s equality check on the field still holds.

Environment: none (the hook is on whenever the overlay is installed on SM120).

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FORCED_RATIOS = (1, 2)


def needs_plan(compress_ratio: int) -> bool:
    return compress_ratio in FORCED_RATIOS


def install(module) -> None:
    """Wrap sglang.srt.layers.attention.dsv4.metadata.PagedIndexerMetadata.__post_init__ on SM120."""
    if not getattr(module, "_IS_SM120", False):
        logger.info("indexer schedule: not SM120, stock planner rules kept")
        return
    cls = module.PagedIndexerMetadata
    stock = cls.__post_init__

    def __post_init__(self):
        if needs_plan(self.compress_ratio) and not self.force_deep_gemm_metadata:
            self.force_deep_gemm_metadata = True
        stock(self)

    __post_init__.__wrapped__ = stock
    __post_init__.__doc__ = stock.__doc__
    cls.__post_init__ = __post_init__
    logger.info("indexer schedule: DeepGEMM plan forced for compress ratios %s on SM120", FORCED_RATIOS)
