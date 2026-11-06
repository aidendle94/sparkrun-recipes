"""Verify only the drafts that are likely to survive: SGLang's compact DSpark verify, made to run on DeepSeek-V4.1 TP4.

Why. A decode step verifies DSpark's block of 5 drafts plus the anchor, 6 rows per request, and the routed MoE is the
largest part of the step (23-25 ms of ~52 ms at one stream on GB10). It is bound by memory bandwidth, so its cost grows
with the experts the rows route to: one MoE layer of the production layout (CUTLASS W4A8, 96 whole experts per rank,
slowest rank) measured 397 / 580 / 758 / 945 / 1,041 / 1,189 us at 1-6 tokens (window 40). On prose ~2 drafts survive,
so most of the 6-row verify is spent on rows that are rejected.

What SGLang already has. With SGLANG_RAGGED_VERIFY_MODE=compact its DSpark planner gives every request a verify length
(the anchor plus a prefix of its drafts) from the draft's confidence head: a host budget (lag-2 relayed confidence and a
step-cost table, chosen to maximise expected committed tokens per second) sets the batch's verify tokens and so the CUDA
graph to replay, and an on-device top-k over survival probabilities spends it per request. Only the scheduled rows are
packed into the forward, and acceptance is capped at each request's length, so outputs keep the target distribution.
The DeepSeek-V4.1-Flash-1017 checkpoint ships the confidence head (mtp.2.confidence_head.proj).

What this overlay adds (SPARK_VERIFY_CAP=1, with SGLANG_RAGGED_VERIFY_MODE=compact and an SPS table):
  * the ragged Engram hasher and token -> request map of lookup_wide.py (stock assumes 6 rows per request and fails
    graph capture on V4.1 in compact mode);
  * token-keyed verify graphs at every count up to 12 and at 6b - 3 and 6b beyond (cap_buckets), via lookup_wide's
    capture rewrite: stock buckets are 6b only, so a single request always replayed the 6-row graph and nothing was saved;
  * the draft confidence broadcast from TP rank 0 right after each proposal (install_proposer): the host budget picks
    the graph every rank replays, and is computed per rank from relayed confidence; ranks whose confidence differed in
    the last bits could pick different graphs and deadlock in the next collective. SGLang already broadcasts the
    per-request lengths (SpecTpSyncSite.DSPARK_PLAN), not the confidence the budget is computed from.

  SPARK_VERIFY_CAP          1: enable (default 0)
  SPARK_VERIFY_CAP_BUCKETS  "fine" (default): 1..12, then 6b - 3 and 6b; "stock": SGLang's 6b; or a comma list

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SPARK_VERIFY_CAP", "0") == "1"


def cap_buckets(capture_bs, req_width: int, limit: int) -> list[int]:
    """Verify graph token counts for the compact planner (each <= limit, the shared logits buffer's rows)."""
    spec = os.environ.get("SPARK_VERIFY_CAP_BUCKETS", "fine").strip()
    stock = {b * req_width for b in capture_bs}
    if spec == "stock":
        out = stock
    elif spec == "fine":
        out = set(range(1, 2 * req_width + 1)) | stock | {t - req_width // 2 for t in stock if t > 2 * req_width}
    else:
        out = {int(x) for x in spec.split(",") if x.strip()}
    out = sorted(t for t in out if 1 <= t <= limit)
    logger.info("verify cap: %d compact verify graphs at %s tokens", len(out), out)
    return out


def install_proposer(module) -> None:
    """speculative.dspark_components.dspark_draft: broadcast each proposal's confidence from TP rank 0, in place."""
    if not ENABLED:
        return
    mode = os.environ.get("SGLANG_RAGGED_VERIFY_MODE", "static")
    if mode != "compact":
        raise RuntimeError(f"SPARK_VERIFY_CAP=1 needs SGLANG_RAGGED_VERIFY_MODE=compact (got {mode!r})")
    cls = module.DraftBlockProposer
    stock = cls.propose
    state = {"logged": False}

    def propose(self, *args, **kwargs):
        proposal = stock(self, *args, **kwargs)
        conf = getattr(proposal, "confidence", None)
        group = getattr(getattr(self, "_tp_sync", None), "_tp_group", None)
        if conf is not None and group is not None and group.world_size > 1 and not torch.cuda.is_current_stream_capturing():
            if not conf.is_contiguous():
                raise RuntimeError("verify cap: proposal confidence is not contiguous; cannot broadcast it in place")
            group.broadcast(conf, src=0)
            if not state["logged"]:
                state["logged"] = True
                logger.info("verify cap: draft confidence %s broadcast from TP rank 0 every step", tuple(conf.shape))
        return proposal

    propose.__wrapped__ = stock
    cls.propose = propose
    logger.info("verify cap on: compact DSpark verify with per-request lengths from the confidence head")
