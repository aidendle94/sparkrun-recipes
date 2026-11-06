"""Keep DeepSeek-V4.1's candidate-block indexer on the dense path on SM12x (GB10, RTX PRO 6000 Blackwell).

SGLang builds the candidate indexer (the layer that publishes the top candidate blocks the later sparse layers read)
with `v41_indexer.make_candidate_indexer`. On any device of compute capability 10 or newer it now picks the sparse-table
backend, which needs DeepGEMM's paged sparse MQA logits; SGLang enables those only on SM100
(`DEEPGEMM_PAGED_SPARSE_MQA_LOGITS` requires `DEEPGEMM_BLACKWELL`), so on SM12x the factory raises at backend init
("the candidate indexer needs DeepGEMM's paged sparse MQA logits"). The dense-blocks backend the same factory returns
below SM100 is what SM12x ran before that change: DeepGEMM's dense fp8/fp4 logits for prefill (unless
SGLANG_DSV41_TORCH_PREFILL_INDEXER is set) and torch block selection for decode.

This hook wraps the backend module's binding of the factory: on SM12x it returns the dense-blocks backend for both
prefill and decode, built with the factory's own arguments and its own DeepGEMM-prefill decision; on every other device
it calls the stock factory. The capability is read when the backend is built (inside the worker, CUDA already set up),
never at import.

Decode scoring. The dense-blocks backend scores a decode row in one of two ways. Its "paged" branch (SGLang enables it
on SM90 only) runs `fp4_index_logits_paged`, a Triton kernel with persistent workers that reads only each row's visible
compressed positions straight from the request table and, for the layers that consume the published candidates, only
inside the candidate blocks; the top-k then runs on SGLang's top-k v2 kernels. Every other device takes the dense-slot
branch: a [rows, width] slot gather and `fp4_index_logits_decode` over the whole graph width (524,555 positions at
compress ratio 1 for a 512K context), then torch top-k and block selection, again in full for every consumer. On GB10
at a 64K context the indexers of one decode step took 10.2-10.5 ms that way, against 0.9 ms on the previous base.
With SPARK_CANDIDATE_DECODE=paged (default) the backend takes SGLang's paged branch on SM12x as well: the wrapper below
is that branch of `scoring.decode_scores` (SGLang, Copyright SGLang Team, Apache License 2.0), entered on SM12x instead
of SM90 only; the kernels are portable (Triton, and the top-k v2 kernels SGLang's full top-k indexer already runs on
SM12x). The scores are the same function of the same inputs as the dense-slot kernel's ("the FP4 decoding and BF16
rounding are shared with the dense-slot path"). SPARK_CANDIDATE_DECODE=stock keeps SGLang's choice.

  SPARK_CANDIDATE_DENSE    1 (default): dense-blocks candidate indexer on SM12x; 0: SGLang's choice
  SPARK_CANDIDATE_DECODE   paged (default) | stock: decode scoring branch of the dense-blocks backend on SM12x

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def install(module) -> None:
    """sglang.srt.layers.attention.deepseek_v4_backend: wrap its make_candidate_indexer."""
    if os.environ.get("SPARK_CANDIDATE_DENSE", "1") != "1":
        return
    stock = module.make_candidate_indexer
    from sglang.srt.layers.attention.dsv4 import v41_indexer
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import DenseBlocksBackend

    def make_candidate_indexer(*, token_to_kv_pool, req_to_token, page_size, candidate_topk_blocks,
                               candidate_block_size, **extra):
        if extra or torch.cuda.get_device_capability()[0] != 12:
            return stock(token_to_kv_pool=token_to_kv_pool, req_to_token=req_to_token, page_size=page_size,
                         candidate_topk_blocks=candidate_topk_blocks, candidate_block_size=candidate_block_size, **extra)
        use_deep_gemm_prefill = v41_indexer._use_deep_gemm_prefill()
        dense = DenseBlocksBackend(
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
            use_deep_gemm_prefill=use_deep_gemm_prefill,
        )
        logger.info("candidate indexer on SM12x: dense blocks for prefill and decode (DeepGEMM prefill logits: %s)",
                    use_deep_gemm_prefill)
        return dense, dense

    module.make_candidate_indexer = make_candidate_indexer
    if os.environ.get("SPARK_CANDIDATE_DECODE", "paged") == "paged":
        _install_paged_decode()


def _install_paged_decode() -> None:
    """Wrap dense_blocks' binding of scoring.decode_scores: SGLang's paged branch on SM12x (see the module docstring)."""
    from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks, scoring

    stock = dense_blocks.decode_scores
    state = {"logged": False}

    def decode_scores(*, inputs, token_to_kv_pool, req_to_token, candidate_mask=None, compute_topk_plan=True, **extra):
        if extra or torch.cuda.get_device_capability()[0] != 12:
            return stock(inputs=inputs, token_to_kv_pool=token_to_kv_pool, req_to_token=req_to_token,
                         candidate_mask=candidate_mask, compute_topk_plan=compute_topk_plan, **extra)
        # From here: the `paged` branch of sglang.srt.layers.attention.dsv4.v41_indexer.scoring.decode_scores.
        pool = token_to_kv_pool
        ratio = inputs.compress_ratio
        indexer = inputs.indexer
        req, pos = inputs.req_rows, inputs.positions
        bs = req.shape[0]
        assert pos.shape[0] == bs, f"decode expects one token per request, {pos.shape=} {bs=}"
        if bs == 0:
            inputs.reset_outputs()
            return None
        lens = (pos + 1) // ratio
        metadata = inputs.paged_metadata
        lmax = min(metadata.max_compressed_seq_len, req_to_token.shape[1] // ratio)
        if lmax == 0:
            inputs.reset_outputs()
            return None
        q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
        weights = indexer.head_weights(inputs.x)
        table = pool.get_index_k_with_scale_buffer(inputs.layer_id)
        lens = lens.to(torch.int32)
        plan = scoring.plan_topk_v2(lens) if compute_topk_plan else None
        scores = scoring.fp4_index_logits_paged(q, weights, req, req_to_token, lens, table, table.shape[1] // 68,
                                                lmax, ratio, candidate_mask)
        if not state["logged"]:
            state["logged"] = True
            logger.info("candidate indexer decode on SM12x: paged scoring (ratio %d, capacity %d, candidate mask %s)",
                        ratio, lmax, candidate_mask is not None)
        return scoring.PagedDecodeScores(bs=bs, lmax=lmax, lens=lens, scores=scores, req=req, req_to_token=req_to_token,
                                         ratio=ratio, plan=plan, has_candidate_mask=candidate_mask is not None)

    dense_blocks.decode_scores = decode_scores
