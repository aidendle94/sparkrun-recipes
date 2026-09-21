"""Split the V4.1 ratio-2 KV source to 64-token pages for the SM12x sparse-MLA prefill kernel.

SGLang's SM120 sparse prefill path re-pages the primary KV source from the pool's 256-token pages to
the 64-token pages FlashInfer's `_sparse_mla_sm120_paged_attention` accepts, but hands the second
("extra", compress-ratio 2, 128-token pages) source through untouched, and the kernel rejects the
configuration (`extra_page_block_size=128`). This hook re-pages the extra source the same way into
its own persistent buffers so the two conversions never share scratch. Token-level indices are
unchanged by the split (page p, offset o -> page 2p + o // 64, offset o % 64 keeps the flat index).

`_split_to_64` follows SGLang's own `_split_kv_pages_to_64` (the same page-mark and page-split Triton
kernels, called from the module) with the scratch buffers made persistent and per source — SGLang,
Copyright SGLang Team, Apache License 2.0 (see NOTICE).

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging

import torch
import triton

logger = logging.getLogger(__name__)


def _split_to_64(m, kv_u8: torch.Tensor, src_pbs: int, touched: torch.Tensor | None) -> torch.Tensor:
    pbs = m._PBS_DST
    assert src_pbs % pbs == 0 and src_pbs > pbs, (src_pbs, pbs)
    from sglang.srt.runtime_context import get_resources

    n_src = kv_u8.shape[0]
    ratio = src_pbs // pbs
    n_dst = n_src * ratio
    dev = kv_u8.device
    buffers = get_resources().buffers
    key = f"spark_extra_split:{dev}"
    buf = buffers.get(key)
    if buf is None or buf.shape[0] < n_dst:
        with torch.inference_mode(False):
            buf = torch.empty(n_dst, m._BYTES_PER_DST_PAGE_PADDED, dtype=torch.uint8, device=dev)
        buffers[key] = buf
    out = buf[:n_dst]
    src_2d = kv_u8
    if src_2d.ndim == 4:
        stride0 = src_2d.stride(0)
        src_2d = torch.as_strided(src_2d, (n_src, stride0), (stride0, 1))
    else:
        stride0 = src_2d.stride(0)
    use_mask = touched is not None and touched.numel() > 0
    mask_ptr = src_2d
    if use_mask:
        mkey = f"spark_extra_mask:{dev}"
        mbuf = buffers.get(mkey)
        if mbuf is None or mbuf.shape[0] < n_src:
            with torch.inference_mode(False):
                mbuf = torch.empty(n_src, dtype=torch.int8, device=dev)
            buffers[mkey] = mbuf
        mask = mbuf[:n_src]
        mask.zero_()
        idx = touched.reshape(-1).contiguous()
        if idx.dtype != torch.int32:
            idx = idx.to(torch.int32)
        m._page_mark_kernel[(triton.cdiv(idx.numel(), 1024),)](idx, mask, idx.numel(), src_pbs, 1024)
        mask_ptr = mask
    m._page_split_kernel[(n_dst,)](
        src_2d, out, n_src, stride0, m._BYTES_PER_DST_PAGE_PADDED,
        pbs * m._NOPE_ROPE_STRIDE, pbs * m._SCALE_STRIDE,
        src_pbs * m._NOPE_ROPE_STRIDE, pbs * m._NOPE_ROPE_STRIDE,
        ratio, 1024, mask_ptr, use_mask,
    )
    bpt = m._NOPE_ROPE_STRIDE + m._SCALE_STRIDE
    return out.as_strided((n_dst, pbs, 1, bpt), (m._BYTES_PER_DST_PAGE_PADDED, bpt, bpt, 1))


def install(module) -> None:
    stock = module._flash_mla_sm120_prefill

    def prefill(q, k_cache, indices, topk_length, attn_sink, head_dim_v, softmax_scale,
                extra_k_cache, extra_indices, extra_topk_length):
        if extra_k_cache is not None:
            pbs = extra_k_cache.shape[1] if extra_k_cache.ndim >= 3 else module._PBS_SRC
            if pbs != module._PBS_DST:
                kv = extra_k_cache.view(torch.uint8) if extra_k_cache.dtype != torch.uint8 else extra_k_cache
                idx = extra_indices.squeeze(1) if extra_indices is not None and extra_indices.dim() == 3 else extra_indices
                extra_k_cache = _split_to_64(module, kv, pbs, idx)
        return stock(q, k_cache, indices, topk_length, attn_sink, head_dim_v, softmax_scale,
                     extra_k_cache, extra_indices, extra_topk_length)

    module._flash_mla_sm120_prefill = prefill
    logger.info("SM12x: ratio-2 KV source re-paged to 64-token pages for the sparse prefill kernel")
