"""Shadow check of GB10's attention kernels: SPARK_ATTN_CHECK=1, diagnosis windows only (off by default).

Every prefill attention call (SGLang's FA4 SM12x forward, reached through sitecustomize.fa4_sm12x_args) and a sample of
the decode / verify calls (the Triton partials kernel behind sitecustomize.triton_decode; decode needs
--disable-cuda-graph, a captured graph does not run Python) is recomputed for a few query rows in plain torch, from the
same inputs the kernel got: the query, the paged K/V pool, the page table, the sequence lengths and the relative bias.

Reference for the query at position p of a sequence, head h (kv head h // group):
    score_j = q . k_j * scale + (bias[row, h, p - j] if p - j < extent else 0)     for keys j <= p,
                                                                                   j >= p - window on local layers
    out = softmax(score) @ v,  lse = log(sum(exp(score)))
(scores in float64, K and V read as float32 from the pool). The kernel's output and LSE are compared with it:
    out_rel = max |out - ref| / max |ref|   over a head's vector, worst head of the row
    lse_abs = max |lse - ref|               over heads
A BF16 output rounds to about 0.004 relative, so healthy lines stay near that; LSE is float32 on both sides.
One log line per prefill geometry (queries, keys) with the kinds of layer side by side; decode rows are collected per
geometry (block length, keys rounded down to 4096, batch) and printed every 20 s of decoding. A row above
SPARK_ATTN_CHECK_TOL (0.05) gets a line of its own with the per-head errors. A failure inside the check is logged once and turns the check off: it never
fails a forward. Rows with more than SPARK_ATTN_CHECK_MAXKEYS (150000) visible keys are skipped (reference memory).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import os
import random
import time

import torch

LOG = logging.getLogger("spark_attn_check")
TOL = float(os.environ.get("SPARK_ATTN_CHECK_TOL", "0.05"))
MAXKEYS = int(os.environ.get("SPARK_ATTN_CHECK_MAXKEYS", "150000"))
DECODE_EVERY = int(os.environ.get("SPARK_ATTN_CHECK_DECODE_EVERY", "16"))
_state = {"on": True, "details": 0, "prefill": None, "decode": {}, "decode_at": time.time(), "shapes": 0}
_rng = random.Random(7)


def _rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _keys(cache, table_row, lo, hi):
    """Pool rows of positions lo..hi-1 of one sequence as float32 (n, heads, dim); cache (pages, page_size, heads, dim)."""
    page_size = cache.shape[1]
    if page_size == 1:
        slots = table_row[lo:hi].to(torch.int64)
    else:
        pos = torch.arange(lo, hi, device=cache.device)
        slots = table_row[pos // page_size].to(torch.int64) * page_size + pos % page_size
    if str(cache.dtype) == "torch.float8_e4m3fn":
        import fp8_kv
        rows = fp8_kv.gather_bf16(cache, slots.to(torch.int32).contiguous())
        return rows.view(hi - lo, cache.shape[2], cache.shape[3]).float()
    return cache.reshape(-1, cache.shape[2], cache.shape[3])[slots].float()


def reference(q_row, k, v, bias_row, scale):
    """q_row (heads_q, d); k, v (n, heads_k, d) float32 in position order, the last one the query's own position;
    bias_row (heads_q, extent) by distance, or None. Returns out (heads_q, d) float32 and lse (heads_q,) float64."""
    n, heads_k, _ = k.shape
    heads_q = q_row.shape[0]
    group = heads_q // heads_k
    s = torch.einsum("nkd,kgd->nkg", k, q_row.float().view(heads_k, group, -1)).double() * scale
    if bias_row is not None:
        m = min(n, bias_row.shape[-1])
        s[n - m:] += bias_row[:, :m].double().t().flip(0).reshape(m, heads_k, group)
    lse = torch.logsumexp(s, 0)
    out = torch.einsum("nkg,nkd->kgd", torch.softmax(s, 0).float(), v)
    return out.reshape(heads_q, -1), lse.reshape(heads_q)


def _compare(out_row, lse_row, q_row, k_cache, v_cache, table_row, pos, bias_row, scale, window_left):
    lo = max(0, pos - window_left) if window_left is not None and window_left >= 0 else 0
    if pos + 1 - lo > MAXKEYS:
        return None
    ref_out, ref_lse = reference(q_row, _keys(k_cache, table_row, lo, pos + 1), _keys(v_cache, table_row, lo, pos + 1),
                                 bias_row, scale)
    per_head = ((out_row.float() - ref_out).abs().amax(-1) / ref_out.abs().amax(-1).clamp_min(1e-12))
    lse_err = (lse_row.double() - ref_lse).abs() if lse_row is not None else torch.zeros_like(ref_lse)
    return per_head, lse_err, ref_lse


def _note(kind, geometry, layer_kind, pos, keys, row, res):
    """Add one compared row to the line of (kind, geometry). Prefill: a new geometry prints the previous line."""
    if kind == "prefill":
        cur = _state["prefill"]
        if cur is None or cur["geometry"] != geometry:
            flush("prefill")
            cur = _state["prefill"] = {"geometry": geometry, "stats": {}}
        stats = cur["stats"]
    else:
        stats = _state["decode"].setdefault(geometry, {})
    per_head, lse_err, ref_lse = res
    o, l = float(per_head.max()), float(lse_err.max())
    st = stats.setdefault(layer_kind, [0, 0.0, 0.0, 0.0, 0.0])
    st[0] += 1
    st[1], st[2], st[3], st[4] = max(st[1], o), st[2] + o, max(st[3], l), st[4] + l
    if (o > TOL or l > TOL or o != o or l != l) and _state["details"] < 400:
        _state["details"] += 1
        LOG.warning("[attn-check] rank=%d ROW %s %s %s row=%d pos=%d keys=%d out_rel=%.4f lse_abs=%.4f | out_rel per head %s "
                    "| lse err per head %s | ref lse %s", _rank(), kind, geometry, layer_kind, row, pos, keys, o, l,
                    [round(x, 3) for x in per_head.tolist()], [round(x, 3) for x in lse_err.tolist()],
                    [round(x, 2) for x in ref_lse.tolist()])


def _line(kind, geometry, stats):
    body = " | ".join(f"{name}: n={s[0]} out_rel max {s[1]:.4f} mean {s[2] / s[0]:.4f}, lse_abs max {s[3]:.4f} "
                      f"mean {s[4] / s[0]:.4f}" for name, s in sorted(stats.items()))
    LOG.warning("[attn-check] rank=%d %s %s %s", _rank(), kind, geometry, body)


def flush(kind):
    if kind == "prefill":
        cur, _state["prefill"] = _state["prefill"], None
        if cur and cur["stats"]:
            _line("prefill", cur["geometry"], cur["stats"])
    else:
        pending, _state["decode"], _state["decode_at"] = _state["decode"], {}, time.time()
        for geometry, stats in sorted(pending.items()):
            _line("decode", geometry, stats)


def _guarded(fn):
    def run(*args, **kwargs):
        if not _state["on"]:
            return
        try:
            with torch.no_grad():
                fn(*args, **kwargs)
        except Exception as e:  # the check must never fail a forward
            _state["on"] = False
            LOG.warning("[attn-check] rank=%d DISABLED by %s: %s", _rank(), type(e).__name__, str(e)[:400])
    return run


@_guarded
def prefill(q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, rel_bias, scale, window, result):
    """After a flash_attn_with_kvcache call: q (tokens, heads_q, d), the queries being the last positions of each
    sequence; result (out, lse) with lse (heads_q, tokens)."""
    if not isinstance(result, tuple) or rel_bias is None or q.dim() != 3:
        return
    out, lse = result[0], result[1]
    tokens, heads_q, dim = q.shape
    out = out.reshape(tokens, heads_q, -1)
    lse = lse.t() if lse.shape == (heads_q, tokens) else lse.reshape(tokens, heads_q)
    scale = dim ** -0.5 if scale is None else float(scale)
    window_left = window[0] if window is not None and window[0] is not None and window[0] >= 0 else None
    batch = page_table.shape[0]
    seq = cache_seqlens.tolist() if torch.is_tensor(cache_seqlens) else [int(cache_seqlens)] * batch
    cu = cu_seqlens_q.tolist() if cu_seqlens_q is not None else [i * (tokens // batch) for i in range(batch + 1)]
    if _state["shapes"] < 6:
        _state["shapes"] += 1
        LOG.warning("[attn-check] rank=%d prefill call: q %s %s k_cache %s %s page_table %s seqlens %s cu_q %s bias %s %s "
                    "scale %.5f window %s out %s %s lse %s", _rank(), tuple(q.shape), q.dtype, tuple(k_cache.shape),
                    k_cache.dtype, tuple(page_table.shape), seq[:4], cu[:5], tuple(rel_bias.shape), rel_bias.dtype, scale,
                    window, tuple(result[0].shape), result[0].dtype, tuple(result[1].shape))
    layer_kind = "global" if window_left is None else f"local{window_left}"
    geometry = f"q={cu[1] - cu[0]} k={seq[0]}" + (f" batch={batch}" if batch > 1 else "")
    for b in range(min(batch, 2)):
        qlen = cu[b + 1] - cu[b]
        if qlen <= 0:
            continue
        for i in sorted({0, qlen - 1, _rng.randrange(qlen), _rng.randrange(qlen)}):
            row, pos = cu[b] + i, seq[b] - qlen + i
            res = _compare(out[row], lse[row], q[row], k_cache, v_cache, page_table[b], pos, rel_bias[row], scale,
                           window_left)
            if res is not None:
                _note("prefill", geometry, layer_kind, pos, pos + 1, row, res)


@_guarded
def decode(q, k_cache, v_cache, page_table, seqused_k, rel_bias, scale, window_left, partials):
    """After a decode / verify partials call: q (batch * qlen, heads_q, d), the last qlen positions of each sequence;
    partials (o, m, l) per split: unnormalized output, running maximum, sum of exp(score - maximum)."""
    if torch.cuda.is_current_stream_capturing():
        return
    if _state["decode"] and time.time() - _state["decode_at"] > 20:
        flush("decode")
    if _rng.randrange(DECODE_EVERY):
        return
    o, m, l = partials[0], partials[1], partials[2]
    tokens, heads_q, dim = q.shape
    batch = seqused_k.shape[0]
    qlen = tokens // batch
    splits = o.shape[0]
    o = o.reshape(splits, tokens, heads_q, dim).double()
    m, l = m.reshape(splits, tokens, heads_q).double(), l.reshape(splits, tokens, heads_q).double()
    top = m.amax(0)
    corr = torch.exp(m - top)
    total = (corr * l).sum(0)
    out, lse = (corr[..., None] * o).sum(0) / total[..., None], top + torch.log(total)
    seq = seqused_k.tolist()
    layer_kind = "global" if window_left is None else f"local{window_left}"
    geometry = f"qlen={qlen} k~{seq[0] // 4096 * 4096} batch={batch}"
    for b in sorted({0, batch - 1}):
        for j in sorted({0, qlen - 1}):
            row, pos = b * qlen + j, seq[b] - qlen + j
            res = _compare(out[row], lse[row], q[row], k_cache, v_cache, page_table[b], pos,
                           rel_bias[row] if rel_bias is not None else None, float(scale), window_left)
            if res is not None:
                _note("decode", geometry, layer_kind, pos, pos + 1, row, res)
