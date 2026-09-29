"""Hybrid prompt-lookup + DSpark drafting: where the text being written repeats text already in the context, draft the
continuation of that earlier occurrence instead of DSpark's guess.

Answers that quote, restate or edit their input (summaries with verbatim quotes, "return the updated file", JSON with an
added field) copy long runs of the context. DSpark drafts 5 tokens from the model's hidden state and is right about 3-4
of them; a lookup drafter finds the last 4 committed tokens earlier in the context and proposes what followed them,
which on a copied run is right almost every time. The offline study over production outputs (tools/lookup_sim.py)
measured the hybrid at +3% tokens/s on copy tasks with DSpark's 5-token block, and +22-29% once the verify window is
widened to 10-16 tokens; this hook is phase 1, the 5-token hybrid, which needs no graph change and is where the
machinery is validated.

Per request row it keeps a device int32 table of the committed tokens (prompt + accepted output), indexed like
req_to_token:
  * prefill writes the prompt (and, after a retraction, the output so far) from NgramEmbeddingManager.prepare_for_forward
    on every EXTEND batch;
  * decode writes the verified row [anchor, drafts...] at the request's prefix length from update_after_verify. All six
    columns are written without a mask (no host sync); the ones past the committed length are rejected drafts that no
    lookup can read (it reads below the committed length) and the next commit overwrites them.
After DSpark proposes, a Triton kernel scans each row for the most recent earlier occurrence of the key
(3 last committed tokens + the anchor, the bonus token the block starts from), and the up-to-5 tokens that followed it
replace DSpark's tokens in rows with a match of at least SPARK_LOOKUP_MIN_TOKENS tokens. The replacement happens before
the worker builds verify_ids_2d, so the eager accept path and the verify graph's folded accept both see it.

It is lossless. Greedy rows: the verifier keeps the longest prefix the target model agrees with, whoever drafted it.
Sampled rows: the replaced positions' draft distribution (corrected_logits) is set to a one-hot on the lookup token,
the distribution that token was in fact drawn from, so speculative sampling's accept test and residual resample stay
exact. Unreplaced later positions keep DSpark's token and distribution, drawn independently of the target's
randomness, which is all the proof needs. Every TP rank builds the same table from the same batches and TP-synced
commit lengths, so every rank proposes the same tokens.

Prefill and verify write the table from different streams (the scheduler's and the forward stream). Under the overlap
scheduler the last verify of a finished request can still write into its row when the next request's prompt lands
there, so write_prefill makes its stream wait (on the GPU, no host sync) for the last write_verify: the new prompt then
always overwrites the old owner's tail on every rank, and every TP rank holds the same table, which the wide mode's
per-step graph choice depends on.

  SPARK_LOOKUP_DRAFT         1 enables (default 0)
  SPARK_LOOKUP_MIN_TOKENS    shortest lookup continuation that replaces DSpark's block (default 2)
  SPARK_LOOKUP_WINDOW        tokens back from the end that are searched, 0 = the whole context (default 0)
  SPARK_LOOKUP_LOG_SECONDS   interval of the hit/acceptance log line (default 60, 0 = off)
  SPARK_LOOKUP_EXT_KEY       extend/wide modes: tokens that must match before extending (default 8)

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

KEY = 4                                    # replace-mode key length: 3 committed tokens + the anchor
BLOCK = 4096                               # positions scanned per program

_ENABLED = os.environ.get("SPARK_LOOKUP_DRAFT", "0") == "1"
_MIN_TOKENS = int(os.environ.get("SPARK_LOOKUP_MIN_TOKENS", "2"))
_WINDOW = int(os.environ.get("SPARK_LOOKUP_WINDOW", "0"))
_LOG_SECONDS = float(os.environ.get("SPARK_LOOKUP_LOG_SECONDS", "60"))
_MODE = os.environ.get("SPARK_LOOKUP_MODE", "replace")
_BASE = int(os.environ.get("SPARK_LOOKUP_BASE", "5"))
# extend/wide key: the last EXT_KEY tokens of [committed..., anchor, DSpark's block]. Short keys match stock phrases in
# chat and code ("of the", "self.") whose continuation differs, and an extension the verifier rejects still costs the
# wider verify; 8 tokens keeps the copy spans (long) and drops most of those (window 32 notes in docs/results.md).
_EXT_KEY = int(os.environ.get("SPARK_LOOKUP_EXT_KEY", "8"))
_NON_TOKEN = 1_000_000          # SGLang's MM_PAD_SHIFT_VALUE: every id at or above it is a placeholder, not a token

# stats columns: rows proposed, rows replaced by lookup, lookup tokens proposed, accepted drafts in replaced rows,
# accepted drafts in DSpark rows
_STAT_NAMES = ("rows", "lookup_rows", "lookup_tokens", "accepted_lookup", "accepted_dspark")


class _State:
    def __init__(self):
        self.table = None                  # [rows, context] int32
        self.written: dict[int, tuple[str, int]] = {}   # row -> (rid, columns written by prefill)
        self.pending = None                # (rows, prefix_lens, replaced) of the proposal awaiting verify
        self.stats = None
        self.stats_host = None
        self.last_log = time.monotonic()
        self.logged_once = False
        self.wide = None                   # lookup_wide: {"gamma", "base", "epilogue", "runner", ...}
        self.layout = None                 # lookup_wide: the ragged verify layout of the step being proposed
        self.cl_buf = None                 # lookup_wide: [max_bs, gamma, V] draft logits of sampled rows
        self.step = 0                      # lookup_wide: decode steps proposed (identical on every TP rank)
        self.hot_until = -1                # sync gate: steps before this one may extend
        self.ring_host = None              # sync gate: pinned [4] int32 demand flags, written at step d, read at d+2
        self.ring_evt = None
        self.ring_meta = None              # sync gate: (step, was_cold) per ring slot
        self.gate = {"cold": 0, "hot": 0, "forced": 0, "missed": 0}
        self.uniform = {}                  # (bs, device) -> cached all-6 RaggedVerifyLayout
        self.verify_evt = None             # recorded after each write_verify; write_prefill's stream waits on it
        self.vocab = None                  # target vocabulary size; lookup continuations stop at the first id outside it


_S = _State()
_kernel = None


def _match_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _lookup_match(table_ptr, table_stride, rows_ptr, lens_ptr, key_ptr, best_ptr, window,
                      BLOCK: tl.constexpr, KLEN: tl.constexpr):
        """best[b] = max p <= L-KLEN with table[row, p:p+KLEN] == key[b]: the latest occurrence inside the committed
        tokens."""
        b = tl.program_id(0)
        blk = tl.program_id(1)
        row = tl.load(rows_ptr + b).to(tl.int64)
        L = tl.load(lens_ptr + b).to(tl.int64)
        lo = tl.maximum(L - window, 0)
        base = table_ptr + row * table_stride
        ok = L >= KLEN
        p = lo + blk * BLOCK + tl.arange(0, BLOCK)
        hit = ok & (p <= L - KLEN)
        for i in tl.static_range(KLEN):
            k = tl.load(key_ptr + b * KLEN + i)
            hit = hit & (tl.load(base + p + i, mask=hit, other=-2) == k)
        best = tl.max(tl.where(hit, p, -1), axis=0)
        tl.atomic_max(best_ptr + b, best, mask=best >= 0)

    _kernel = _lookup_match
    return _kernel


def match(table, rows, lens, key, window: int = 0):
    """Start of the most recent occurrence of key [bs, 4] inside each row's committed tokens, -1 if none. All device
    tensors; no host sync."""
    import torch
    import triton
    bs = rows.shape[0]
    best = torch.full((bs,), -1, dtype=torch.int64, device=table.device)
    span = window if window > 0 else table.shape[1]
    grid = (bs, triton.cdiv(span, BLOCK))
    _match_kernel()[grid](table, table.stride(0), rows, lens, key.to(torch.int32).contiguous(), best,
                          window if window > 0 else table.shape[1] + 1, BLOCK=BLOCK, KLEN=int(key.shape[1]))
    return best


def lookup_after(table, rows, lens, key, tail, n: int, window: int = 0):
    """(tokens [bs, n] int64, available [bs] int64): the n tokens that followed the latest occurrence of key. The
    sequence is the row's committed tokens (positions < L) followed by `tail` [bs, t] (positions L .. L+t-1, not in the
    table yet: the anchor, and in extend mode DSpark's drafts); `available` counts the leading tokens that exist."""
    import torch
    best = match(table, rows, lens, key, window)
    L = lens.to(torch.int64)
    t = tail.shape[1]
    follow = best + key.shape[1]
    idx = follow[:, None] + torch.arange(n, device=table.device)[None, :]
    safe = idx.clamp(0, table.shape[1] - 1)
    toks = table[rows.to(torch.int64)[:, None], safe].to(torch.int64)
    in_tail = (idx - L[:, None]).clamp(0, t - 1)
    toks = torch.where(idx < L[:, None], toks, tail.to(torch.int64).gather(1, in_tail))
    avail = torch.where(best >= 0, (L + t - follow).clamp(0, n), torch.zeros_like(best))
    # Prompts carry ids that are not tokens: image placeholders (MM_PAD_SHIFT_VALUE and up) stand in the table where
    # the image was. A continuation that runs into one ends there; drafted, it would index the embedding and the
    # draft distribution past the vocabulary (a device-side assert that took production-1.7 down, 2026-09-29).
    bad = (toks < 0) | (toks >= (_S.vocab or _NON_TOKEN))
    first_bad = torch.where(bad.any(1), bad.to(torch.int32).argmax(1), torch.full_like(avail, n))
    return toks, torch.minimum(avail, first_bad)


def _committed_tail(table, rows, lens, k: int):
    """The last k committed tokens of each row, [bs, k] (garbage for rows shorter than k, which cannot match)."""
    import torch
    L = lens.to(torch.int64)
    idx = (L[:, None] - k + torch.arange(k, device=table.device)[None, :]).clamp(0, table.shape[1] - 1)
    return table[rows.to(torch.int64)[:, None], idx].to(torch.int64)


def lookup_block(table, rows, lens, anchor, gamma: int, window: int = 0):
    """Replace mode: the continuation of (3 last committed tokens + anchor), up to gamma tokens."""
    import torch
    key = torch.cat([_committed_tail(table, rows, lens, KEY - 1), anchor.to(torch.int64)[:, None]], dim=1)
    return lookup_after(table, rows, lens, key, anchor.to(torch.int64)[:, None], gamma, window)


def lookup_extension(table, rows, lens, anchor, drafts, base: int, window: int = 0, n: int | None = None):
    """Extend mode: the continuation of the last 4 tokens of [anchor, drafts[:, :base]], for positions base.. of the
    block. The key includes DSpark's drafts, the occurrence must lie in the committed tokens."""
    import torch
    key, tail = _ext_key(table, rows, lens, anchor, drafts, base)
    return lookup_after(table, rows, lens, key, tail, drafts.shape[1] - base if n is None else n, window)


def _ext_key(table, rows, lens, anchor, drafts, base: int, key_len: int | None = None):
    """(key [bs, K], tail [bs, base+1]): the last K tokens of [committed..., anchor, drafts[:, :base]] and the part of
    that sequence not yet in the table. Rows with fewer committed tokens than the key needs cannot match (the matcher
    requires L >= K)."""
    import torch
    k = _EXT_KEY if key_len is None else key_len
    tail = torch.cat([anchor.to(torch.int64)[:, None], drafts[:, :base].to(torch.int64)], dim=1)
    t = tail.shape[1]
    if k <= t:
        return tail[:, -k:], tail
    return torch.cat([_committed_tail(table, rows, lens, k - t), tail], dim=1), tail


def _ensure_table(batch):
    import torch
    if _S.table is None:
        r2t = batch.req_to_token_pool.req_to_token
        _S.table = torch.zeros(r2t.shape[0], r2t.shape[1], dtype=torch.int32, device=r2t.device)
        _S.stats = torch.zeros(len(_STAT_NAMES), dtype=torch.int64, device=r2t.device)
        _S.stats_host = torch.zeros(len(_STAT_NAMES), dtype=torch.int64)
        if r2t.is_cuda:
            _S.stats_host = _S.stats_host.pin_memory()
        logger.warning("lookup draft on (%s%s): token table %d rows x %d tokens (%.0f MB), min tokens %d, window %s",
                       _MODE, f" after {_BASE} DSpark tokens" if _MODE != "replace" else "", r2t.shape[0], r2t.shape[1],
                       _S.table.numel() * 4 / 2**20, _MIN_TOKENS, _WINDOW or "all")
    return _S.table


def write_prefill(batch) -> None:
    """Copy each extending request's tokens into its row (from 0 when the row held another request)."""
    import torch
    table = _ensure_table(batch)
    width = table.shape[1]
    spans, flat = [], []
    for req in batch.reqs:
        row = req.kv.req_pool_idx
        rng = getattr(req, "extend_range", None)
        if row is None or rng is None:
            continue
        start = len(req.prefix_indices)
        end = min(start + rng.length, width)
        rid, done = _S.written.get(row, (None, 0))
        lo = min(done, start) if rid == req.rid else 0
        if end <= lo:
            continue
        origin = req.origin_input_ids
        toks = origin[lo:end] if end <= len(origin) else (origin + req.output_ids)[lo:end]
        spans.append((row, lo, len(toks)))
        flat.extend(toks)
        _S.written[row] = (req.rid, end)
    if not spans:
        return
    dev = torch.tensor(flat, dtype=torch.int32)
    if table.is_cuda:
        dev = dev.pin_memory().to(table.device, non_blocking=True)
        if _S.verify_evt is not None:      # land after the last verify write (a finished owner's overshoot step)
            torch.cuda.current_stream(table.device).wait_event(_S.verify_evt)
    off = 0
    for row, lo, n in spans:
        table[row, lo:lo + n] = dev[off:off + n]
        off += n


def write_verify(verify_ids_2d, req_pool_indices, commit_lens) -> None:
    import torch
    if _S.table is None or _S.pending is None:
        return
    rows, prefix, replaced = _S.pending
    _S.pending = None
    table = _S.table
    w = verify_ids_2d.shape[1]
    pos = (prefix.to(torch.int64)[:, None] + torch.arange(w, device=table.device)[None, :]).clamp(max=table.shape[1] - 1)
    table[req_pool_indices.to(torch.int64)[:, None], pos] = verify_ids_2d.to(torch.int32)
    if table.is_cuda:
        if _S.verify_evt is None:
            _S.verify_evt = torch.cuda.Event()
        _S.verify_evt.record()             # forward stream, after this step's (possibly stale) row write
    accepted = (commit_lens.to(torch.int64) - 1).clamp_min(0)
    if _MODE in ("extend", "wide"):   # lookup rows: accepted tokens beyond DSpark's block; other column: all accepted drafts
        _S.stats[3] += ((accepted - _BASE).clamp_min(0) * replaced).sum()
        _S.stats[4] += accepted.sum()
    else:
        _S.stats[3] += (accepted * replaced).sum()
        _S.stats[4] += (accepted * ~replaced).sum()
    _maybe_log()


def _maybe_log() -> None:
    if _LOG_SECONDS <= 0:
        return
    now = time.monotonic()
    if now - _S.last_log < _LOG_SECONDS:
        return
    _S.last_log = now
    if _S.logged_once:   # the previous interval's non-blocking copy has long landed
        v = dict(zip(_STAT_NAMES, _S.stats_host.tolist()))
        rows, lk = max(1, v["rows"]), max(1, v["lookup_rows"])
        other = max(1, v["rows"] - v["lookup_rows"])
        if _MODE in ("extend", "wide"):
            logger.warning("lookup extend: %d/%d rows (%.1f%%) extended, %.2f extension tokens/row, %.2f accepted beyond "
                           "DSpark's %d; accepted drafts per row overall %.2f; gate steps %s", v["lookup_rows"], v["rows"],
                           100 * v["lookup_rows"] / rows, v["lookup_tokens"] / lk, v["accepted_lookup"] / lk, _BASE,
                           v["accepted_dspark"] / rows, _S.gate)
        else:
            logger.warning("lookup draft: %d/%d rows (%.1f%%) drafted by lookup, %.2f lookup tokens/row; accepted "
                           "drafts per row: lookup %.2f, DSpark %.2f", v["lookup_rows"], v["rows"],
                           100 * v["lookup_rows"] / rows, v["lookup_tokens"] / lk, v["accepted_lookup"] / lk,
                           v["accepted_dspark"] / other)
    _S.stats_host.copy_(_S.stats, non_blocking=True)
    _S.logged_once = True


def apply(proposal, batch):
    """Replace DSpark's block by the lookup continuation in rows that have one."""
    import msgspec
    import torch
    if _S.table is None or getattr(batch, "has_grammar", False):
        return proposal
    block = proposal.draft_block
    drafts = block.draft_tokens
    bs, gamma = drafts.shape
    rows = batch.req_pool_indices
    lens = batch.seq_lens
    anchor = proposal.draft_block_ids[:, 0]
    col = torch.arange(gamma, device=drafts.device)[None, :]
    if _MODE == "extend":
        if gamma <= _BASE:
            return proposal
        ext, avail = lookup_extension(_S.table, rows, lens, anchor, drafts, _BASE, _WINDOW)
        toks = torch.cat([drafts[:, :_BASE].to(torch.int64), ext], dim=1)
        replaced = avail >= _MIN_TOKENS
        pos_mask = replaced[:, None] & (col >= _BASE) & (col < _BASE + avail[:, None])
    else:
        toks, avail = lookup_block(_S.table, rows, lens, anchor, gamma, _WINDOW)
        replaced = avail >= _MIN_TOKENS
        pos_mask = replaced[:, None] & (col < avail[:, None])
    new = torch.where(pos_mask, toks, drafts)
    logits = block.corrected_logits
    if logits is not None:
        # One-hot draft distribution at the replaced positions (in place: the buffer is rewritten every draft step).
        kept = logits.gather(2, new[:, :, None])
        logits.masked_fill_(pos_mask[:, :, None], torch.finfo(logits.dtype).min)
        logits.scatter_(2, new[:, :, None], torch.where(pos_mask[:, :, None], torch.zeros_like(kept), kept))
    _S.pending = (rows, lens.clone(), replaced)
    _S.stats[0] += bs
    _S.stats[1] += replaced.sum()
    _S.stats[2] += (avail * replaced).sum()
    return msgspec.structs.replace(proposal, draft_block=msgspec.structs.replace(block, draft_tokens=new))


def _forced_sync(batch) -> bool:
    """SPARK_LOOKUP_GATE_OVERRIDE: a request asking for the synced path (measurement only)."""
    for req in getattr(batch, "reqs", None) or ():
        cp = getattr(getattr(req, "sampling_params", None), "custom_params", None)
        if cp and cp.get("spark_lookup_force_sync"):
            return True
    return False


def _host_lens(verify_lens) -> list:
    """The one host synchronisation apply_wide can make (tests wrap this to prove cold steps never call it)."""
    return verify_lens.tolist()


def _graph_grid(w, all_narrow: bool):
    """Captured token buckets a step may use; with the fused ratio-2 compressor, all-6 steps use the uniform 6b graphs
    and every other step the others (a uniform graph must only ever replay a uniform layout)."""
    runner = w["runner"].decode_cuda_graph_runner
    grid = runner.capture_num_tokens if runner is not None and runner.capture_num_tokens else None
    if grid is None or not w.get("split_grid"):
        return grid
    narrow = w["base"] + 1
    return [g for g in grid if (g % narrow == 0) == all_narrow]


def _uniform_layout(w, bs: int, dev):
    """The all-6 layout of a batch of bs requests, built once per bs and only ever read afterwards."""
    import torch
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, round_up_grid
    key = (bs, str(dev))
    lay = _S.uniform.get(key)
    if lay is None:
        narrow = w["base"] + 1
        total = narrow * bs
        grid = _graph_grid(w, True)
        lay = RaggedVerifyLayout._assemble_device(
            verify_lens=torch.full((bs,), narrow, dtype=torch.int32, device=dev),
            graph_num_tokens=round_up_grid(total, grid) if grid else total,
            verify_lens_cpu=[narrow] * bs, total_verify_tokens=total)
        _S.uniform[key] = lay
    return lay


def _gate_read(w, d: int) -> None:
    """Open the gate for HOLD steps when the batch showed extension demand two steps ago. Two steps back, that
    step's results have been synchronised by the overlap scheduler already, so the event wait is free, and the value
    is the same on every TP rank (never a query(): a timing-dependent answer would split the ranks' graph choice)."""
    k = (d - 2) % 4
    meta = _S.ring_meta[k]
    if meta is None or meta[0] != d - 2:
        return
    if _S.ring_evt[k] is not None:
        _S.ring_evt[k].synchronize()
    if int(_S.ring_host[k]):
        _S.hot_until = d + w["hold"]
        if meta[1]:
            _S.gate["missed"] += 1


def _gate_record(demand, d: int, cold: bool) -> None:
    import torch
    k = d % 4
    if _S.ring_evt[k] is not None:
        _S.ring_host[k:k + 1].copy_(demand.view(1).to(torch.int32), non_blocking=True)
        _S.ring_evt[k].record()
    else:
        _S.ring_host[k:k + 1].copy_(demand.view(1).to(torch.int32))
    _S.ring_meta[k] = (d, cold)


def apply_wide(proposal, batch):
    """lookup_wide: DSpark's block followed by the lookup continuation, padded to the verify gamma; builds the step's
    ragged verify layout and fills the folded epilogue's draft buffer. A step that may extend copies its row lengths to
    the host once (after every device op is queued) to pick the smallest graph; with the sync gate, a step whose batch
    showed no extension demand in the last HOLD steps (read two steps late) verifies DSpark's block only and reuses a
    cached all-6 layout: no host synchronisation at all."""
    import msgspec
    import torch
    from sglang.kernels.ops.speculative.ragged_verify_kernels import BuildQoIndptr
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, round_up_grid
    w = _S.wide
    gamma, base, budget = w["gamma"], w["base"], w["budget"]
    block = proposal.draft_block
    drafts = block.draft_tokens
    bs = drafts.shape[0]
    dev = drafts.device
    n = gamma - base
    anchor = proposal.draft_block_ids[:, 0]
    d = _S.step
    _S.step += 1
    eligible = _S.table is not None and not getattr(batch, "has_grammar", False) and n > 0
    gate = w.get("gate", False)
    if gate:
        if _S.ring_host is None:
            _S.ring_host = torch.zeros(4, dtype=torch.int32, pin_memory=dev.type == "cuda")
            _S.ring_evt = [torch.cuda.Event() if dev.type == "cuda" else None for _ in range(4)]
            _S.ring_meta = [None] * 4
        _gate_read(w, d)
        forced = w.get("override", False) and _forced_sync(batch)
        hot = eligible and (d < _S.hot_until or forced)
        _S.gate["forced" if forced else ("hot" if hot else "cold")] += 1
    else:
        hot = eligible
    zeros = None
    if hot:
        ext, avail = lookup_extension(_S.table, batch.req_pool_indices, batch.seq_lens, anchor, drafts, base,
                                      _WINDOW, n)
        want = avail >= _MIN_TOKENS
        ext_len = torch.where(want, avail, torch.zeros_like(avail))
        before = torch.cumsum(ext_len, 0) - ext_len
        ext_len = torch.minimum(ext_len, (budget - before).clamp_min(0))
        col = torch.arange(n, device=dev)[None, :]
        ext = torch.where(col < ext_len[:, None], ext, torch.zeros_like(ext))
        if gate:
            _gate_record(want.any(), d, cold=False)
    else:
        zeros = torch.zeros(bs, n, dtype=torch.int64, device=dev)
        ext = zeros
        ext_len = None
        if gate and eligible:   # demand only: the matcher, no gathers
            key, tail = _ext_key(_S.table, batch.req_pool_indices, batch.seq_lens, anchor, drafts, base)
            best = match(_S.table, batch.req_pool_indices, batch.seq_lens, key, _WINDOW)
            avail = torch.where(best >= 0,
                                (batch.seq_lens.to(torch.int64) + tail.shape[1] - best - key.shape[1]).clamp(0, n),
                                torch.zeros_like(best))
            _gate_record((avail >= _MIN_TOKENS).any(), d, cold=True)
        elif gate:
            _S.ring_meta[d % 4] = None
    new = torch.cat([drafts.to(torch.int64), ext], dim=1).contiguous()
    logits = block.corrected_logits
    if logits is not None:
        if _S.cl_buf is None or _S.cl_buf.shape[0] < bs or _S.cl_buf.dtype != logits.dtype:
            _S.cl_buf = torch.empty(max(bs, w["max_bs"]), gamma, logits.shape[-1], dtype=logits.dtype, device=dev)
        buf = _S.cl_buf[:bs]
        buf[:, :base].copy_(logits)
        tail = buf[:, base:]
        tail.fill_(torch.finfo(logits.dtype).min)
        tail.scatter_(2, ext[:, :, None], 0.0)             # one-hot on the lookup token (padding rows: never verified)
        logits = buf
    epilogue = w["epilogue"]
    if epilogue is not None:
        epilogue.draft_tokens_buf[: bs * gamma].copy_(new.view(-1))
    # SGLang's compact verify-id gather (dspark_verify_window._compact_verify_ids_gather_kernel) reads row r's anchor
    # at draft_block_ids[r * gamma] with gamma = draft_tokens.shape[1]: the ids must be as wide as the drafts. The
    # split proposer's ids are [bs, base]; left at that stride, every row but the first verified a stray anchor.
    ids = proposal.draft_block_ids
    if ids.shape[1] < gamma:
        ids = torch.cat([ids, ids[:, :1].expand(bs, gamma - ids.shape[1])], dim=1)
    ids = ids.to(torch.int64).contiguous()
    extended = ext_len > 0 if ext_len is not None else torch.zeros(bs, dtype=torch.bool, device=dev)
    _S.pending = (batch.req_pool_indices, batch.seq_lens.clone(), extended)
    if _S.stats is not None:
        _S.stats[0] += bs
        if ext_len is not None:
            _S.stats[1] += extended.sum()
            _S.stats[2] += ext_len.sum()
    if ext_len is None:
        _S.layout = _uniform_layout(w, bs, dev)
    else:
        verify_lens = (ext_len + base + 1).to(torch.int32)
        qo = BuildQoIndptr.execute(verify_lens=verify_lens)
        lens_cpu = _host_lens(verify_lens)                # everything above is queued before this wait
        total = sum(lens_cpu)
        grid = _graph_grid(w, all(v == base + 1 for v in lens_cpu))
        _S.layout = RaggedVerifyLayout(
            verify_lens=verify_lens, graph_num_tokens=round_up_grid(total, grid) if grid else total,
            extend_start_loc=qo.extend_start_loc, qo_indptr_device=qo.qo_indptr,
            verify_lens_cpu=lens_cpu, total_verify_tokens=total)
    return msgspec.structs.replace(
        proposal, draft_block_ids=ids,
        draft_block=msgspec.structs.replace(block, draft_tokens=new, corrected_logits=logits))


def install_manager(module) -> None:
    """model_runner_components.ngram_embedding_manager: record prefill and verified tokens in the lookup table."""
    if not _ENABLED:
        return
    cls = module.NgramEmbeddingManager
    stock_prepare, stock_verify = cls.prepare_for_forward, cls.update_after_verify
    extend = module.ForwardMode.EXTEND

    def prepare_for_forward(self, batch, *, chunked_req):
        out = stock_prepare(self, batch, chunked_req=chunked_req)
        if batch is not None and batch.forward_mode == extend:
            write_prefill(batch)
        return out

    def update_after_verify(self, *, verify_ids_2d, req_pool_indices, commit_lens):
        stock_verify(self, verify_ids_2d=verify_ids_2d, req_pool_indices=req_pool_indices, commit_lens=commit_lens)
        write_verify(verify_ids_2d, req_pool_indices, commit_lens)

    cls.prepare_for_forward = prepare_for_forward
    cls.update_after_verify = update_after_verify


def install_proposer(module) -> None:
    """speculative.dspark_components.dspark_draft: swap lookup continuations into DSpark's proposal."""
    if not _ENABLED:
        return
    cls = module.DraftBlockProposer
    stock = cls.propose

    def propose(self, *, batch, **kw):
        proposal = stock(self, batch=batch, **kw)
        return apply_wide(proposal, batch) if _S.wide is not None else apply(proposal, batch)

    cls.propose = propose
