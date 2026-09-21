"""Warm the next prefill chunk's Engram rows while the current chunk computes.

The Engram lookup of a prefill chunk gathers tens of thousands of rows per layer; on node-local NVMe
the cold ones cost a page fault each inside the gather callback, on the model's critical path. A
chunk's hash ids depend only on its tokens and their three predecessors, and the scheduler knows the
whole prompt, so when it launches an extend batch this hook hashes the *following* chunk of every
request on the CPU (SGLang's own hash arithmetic, `compute_engram_hash_ids`) and asks the row store
to page those rows in (`posix_fadvise(WILLNEED)`, no copy) from a background thread. By the time the
next chunk's callback runs, its rows are in the page cache; the measured effect is in docs/results.md
(Engram gather cost).

  SPARK_ENGRAM_PREFETCH   1 enables (default 1 when SPARK_ENGRAM_DIR is set)

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)

# Filled by engram_store when the model builds its hasher: CPU copies of the hash tables.
HASHER: dict = {}


def _impl():
    """The Engram row implementation in use: engram_store (rows gathered by a host-function node inside the
    graph, the default) or engram_staged (rows staged before each forward; SPARK_ENGRAM_MODE=staged)."""
    import importlib
    name = "engram_staged" if os.environ.get("SPARK_ENGRAM_MODE", "hostnode") == "staged" else "engram_store"
    return importlib.import_module(name)


def _hash_ids(tokens: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers.engram import compute_engram_hash_ids
    h = HASHER
    return compute_engram_hash_ids(tokens, blocked, h["pad_id"], h["token_map"], h["multipliers"],
                                   h["primes"], h["offsets"])


def _prefetch_chunk(ids: list, start: int, n: int, stores: dict, layer_ids: list) -> None:
    """ids = the request's prompt token ids; the chunk is ids[start:start+n] with 3 predecessors each."""
    store_mod = _impl()
    end = min(len(ids), start + n)
    if end <= start:
        return
    look = HASHER["ngram"]  # n: token + n-1 predecessors
    cols = []
    blocked_cols = []
    for s in range(look):
        lo = start - s
        col = [ids[t - s] if t - s >= 0 else 0 for t in range(start, end)]
        blk = [t - s < 0 for t in range(start, end)]
        cols.append(torch.tensor(col, dtype=torch.int64))
        blocked_cols.append(torch.tensor(blk, dtype=torch.bool))
    tokens = torch.stack(cols, dim=-1)
    blocked = torch.stack(blocked_cols, dim=-1)
    hashes = _hash_ids(tokens, blocked)  # [T, n_layers, n_hash_cols]
    for li, layer_id in enumerate(layer_ids):
        store = stores.get(layer_id)
        if store is None:
            continue
        flat = hashes[:, li, :].reshape(-1).contiguous()
        arr = flat.numpy()
        store_mod.prefetch_rows(layer_id, arr)


def install(module) -> None:
    """Patch sglang.srt.managers.scheduler.Scheduler.run_batch to prefetch the next chunk."""
    if os.environ.get("SPARK_ENGRAM_PREFETCH", "1") != "1":
        return
    cls = module.Scheduler
    stock = cls.run_batch
    state = {"pool": None, "warned": False}

    def run_batch(self, batch, *args, **kwargs):
        try:
            mode = getattr(batch, "forward_mode", None)
            chunk = getattr(self, "chunked_prefill_size", None)
            if mode is not None and mode.is_extend() and chunk and HASHER:
                store_mod = _impl()
                stores = {layer: store for layer, store in store_mod.STORES}
                layer_ids = HASHER["layer_ids"]
                jobs = []
                for req in getattr(batch, "reqs", []) or []:
                    ids = getattr(req, "origin_input_ids", None)
                    rng = getattr(req, "extend_range", None)
                    if ids is None or rng is None:
                        continue
                    start = len(getattr(req, "prefix_indices", ())) + int(getattr(rng, "length", 0))
                    if start < len(ids):
                        jobs.append((list(ids), start, int(chunk)))
                if jobs:
                    def work(jobs=jobs):
                        for ids, start, n in jobs:
                            _prefetch_chunk(ids, start, n, stores, layer_ids)
                    threading.Thread(target=work, daemon=True, name="engram-prefetch").start()
        except Exception as exc:  # noqa: BLE001 — prefetch is best effort
            if not state["warned"]:
                state["warned"] = True
                logger.warning("Engram prefetch disabled after error: %s", exc)
        return stock(self, batch, *args, **kwargs)

    cls.run_batch = run_batch
    logger.info("Engram prefetch: next-chunk rows warmed from the scheduler")
