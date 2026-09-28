"""Engram rows staged from node-local NVMe before every SGLang forward.

DeepSeek-V4.1-Flash keeps two Engram hash tables (layers 1 and 14) of ~384 million
fp8 rows each: [N, 256] e4m3 payload plus [N, 8] e8m0 block scales, ~101 GB per
layer. Stock SGLang shards the rows over the TP group in device memory or pins the
whole table in host memory. On a DGX Spark the "host" memory IS the GPU memory
(128 GB unified), so neither fits next to the model. This module replaces how
``EngramEmbedding`` obtains rows: each TP rank owns one contiguous global row range
[lo, hi) held in a node-local sparse copy of the checkpoint shard (or in the full
shard on the node that has it) and reads the rows of a lookup on demand through a
read-only memory map with readahead disabled.

The mechanism (the one our vLLM port runs in production, mapped onto SGLang):

1. ``install_runner`` hooks ``ModelRunner.forward``. Before the stock forward it
   hashes the batch's token ids on the GPU with the model's own ``EngramHasher``
   inputs and kernel -- but WITHOUT the hasher's history commit, which the model's
   own call performs exactly once per step, eagerly or inside the captured graph --
   copies the ``[T, n_layers, n_cols]`` ids to pinned host memory (the one host sync
   of the step), gathers this rank's rows for every Engram layer from the memory
   map with a thread pool straight into pinned staging (rows the rank does not own
   are zero), and issues one async H2D copy per layer into that layer's persistent
   device staging on the current stream.
2. ``install`` hooks ``EngramEmbedding``: ``__init__`` allocates nothing but
   zero-size parameters (their ``weight_loader`` only validates shapes) and opens
   the memory map; ``_owned_rows`` no longer reads a table at all -- it gathers the
   staged device rows BY POSITION with SGLang's Triton ``engram_gather`` (which does
   the e8m0 dequantization to bf16), so the captured decode / DSpark-verify graphs
   contain nothing but that kernel. The graph runner captures ``model.forward``
   directly, so the runner hook never runs during capture: warm-up calls stage
   in-forward (synchronously), the recorded kernel reads the persistent device
   staging, and every replay is preceded by the runner hook refilling it. A padded
   graph reads stale-but-finite rows for its padding tokens, whose outputs SGLang
   discards.
3. ``prefetch_rows`` warms rows in the page cache with ``posix_fadvise(WILLNEED)``
   from a small pool; ``engram_prefetch`` calls it for the next prefill chunk.

Why the memory map with MADV_RANDOM: a cold row costs one 4 KiB page fault on NVMe;
with kernel readahead every fault would pull a whole readahead window (measured on
the vLLM stack: 9K-token prefill 14.7 s vs 8.1 s, decode 53 vs 61 tok/s). Small
gathers (decode) first ``fadvise`` every row so all reads are in flight together,
then copy on the calling thread; large gathers (prefill chunks) are spread over the
pool so many faults are in flight.

Environment (SPARK_ENGRAM_DIR is the master switch; nothing here runs without it):
  SPARK_ENGRAM_DIR            directory with model.safetensors.index.json, the shard(s)
                              holding the Engram tables and, for a sparse copy made by
                              tools/engram_local.py, engram-local.json with this rank's
                              row range per layer.
  SPARK_ENGRAM_ROWS           "1:lo:hi,14:lo:hi" -- the ranges when there is no manifest
                              (a full checkpoint snapshot on the node that holds it).
  SPARK_ENGRAM_THREADS        gather pool size (default 64; more threads = more page
                              faults in flight, the pool mostly waits on NVMe).
  SPARK_ENGRAM_MAX_IDS        staging capacity in ids per layer per step (default 262144
                              = tokens x hash columns); a larger lookup is an error.
  SPARK_ENGRAM_STATS_SECONDS  period of the per-layer INFO line (default 60, 0 disables):
                              stagings, owned / zeroed rows, mean stage time per step.
  SPARK_ENGRAM_CACHE_GB       page-cache budget of the row spans per process (default 4; 0 = unbounded): above it
                              the cached rows are dropped and read from NVMe again when needed (see engram_store).
  SPARK_ENGRAM_CACHE_SECONDS  how often the budget is checked (default 5).

Ported from the author's vLLM Engram-on-disk implementation, which builds on
tonyd2wild and Kai's MIT DeepSeek-V4.1-Flash-vLLM-DGX-Spark disk path (Copyright (c)
2026 Tech2wild) and on vLLM's engram.py (Copyright contributors to the vLLM project,
Apache-2.0). MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import ctypes
import json
import logging
import mmap
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

ROW_BYTES = 256  # fp8 e4m3 payload bytes per row (= head_dim)
BLOCK_SIZE = 32  # values per e8m0 scale
SCALE_BYTES = ROW_BYTES // BLOCK_SIZE  # 8 e8m0 bytes per row
PAGE = 4096

# Gathers of at most this many rows skip the pool: fadvise every row (all reads in
# flight at once), then one fancy-index copy per table on the calling thread.
INLINE_ROWS = 2048
# Rows per prefetch task.
PREFETCH_TASK_ROWS = 512

# Filled by EngramEmbedding.__init__: (layer_id, RowStore) per Engram layer of this rank.
STORES: list[tuple[int, "RowStore"]] = []
_LAYERS: dict[int, "torch.nn.Module"] = {}

_POOL: Optional[ThreadPoolExecutor] = None
_PREFETCH_POOL: Optional[ThreadPoolExecutor] = None
_ENGRAM_MOD = None  # the executed sglang.srt.layers.engram module, set by install()
_RUNNER_HOOKED = False
_STATS_THREAD: Optional[threading.Thread] = None


# ----------------------------------------------------------------------------- env

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name}={raw!r} is not an integer") from exc


def engram_dir() -> Optional[Path]:
    raw = os.environ.get("SPARK_ENGRAM_DIR", "").strip()
    return Path(raw) if raw else None


def threads() -> int:
    return max(1, _env_int("SPARK_ENGRAM_THREADS", 64))


def capacity() -> int:
    return max(1, _env_int("SPARK_ENGRAM_MAX_IDS", 262144))


# --------------------------------------------------------------- checkpoint layout

def _read_header(shard: Path) -> tuple[int, dict]:
    """safetensors header: 8-byte little-endian JSON length, then the JSON."""
    with open(shard, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return n, json.loads(f.read(n))


def table_span(model_dir: Path, layer_id: int) -> tuple[Path, int, int, int]:
    """(shard path, N, weight byte offset, scale byte offset) of one layer's table.

    Both tensors must sit in the same shard; the byte offset of row 0 is
    8 + header length + data_offsets[0]. Adapted from DiskEngramTable._open."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if not index.is_file():
        raise RuntimeError(f"Engram: {index} not found (SPARK_ENGRAM_DIR={model_dir})")
    with open(index) as f:
        weight_map = json.load(f)["weight_map"]
    wname = f"layers.{layer_id}.engram.embed.weight"
    sname = f"layers.{layer_id}.engram.embed.scale"
    for name in (wname, sname):
        if name not in weight_map:
            raise RuntimeError(f"Engram: {name} is not in {index}")
    if weight_map[wname] != weight_map[sname]:
        raise RuntimeError(
            f"Engram layer {layer_id}: weight and scale live in different shards "
            f"({weight_map[wname]} / {weight_map[sname]}); one memory map per layer needs both in one"
        )
    shard = model_dir / weight_map[wname]
    if not shard.is_file():
        raise RuntimeError(f"Engram layer {layer_id}: shard {shard} not found")
    hlen, header = _read_header(shard)
    w, s = header[wname], header[sname]
    if w["dtype"] != "F8_E4M3" or s["dtype"] != "F8_E8M0":
        raise RuntimeError(f"Engram layer {layer_id}: unexpected dtypes {w['dtype']} / {s['dtype']}")
    n = int(w["shape"][0])
    if tuple(w["shape"]) != (n, ROW_BYTES) or tuple(s["shape"]) != (n, SCALE_BYTES):
        raise RuntimeError(
            f"Engram layer {layer_id}: shapes {w['shape']} / {s['shape']} are not "
            f"[N, {ROW_BYTES}] / [N, {SCALE_BYTES}]"
        )
    base = 8 + hlen
    return shard, n, base + int(w["data_offsets"][0]), base + int(s["data_offsets"][0])


def _parse_rows_env(raw: str) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            layer, lo, hi = (int(x) for x in item.split(":"))
        except ValueError as exc:
            raise RuntimeError(
                f"SPARK_ENGRAM_ROWS={raw!r}: expected 'layer:lo:hi,...' (got {item!r})"
            ) from exc
        ranges[layer] = (lo, hi)
    return ranges


def row_ranges(model_dir: Path) -> dict[int, tuple[int, int]]:
    """This rank's global row range per Engram layer: from the sparse copy's
    engram-local.json, else from SPARK_ENGRAM_ROWS. The manifest describes what is
    physically on disk, so when both exist they must agree."""
    model_dir = Path(model_dir)
    manifest = model_dir / "engram-local.json"
    env_ranges = _parse_rows_env(os.environ.get("SPARK_ENGRAM_ROWS", ""))
    if manifest.is_file():
        with open(manifest) as f:
            layers = json.load(f)["layers"]
        ranges = {int(k): (int(v[0]), int(v[1])) for k, v in layers.items()}
        for layer, rng in env_ranges.items():
            if layer in ranges and ranges[layer] != rng:
                raise RuntimeError(
                    f"Engram layer {layer}: SPARK_ENGRAM_ROWS says {rng} but {manifest} "
                    f"holds rows {ranges[layer]}; the sparse copy has holes outside its range"
                )
        return ranges
    if env_ranges:
        return env_ranges
    raise RuntimeError(
        f"Engram: no {manifest} and SPARK_ENGRAM_ROWS is unset; the row range per layer "
        "must come from one of them"
    )


# ------------------------------------------------------------------- memmap gather

def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=threads(), thread_name_prefix="engram-rows")
    return _POOL


def _prefetch_pool() -> ThreadPoolExecutor:
    global _PREFETCH_POOL
    if _PREFETCH_POOL is None:
        _PREFETCH_POOL = ThreadPoolExecutor(
            max_workers=max(4, threads() // 2), thread_name_prefix="engram-prefetch"
        )
    return _PREFETCH_POOL


def _map_file(path: Path) -> tuple[mmap.mmap, np.ndarray]:
    """Read-only shared mapping of a whole shard with readahead DISABLED
    (MADV_RANDOM): a fault brings exactly the 4 KiB page of the row."""
    fd = os.open(path, os.O_RDONLY)
    try:
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)
    try:
        mm.madvise(mmap.MADV_RANDOM)
    except (AttributeError, OSError):  # pragma: no cover - old kernels
        pass
    return mm, np.frombuffer(mm, dtype=np.uint8)


def _mmap_rows(view: np.ndarray, rel: np.ndarray, lo: int, hi: int, out: np.ndarray) -> None:
    """out[lo:hi] = view[rel[lo:hi]]: one fancy-index gather, C memcpy for resident
    pages, page faults for cold ones; NumPy releases the GIL inside."""
    np.take(view, rel[lo:hi], axis=0, out=out[lo:hi])


def _fadvise(fd: int, off: int, length: int) -> None:
    try:
        os.posix_fadvise(fd, off, length, os.POSIX_FADV_WILLNEED)
    except OSError:
        pass


def _parallel_read(jobs: list) -> None:
    """jobs: [(store, rel int64 array, view, out uint8 array, row_bytes, base_off)].
    Every row of every job is in flight at once. Small batches (decode sizes) skip
    the pool: readahead of every row page first, then one gather per job on the
    calling thread. Large ones (prefill chunks) carry ceil(total / threads) rows per
    task. Only the calling thread submits, so the shared pool cannot deadlock."""
    total = sum(job[1].shape[0] for job in jobs)
    if total == 0:
        return
    if total <= INLINE_ROWS:
        for store, rel, view, out, row_bytes, base in jobs:
            fd = store.fd
            if row_bytes >= PAGE // 8:
                for r in rel.tolist():
                    _fadvise(fd, base + r * row_bytes, row_bytes)
            else:
                # Many scale rows share one page: one advice per page.
                for p in np.unique((base + rel * row_bytes) >> 12).tolist():
                    _fadvise(fd, p << 12, PAGE)
        for store, rel, view, out, row_bytes, base in jobs:
            _mmap_rows(view, rel, 0, rel.shape[0], out)
        return
    chunk = max(1, -(-total // threads()))
    pool = _pool()
    futs = [
        pool.submit(_mmap_rows, view, rel, lo, min(lo + chunk, rel.shape[0]), out)
        for store, rel, view, out, row_bytes, base in jobs
        for lo in range(0, rel.shape[0], chunk)
    ]
    for fut in futs:
        fut.result()


class RowStore:
    """This rank's rows [lo, hi) of one layer's table, as [rows, 256] / [rows, 8]
    uint8 views through a read-only memory map of the shard (opened once; pages come
    and go with the page cache, nothing is pinned)."""

    def __init__(self, layer_id: int, shard: Path, n_rows: int, w_off: int, s_off: int,
                 lo: int, hi: int):
        self.layer_id = layer_id
        self.path = shard
        self.n_rows = n_rows
        self.lo, self.hi = lo, hi
        self.rows = hi - lo
        self.fd = os.open(shard, os.O_RDONLY)
        try:
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        except OSError:
            pass
        self.mm, buf = _map_file(shard)
        self.w_base = w_off + lo * ROW_BYTES
        self.s_base = s_off + lo * SCALE_BYTES
        self.w_view = np.ndarray((self.rows, ROW_BYTES), dtype=np.uint8, buffer=buf, offset=self.w_base)
        self.s_view = np.ndarray((self.rows, SCALE_BYTES), dtype=np.uint8, buffer=buf, offset=self.s_base)
        # Cumulative stats, read by the stats thread (single writer: the forward thread).
        self.stat_calls = 0
        self.stat_inline = 0
        self.stat_owned = 0
        self.stat_zeroed = 0
        self.stat_seconds = 0.0

    def _spans(self):
        pg = mmap.PAGESIZE
        for base, length in ((self.w_base, self.rows * ROW_BYTES), (self.s_base, self.rows * SCALE_BYTES)):
            lo = base // pg * pg
            hi = min((base + length + pg - 1) // pg * pg, len(self.mm))
            if hi > lo:
                yield lo, hi - lo

    def resident(self) -> int:
        """Bytes of this rank's row spans in the page cache (mincore through a private mapping of the same file)."""
        total = 0
        for off, length in self._spans():
            addr = _LIBC.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, self.fd, off)
            if addr in (None, ctypes.c_void_p(-1).value):
                continue
            try:
                vec = ctypes.create_string_buffer(length // mmap.PAGESIZE)
                if _LIBC.mincore(addr, length, vec) == 0:
                    total += int((np.frombuffer(vec.raw, dtype=np.uint8) & 1).sum()) * mmap.PAGESIZE
            finally:
                _LIBC.munmap(addr, length)
        return total

    def release(self) -> None:
        """Drop the shard from this process's mapping, then its page cache. Whole file: the kernel keeps file data in
        large folios and skips any folio reaching outside a DONTNEED range (engram_rows.c explains more)."""
        try:
            self.mm.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def jobs(self, ids: np.ndarray, w_out: np.ndarray, s_out: np.ndarray):
        """Read jobs for global ids -> rows into w_out [n, 256] / s_out [n, 8]; the
        caller zeroes the unowned positions afterwards. Returns (jobs, owned mask)."""
        owned = (ids >= self.lo) & (ids < self.hi)
        if self.rows == 0:
            return [], owned
        rel = np.where(owned, ids - self.lo, 0).astype(np.int64, copy=False)
        return [
            (self, rel, self.w_view, w_out, ROW_BYTES, self.w_base),
            (self, rel, self.s_view, s_out, SCALE_BYTES, self.s_base),
        ], owned

    def warm(self, rel: np.ndarray) -> None:
        """Page in rank-local rows `rel`: weight rows by exact byte range, scale rows
        by 4 KiB page (512 rows share one). Advice only, no copy."""
        for r in rel.tolist():
            _fadvise(self.fd, self.w_base + r * ROW_BYTES, ROW_BYTES)
        for p in np.unique((self.s_base + rel * SCALE_BYTES) >> 12).tolist():
            _fadvise(self.fd, p << 12, PAGE)


def gather_many(requests: list) -> list[np.ndarray]:
    """requests = [(RowStore, ids int64 [n], w_out uint8 [n, 256], s_out uint8 [n, 8])].
    The reads of every layer go out in ONE parallel batch, so a decode step costs
    about one NVMe latency instead of one per table. Returns the owned masks."""
    jobs, masks = [], []
    for store, ids, w_out, s_out in requests:
        j, owned = store.jobs(ids, w_out, s_out)
        jobs += j
        masks.append(owned)
    _parallel_read(jobs)
    for (store, ids, w_out, s_out), owned in zip(requests, masks):
        if not owned.all():
            unowned = ~owned
            w_out[unowned] = 0
            s_out[unowned] = 0
    return masks


def prefetch_rows(layer_id: int, ids: np.ndarray) -> None:
    """Warm this rank's rows among global `ids` (int64) of `layer_id` in the page
    cache, in background tasks of PREFETCH_TASK_ROWS rows. Best effort."""
    entry = _LAYERS.get(layer_id)
    if entry is None:
        return
    store = entry.store
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    rel = np.unique(ids[(ids >= store.lo) & (ids < store.hi)] - store.lo)
    pool = _prefetch_pool()
    for lo in range(0, rel.shape[0], PREFETCH_TASK_ROWS):
        pool.submit(store.warm, rel[lo:lo + PREFETCH_TASK_ROWS])


# ------------------------------------------------------------- EngramEmbedding hook

def _pinned(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Explicit CPU device: the model is built under a CUDA default-device context."""
    return torch.empty(shape, dtype=dtype, device=torch.device("cpu"),
                       pin_memory=torch.cuda.is_available())


def _capturing(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.is_current_stream_capturing()


def _init(self, num_embeddings: int, dim: int, layer_id: int):
    """Replacement EngramEmbedding.__init__: no table, zero-size parameters, one
    memory map of the shard, pinned + device staging of `capacity()` ids."""
    torch.nn.Module.__init__(self)
    from sglang.srt.runtime_context import get_parallel

    if dim != ROW_BYTES:
        raise RuntimeError(f"Engram layer {layer_id}: dim {dim} != {ROW_BYTES}")
    model_dir = engram_dir()
    if model_dir is None:
        raise RuntimeError("engram_staged is installed but SPARK_ENGRAM_DIR is unset")
    self.dim = dim
    self.layer_id = layer_id
    self.tp_size = get_parallel().tp_size
    tp_rank = get_parallel().tp_rank
    self.host_table = None

    shard, n, w_off, s_off = table_span(model_dir, layer_id)
    if n != num_embeddings:
        raise RuntimeError(
            f"Engram layer {layer_id}: shard {shard.name} holds {n} rows, the model expects {num_embeddings}"
        )
    ranges = row_ranges(model_dir)
    if layer_id not in ranges:
        raise RuntimeError(f"Engram layer {layer_id}: no row range in {model_dir} (have {sorted(ranges)})")
    lo, hi = ranges[layer_id]
    if not (0 <= lo <= hi <= n):
        raise RuntimeError(f"Engram layer {layer_id}: range [{lo}, {hi}) is not inside [0, {n})")
    if self.tp_size > 1:
        from sglang.srt.distributed import get_tp_group

        got: list = [None] * self.tp_size
        torch.distributed.all_gather_object(got, (lo, hi), group=get_tp_group().cpu_group)
        order = sorted(range(self.tp_size), key=lambda r: got[r][0])
        edge = 0
        tiled = True
        for r in order:
            if got[r][0] != edge:
                tiled = False
            edge = got[r][1]
        if not tiled or edge != n:
            raise RuntimeError(
                f"Engram layer {layer_id}: the TP ranks' row ranges do not tile [0, {n}): "
                + ", ".join(f"rank {r}: [{got[r][0]}, {got[r][1]})" for r in range(self.tp_size))
            )
    self.row_start = lo
    self.rows = hi - lo
    self.store = RowStore(layer_id, shard, n, w_off, s_off, lo, hi)
    STORES.append((layer_id, self.store))
    _LAYERS[layer_id] = self

    # Zero-size parameters of the right dtypes: the loader maps the checkpoint keys
    # to them by name and calls weight_loader, which only checks shapes.
    self.weight = torch.nn.Parameter(torch.empty(0, dim, dtype=torch.float8_e4m3fn), requires_grad=False)
    self.scale = torch.nn.Parameter(
        torch.empty(0, dim // BLOCK_SIZE, dtype=torch.float8_e8m0fnu), requires_grad=False
    )
    self.weight.weight_loader = self._check_loaded
    self.scale.weight_loader = self._check_loaded

    cap = capacity()
    self.capacity = cap
    self._w_host = _pinned((cap, ROW_BYTES), torch.uint8)
    self._s_host = _pinned((cap, SCALE_BYTES), torch.uint8)
    self._w_np = self._w_host.numpy()
    self._s_np = self._s_host.numpy()
    self._w_dev: Optional[torch.Tensor] = None
    self._s_dev: Optional[torch.Tensor] = None
    self._pos: Optional[torch.Tensor] = None
    self._staged = -1  # ids staged by the runner hook and not yet consumed
    self._checked_first = False
    self._warned_mismatch = False
    logger.info(
        "Engram layer %d rank %d/%d: rows [%d, %d) of %d from %s (memmap, MADV_RANDOM), "
        "%d threads, capacity %d ids, %.0f MiB pinned + %.0f MiB device staging",
        layer_id, tp_rank, self.tp_size, lo, hi, n, shard.name, threads(), cap,
        cap * (ROW_BYTES + SCALE_BYTES) / 2**20, cap * (ROW_BYTES + SCALE_BYTES) / 2**20,
    )


def _check_loaded(self, param: torch.nn.Parameter, loaded: torch.Tensor) -> None:
    """weight_loader: the checkpoint tensor must be the full [N, 256] / [N, 8] table;
    nothing is copied (the rows are read from disk on demand)."""
    width = ROW_BYTES if param is self.weight else SCALE_BYTES
    want = (self.store.n_rows, width)
    if tuple(loaded.shape) != want:
        raise RuntimeError(
            f"Engram layer {self.layer_id}: checkpoint tensor {tuple(loaded.shape)} != {want}"
        )


def _ensure_device(self, device: torch.device) -> None:
    if self._w_dev is not None:
        return
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"Engram layer {self.layer_id}: first lookup happened under CUDA-graph capture; "
            "the device staging must be created by an eager forward first"
        )
    cap = self.capacity
    # Zeroed: padding tokens of a captured graph read positions nothing staged.
    self._w_dev = torch.zeros((cap, ROW_BYTES), dtype=torch.uint8, device=device)
    self._s_dev = torch.zeros((cap, SCALE_BYTES), dtype=torch.uint8, device=device)
    self._pos = torch.arange(cap, dtype=torch.int64, device=device)


def stage_many(requests: list, device: torch.device) -> None:
    """requests = [(EngramEmbedding, ids int64 numpy [n])]: gather every layer's rows
    in one parallel batch into pinned staging, then async H2D per layer."""
    t0 = time.perf_counter()
    plan = []
    for layer, ids in requests:
        n = int(ids.shape[0])
        layer._staged = -1
        if n > layer.capacity:
            raise RuntimeError(
                f"Engram layer {layer.layer_id}: lookup of {n} ids exceeds SPARK_ENGRAM_MAX_IDS={layer.capacity}"
            )
        layer._ensure_device(device)
        plan.append((layer, ids, n))
    masks = gather_many([(layer.store, ids, layer._w_np[:n], layer._s_np[:n]) for layer, ids, n in plan if n])
    masks = iter(masks)
    for layer, ids, n in plan:
        if n:
            owned = next(masks)
            layer._w_dev[:n].copy_(layer._w_host[:n], non_blocking=True)
            layer._s_dev[:n].copy_(layer._s_host[:n], non_blocking=True)
            got = int(owned.sum())
            layer.store.stat_owned += got
            layer.store.stat_zeroed += n - got
        layer._staged = n
        layer.store.stat_calls += 1
    dt = time.perf_counter() - t0
    for layer, ids, n in plan:
        layer.store.stat_seconds += dt


def _stage(self, ids: np.ndarray, device: torch.device) -> None:
    """Stage `ids` (global row ids, int64, in lookup order) for the next lookup."""
    stage_many([(self, np.asarray(ids, dtype=np.int64).reshape(-1))], device)


def _stage_inline(self, indices: torch.Tensor) -> None:
    """In-forward staging (port of _disk_lookup): the graph-capture warm-ups, and
    any eager lookup the runner hook did not serve (CP / DP-attention gathers)."""
    ids = indices.detach().reshape(-1).to(device="cpu", dtype=torch.int64).numpy()
    stage_many([(self, ids)], indices.device)
    self.store.stat_inline += 1


def _owned_rows(self, indices: torch.Tensor) -> torch.Tensor:
    """Replacement EngramEmbedding._owned_rows: this rank's rows of `indices`
    dequantized to bf16, zeros for the rest -- read from the staged device rows by
    position, never from a table."""
    n = indices.numel()
    out = self._empty(indices)
    if n == 0:
        return out
    if n > self.capacity:
        raise RuntimeError(
            f"Engram layer {self.layer_id}: lookup of {n} ids exceeds SPARK_ENGRAM_MAX_IDS={self.capacity}"
        )
    self._ensure_device(indices.device)
    if self._staged != n:
        if _capturing(indices):
            if not _RUNNER_HOOKED:
                raise RuntimeError(
                    f"Engram layer {self.layer_id}: CUDA-graph capture reached the lookup but "
                    "install_runner() was never called; replays would read rows nobody staged"
                )
            # Warm-ups staged this batch in-forward; the recorded gather reads the
            # persistent staging that the runner hook refills before every replay.
        else:
            if _RUNNER_HOOKED and self._staged >= 0 and not self._warned_mismatch:
                self._warned_mismatch = True
                logger.warning(
                    "Engram layer %d: lookup of %d ids but the runner hook staged %d; "
                    "staging in-forward instead (slow path)", self.layer_id, n, self._staged,
                )
            self._stage_inline(indices)
    elif not self._checked_first and not _capturing(indices):
        self._checked_first = True
        logger.info("Engram layer %d: first eager lookup, %d ids staged == %d looked up",
                    self.layer_id, self._staged, n)
    self._staged = -1
    if indices.is_cuda and torch.version.cuda is not None:
        _ENGRAM_MOD.engram_gather(
            self._w_dev.data_ptr(), self._s_dev.data_ptr(), self._pos[:n], out.view(-1, self.dim),
            self.dim, BLOCK_SIZE, row_lo=0, row_hi=self.capacity,
        )
        return out
    # Non-CUDA fallback (CPU tests): the kernel's arithmetic in torch.
    vals = self._w_dev[:n].view(torch.float8_e4m3fn).float().view(n, SCALE_BYTES, BLOCK_SIZE)
    exps = self._s_dev[:n].to(torch.int32)
    scale = (exps << 23).view(torch.float32)
    scale = torch.where(exps == 0, torch.full_like(scale, 2.0 ** -127), scale)
    out.view(-1, self.dim).copy_((vals * scale[:, :, None]).reshape(n, self.dim).to(torch.bfloat16))
    return out


_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_CACHE = {"releases": 0, "peak": 0}


def _cache_loop(budget: int, period: float) -> None:
    """Keep the row spans' page cache under the budget (the reason is in engram_store's docstring)."""
    while True:
        time.sleep(period)
        try:
            stores = [store for _, store in list(STORES)]
            resident = sum(store.resident() for store in stores)
            _CACHE["peak"] = max(_CACHE["peak"], resident)
            if resident > budget:
                for store in stores:
                    store.release()
                _CACHE["releases"] += 1
        except Exception as exc:  # noqa: BLE001 - the governor must never take the server down
            logger.warning("Engram page-cache governor stopped: %s", exc)
            return


def _stats_loop(period: float) -> None:
    last = {}
    while True:
        time.sleep(period)
        logger.info("Engram page cache: peak %.2f GB since start, %d releases", _CACHE["peak"] / (1 << 30), _CACHE["releases"])
        for layer_id, store in list(STORES):
            prev = last.get(layer_id, (0, 0, 0, 0, 0.0))
            cur = (store.stat_calls, store.stat_inline, store.stat_owned, store.stat_zeroed, store.stat_seconds)
            calls = cur[0] - prev[0]
            if calls:
                logger.info(
                    "Engram layer %d: %d stagings (%d in-forward), %d owned rows, %d zeroed rows, "
                    "mean stage %.2f ms", layer_id, calls, cur[1] - prev[1], cur[2] - prev[2],
                    cur[3] - prev[3], 1e3 * (cur[4] - prev[4]) / calls,
                )
            last[layer_id] = cur


def _wrap_hasher(hasher_cls) -> None:
    """After EngramHasher.__init__: hand engram_prefetch CPU copies of the hash tables
    so the scheduler can hash the next chunk without the GPU."""
    stock = hasher_cls.__init__

    def __init__(self, layout, *args, **kwargs):
        stock(self, layout, *args, **kwargs)
        try:
            import engram_prefetch
        except ImportError:
            return
        engram_prefetch.HASHER.update(
            pad_id=int(self.pad_id),
            ngram=int(self.max_ngram_size),
            layer_ids=[int(x) for x in layout.layer_ids],
            token_map=self.token_map.detach().cpu().clone(),
            multipliers=self.multipliers.detach().cpu().clone(),
            primes=self.primes.detach().cpu().clone(),
            offsets=self.offsets.detach().cpu().clone(),
            obj=self,                     # image_token_id is set after __init__ (EngramHasher.from_config)
        )

    __init__.__wrapped__ = stock
    hasher_cls.__init__ = __init__


def install(module) -> None:
    """Hook the executed sglang.srt.layers.engram module (called once by sitecustomize)."""
    global _ENGRAM_MOD, _STATS_THREAD
    if engram_dir() is None:
        logger.info("engram_staged: SPARK_ENGRAM_DIR unset, stock Engram tables stay")
        return
    _ENGRAM_MOD = module
    cls = module.EngramEmbedding
    if getattr(cls, "_spark_installed", False):
        return
    cls._spark_installed = True
    cls.__init__ = _init
    cls._owned_rows = _owned_rows
    cls._check_loaded = _check_loaded
    cls._ensure_device = _ensure_device
    cls._stage_inline = _stage_inline
    cls.stage = _stage
    _wrap_hasher(module.EngramHasher)
    period = _env_int("SPARK_ENGRAM_STATS_SECONDS", 60)
    if period > 0 and _STATS_THREAD is None:
        _STATS_THREAD = threading.Thread(target=_stats_loop, args=(float(period),),
                                         daemon=True, name="engram-stats")
        _STATS_THREAD.start()
    budget = int(float(os.environ.get("SPARK_ENGRAM_CACHE_GB", "4")) * (1 << 30))
    cache_period = float(os.environ.get("SPARK_ENGRAM_CACHE_SECONDS", "5"))
    if budget > 0 and cache_period > 0 and not _CACHE.get("started"):
        _CACHE["started"] = True
        threading.Thread(target=_cache_loop, args=(budget, cache_period), daemon=True, name="engram-cache").start()
        logger.info("Engram page-cache budget %.1f GB, checked every %.0f s", budget / (1 << 30), cache_period)
    logger.info("Engram rows from %s: EngramEmbedding reads staged rows by position (threads %d, capacity %d)",
                engram_dir(), threads(), capacity())


# ---------------------------------------------------------------- ModelRunner hook

_IDS_HOST: Optional[torch.Tensor] = None
_IDS_READY: Optional["torch.cuda.Event"] = None


def _ids_host(numel: int) -> torch.Tensor:
    global _IDS_HOST
    if _IDS_HOST is None or _IDS_HOST.numel() < numel:
        _IDS_HOST = _pinned((max(numel, 1),), torch.int64)
    return _IDS_HOST[:numel]


def hash_ids_no_commit(hasher, input_ids: torch.Tensor, forward_batch) -> torch.Tensor:
    """EngramHasher.forward's dispatch and kernel, minus its history commit.

    The model's own call commits exactly once per step, eagerly or inside the
    captured graph; a second commit from here would shift every request's
    predecessors by one token. Returns [T, n_layers, n_cols] int64 on input_ids'
    device."""
    eng = _ENGRAM_MOD
    mode = forward_batch.forward_mode
    req_slots = forward_batch.req_pool_indices.to(torch.int64)
    bs = req_slots.shape[0]
    num_tokens = input_ids.shape[0]
    device = input_ids.device
    num_real, block, row, starts = num_tokens, 1, None, None
    history, via_slots = hasher.history, True
    if mode.is_decode():
        kmode = eng.MODE_DECODE
    elif mode.is_target_verify():
        block = int(forward_batch.spec_info.draft_token_num)
        if num_tokens != bs * block:
            raise RuntimeError(
                f"Engram staging: target-verify expects {bs} x {block} tokens, got {num_tokens}"
            )
        kmode = eng.MODE_VERIFY
    else:
        lens = forward_batch.extend_seq_lens.to(torch.int64)
        starts = forward_batch.extend_start_loc.to(torch.int64)
        row = torch.repeat_interleave(torch.arange(bs, device=device), lens)
        num_real = row.shape[0]
        kmode = eng.MODE_EXTEND
        if forward_batch.ngram_history is not None:
            history, via_slots = forward_batch.ngram_history, False
    if input_ids.is_cuda and torch.version.cuda is not None:
        hash_ids, _ = eng.engram_hash_ids(
            input_ids, forward_batch.positions, mode=kmode, history=history,
            token_map=hasher.token_map, multipliers=hasher.multipliers, primes=hasher.primes,
            offsets=hasher.offsets, pad_id=hasher.pad_id, num_real=num_real,
            req_slots=req_slots if via_slots else None, block=block, row=row, starts=starts,
            image_token_id=hasher.image_token_id, mm_pad_shift=eng.MM_PAD_SHIFT_VALUE,
        )
    else:
        hash_ids, _ = hasher._torch_hash_ids(
            input_ids, forward_batch.positions, kmode,
            history[req_slots] if via_slots else history, num_real, block, row, starts,
        )
    return hash_ids


def _find_engram(model) -> tuple:
    """(hasher, [(hash column index, EngramEmbedding), ...]) of a model, or ()."""
    eng = _ENGRAM_MOD
    hasher, layers = None, []
    for m in model.modules():
        if isinstance(m, eng.EngramHasher):
            hasher = m
        elif isinstance(m, eng.Engram) and hasattr(m.embed, "store"):
            layers.append((int(m.layer_hash_index), m.embed))
    if hasher is None or not layers:
        return ()
    return hasher, layers


def _cp_extend(forward_batch) -> bool:
    """Prefill under context parallelism v2: the model hashes the whole prompt and
    the lookup all-gathers the ids over the CP group, so the staged order would not
    be the lookup order. Not our launch configuration; served in-forward instead."""
    if not forward_batch.forward_mode.is_extend():
        return False
    try:
        from sglang.srt.layers.cp.utils import is_cp_v2_active
    except ImportError:
        return False
    return bool(is_cp_v2_active(forward_batch))


def stage_batch(runner, forward_batch) -> None:
    """The pre-forward staging of one step (see the module docstring)."""
    plan = runner.__dict__.get("_spark_engram")
    if plan is None:
        plan = _find_engram(runner.model)
        runner._spark_engram = plan
    if not plan:
        return
    hasher, layers = plan
    for _, layer in layers:
        layer._staged = -1
    ids = forward_batch.input_ids
    mode = forward_batch.forward_mode
    if ids is None or ids.shape[0] == 0 or mode.is_idle() or hasher.history is None:
        return
    if not (mode.is_decode() or mode.is_target_verify() or mode.is_extend()):
        return  # a mode the hasher does not serve; the model decides what to do
    if _cp_extend(forward_batch):
        return  # the lookup all-gathers ids over CP: served in-forward
    hash_ids = hash_ids_no_commit(hasher, ids, forward_batch)
    T, L, C = hash_ids.shape
    host = _ids_host(T * L * C).view(T, L, C)
    if hash_ids.is_cuda:
        global _IDS_READY
        if _IDS_READY is None:
            _IDS_READY = torch.cuda.Event()
        host.copy_(hash_ids, non_blocking=True)
        _IDS_READY.record()
        # The one host sync of the step. It also orders this step's rewrite of the
        # pinned staging after the previous step's async H2D copies from it.
        _IDS_READY.synchronize()
    else:
        host.copy_(hash_ids)
    arr = host.numpy()
    stage_many([(layer, np.ascontiguousarray(arr[:, col, :]).reshape(-1)) for col, layer in layers],
               ids.device)


def install_runner(module) -> None:
    """Hook sglang.srt.model_executor.model_runner.ModelRunner.forward."""
    global _RUNNER_HOOKED
    if engram_dir() is None:
        return
    if _ENGRAM_MOD is None:
        import sglang.srt.layers.engram as eng

        install(eng)
    if _RUNNER_HOOKED:
        return
    cls = module.ModelRunner
    stock = cls.forward

    def forward(self, forward_batch, *args, **kwargs):
        stage_batch(self, forward_batch)
        return stock(self, forward_batch, *args, **kwargs)

    forward.__wrapped__ = stock
    cls.forward = forward
    _RUNNER_HOOKED = True
    logger.info("Engram rows staged before ModelRunner.forward (hash on GPU, one host sync, memmap gather)")
