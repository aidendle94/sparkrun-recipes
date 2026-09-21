"""Engram rows for DeepSeek-V4.1 on DGX Spark, gathered per lookup from node-local NVMe.

Why this hook exists. SGLang's ``EngramEmbedding`` keeps one layer's Engram hash table -- about 384 million
rows of 256 fp8 e4m3 bytes plus 8 e8m0 block scales, roughly 101 GB per layer, two layers (1 and 14) --
either sharded over the TP ranks in device memory or pinned in host memory. A DGX Spark (GB10) has 128 GB of
unified memory that is host memory and GPU memory at once, so even one rank's quarter of both tables
(~50 GB) leaves no room for the rest of the model and a usable KV pool. This module changes only where the
rows come from: each rank owns one contiguous global row range and every lookup fetches the rows it owns
from a memory-mapped safetensors shard on local NVMe through the C library ``engram_rows.c``, run as a
host-function node inside the CUDA graph. Rows the rank does not own come out as zeros and SGLang's TP
all-reduce adds the ranks up exactly as it does for the stock device shard. The hashing, the all-reduce,
the gate and the projections stay SGLang's.

A lookup does the same four steps eagerly and under CUDA-graph capture, with no Python on the replay path:
an async D2H copy of the ids into a pinned buffer, a ``cudaLaunchHostFunc`` node that runs the C gather
into pinned staging, async H2D copies into device staging, and SGLang's Triton ``engram_gather``
dequantizing the staged rows by position. Every buffer, the ctypes ``Work`` structs and the device staging
are allocated once per layer and kept for the life of the process, because captured graphs keep pointing
at them. The pinned buffers are created with an explicit CPU device: the model is built under a CUDA
default-device context, where a plain ``torch.empty`` would land on the GPU.

Environment (SPARK_ENGRAM_DIR is the master switch: nothing here is installed without it):
  SPARK_ENGRAM_DIR            directory holding model.safetensors.index.json and the shards with the two
                              tables: a sparse node-local copy from tools/engram_local.py, whose
                              engram-local.json gives this rank's row range per layer, or a full snapshot.
  SPARK_ENGRAM_ROWS           "1:lo:hi,14:lo:hi": this rank's global row range per layer when the directory
                              has no engram-local.json (a full checkpoint on the node that holds it).
  SPARK_ENGRAM_THREADS        worker threads of the C gather that fault cold rows in (default 64; NVMe
                              wants queue depth).
  SPARK_ENGRAM_MAX_IDS        ids one lookup may carry (default 262144); sizes the pinned and device staging.
                              Chunked-prefill tokens times hash columns per layer must fit.
  SPARK_ENGRAM_STATS_SECONDS  interval of the per-layer gather statistics log line (default 60; 0 disables).

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import json
import logging
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = ["LIB", "STORES", "Work", "install", "row_ranges", "table_span"]

ENV_DIR = "SPARK_ENGRAM_DIR"
ENV_ROWS = "SPARK_ENGRAM_ROWS"
ENV_THREADS = "SPARK_ENGRAM_THREADS"
ENV_MAX_IDS = "SPARK_ENGRAM_MAX_IDS"
ENV_STATS = "SPARK_ENGRAM_STATS_SECONDS"

MANIFEST_NAME = "engram-local.json"
INDEX_NAME = "model.safetensors.index.json"
WEIGHT_DTYPE = "F8_E4M3"
SCALE_DTYPE = "F8_E8M0"
ROW_BYTES = 256  # fp8 payload bytes per row: the table dim, fixed by the C library
SCALE_BYTES = 8  # e8m0 exponents per row: one per 32-value block
ABI_VERSION = 2
DEFAULT_THREADS = 64
DEFAULT_MAX_IDS = 262144
DEFAULT_STATS_SECONDS = 60.0

# cudaMemcpyKind values of the CUDA runtime API.
_MEMCPY_H2D = 1
_MEMCPY_D2H = 2


class Work(ctypes.Structure):
    """Mirror of the C ``Work`` struct: what one host-function node hands to ``engram_rows_gather``.

    One instance per (layer, id count) lives for the whole process: a captured graph replays with the
    pointer it was given, and the C side reads ``count`` from that memory at run time."""

    _fields_ = [
        ("count", ctypes.c_uint64),
        ("store", ctypes.c_void_p),
        ("w_out", ctypes.c_void_p),
        ("s_out", ctypes.c_void_p),
        ("ids", ctypes.c_void_p),
    ]


def _load_library() -> ctypes.CDLL:
    """Load overlay/libengram_rows.so from next to this module and declare its ABI."""
    path = Path(__file__).resolve().with_name("libengram_rows.so")
    if not path.is_file():
        raise ImportError(
            f"{path} is missing; build it with: gcc -O2 -std=gnu11 -shared -fPIC -pthread "
            f"-o {path} {path.with_name('engram_rows.c')}"
        )
    lib = ctypes.CDLL(str(path), use_errno=True)
    u64 = ctypes.c_uint64
    lib.engram_rows_abi_version.argtypes = []
    lib.engram_rows_abi_version.restype = ctypes.c_int
    lib.engram_rows_open.argtypes = [ctypes.c_char_p, u64, u64, u64, u64, ctypes.c_int]
    lib.engram_rows_open.restype = ctypes.c_void_p
    lib.engram_rows_gather.argtypes = [ctypes.c_void_p]
    lib.engram_rows_gather.restype = None
    lib.engram_rows_prefetch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), u64]
    lib.engram_rows_prefetch.restype = None
    lib.engram_rows_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(u64)]
    lib.engram_rows_stats.restype = None
    lib.engram_rows_close.argtypes = [ctypes.c_void_p]
    lib.engram_rows_close.restype = None
    version = lib.engram_rows_abi_version()
    if version != ABI_VERSION:
        raise ImportError(f"{path} reports ABI version {version}; this module needs {ABI_VERSION} (rebuild it)")
    return lib


# The ctypes handle of the C library and the open stores as (layer_id, Store*), consumed by
# engram_prefetch to warm the next prefill chunk's rows from the scheduler.
LIB: ctypes.CDLL = _load_library()
STORES: list[tuple[int, int]] = []

# Address of the C gather entry point, the function pointer every host node is launched with.
_GATHER_FN: int = ctypes.cast(LIB.engram_rows_gather, ctypes.c_void_p).value

# Everything a lookup keeps alive for the process lifetime, in construction order.
_ROW_STORES: list["_RowStore"] = []

# The executed sglang.srt.layers.engram module and the pieces borrowed from it; set by install().
_SGL: Any = None


# ----------------------------------------------------------------------------------------------------------
# Row source: where this rank's rows are, and which they are.
# ----------------------------------------------------------------------------------------------------------


def _parse_rows_env(text: str) -> dict[int, tuple[int, int]]:
    """``"1:lo:hi,14:lo:hi"`` -> {layer: (lo, hi)}."""
    ranges: dict[int, tuple[int, int]] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise RuntimeError(f"{ENV_ROWS}: expected LAYER:LO:HI entries separated by commas, got {item!r}")
        try:
            layer, lo, hi = (int(p) for p in parts)
        except ValueError:
            raise RuntimeError(f"{ENV_ROWS}: {item!r} is not three integers LAYER:LO:HI") from None
        ranges[layer] = (lo, hi)
    if not ranges:
        raise RuntimeError(f"{ENV_ROWS} is set but names no LAYER:LO:HI range")
    return ranges


def row_ranges(model_dir: Path) -> dict[int, tuple[int, int]]:
    """This rank's global row range per Engram layer, {layer_id: (lo, hi)}.

    A sparse node-local copy carries its ranges in engram-local.json (only those rows exist on disk, so the
    manifest is authoritative); a full snapshot has no manifest and takes them from SPARK_ENGRAM_ROWS."""
    model_dir = Path(model_dir)
    manifest = model_dir / MANIFEST_NAME
    env_rows = os.environ.get(ENV_ROWS, "").strip()
    if manifest.is_file():
        with open(manifest) as f:
            doc = json.load(f)
        layers = doc.get("layers") if isinstance(doc, dict) else None
        if not isinstance(layers, dict) or not layers:
            raise RuntimeError(f"{manifest} has no 'layers' map of LAYER -> [lo, hi]")
        ranges: dict[int, tuple[int, int]] = {}
        for key, span in layers.items():
            try:
                lo, hi = (int(x) for x in span)
                ranges[int(key)] = (lo, hi)
            except (TypeError, ValueError):
                raise RuntimeError(f"{manifest}: layer {key!r} has a malformed range {span!r}") from None
        if env_rows and _parse_rows_env(env_rows) != ranges:
            logger.warning(
                "%s=%s disagrees with %s; the manifest wins because only its rows exist in the sparse copy",
                ENV_ROWS, env_rows, manifest,
            )
        return ranges
    if env_rows:
        return _parse_rows_env(env_rows)
    raise RuntimeError(
        f"{model_dir} has no {MANIFEST_NAME} (tools/engram_local.py writes it for a sparse copy) and "
        f"{ENV_ROWS} is unset (a full checkpoint needs it: 'LAYER:LO:HI,...')"
    )


def _read_header(shard: Path) -> tuple[int, dict]:
    """(byte offset of the data block, header dict) of a safetensors file: 8-byte LE length, then JSON."""
    with open(shard, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"{shard}: not a safetensors file (no 8-byte header length)")
        (hlen,) = struct.unpack("<Q", raw)
        body = f.read(hlen)
    if len(body) != hlen:
        raise RuntimeError(f"{shard}: truncated safetensors header ({len(body)} of {hlen} bytes)")
    try:
        header = json.loads(body)
    except ValueError as exc:
        raise RuntimeError(f"{shard}: safetensors header is not JSON: {exc}") from None
    return 8 + hlen, header


_INDEX_CACHE: dict[Path, dict] = {}


def _weight_map(model_dir: Path) -> dict:
    """The index's tensor name -> shard file map, parsed once per process (it is several MB)."""
    index = model_dir / INDEX_NAME
    cached = _INDEX_CACHE.get(index)
    if cached is None:
        if not index.is_file():
            raise RuntimeError(f"{index} is missing; {ENV_DIR} must name a checkpoint directory or a sparse copy")
        with open(index) as f:
            doc = json.load(f)
        cached = doc.get("weight_map") if isinstance(doc, dict) else None
        if not isinstance(cached, dict):
            raise RuntimeError(f"{index} has no 'weight_map'")
        _INDEX_CACHE[index] = cached
    return cached


def table_span(model_dir: Path, layer_id: int) -> tuple[Path, int, int, int]:
    """(shard path, N, weight byte offset, scale byte offset) of one layer's Engram table.

    The offsets are of row 0 in the shard file: 8 + header length + data_offsets[0]. The weight and the
    scale must be in the same shard, because one mapping serves both."""
    model_dir = Path(model_dir)
    weight_map = _weight_map(model_dir)
    w_name = f"layers.{layer_id}.engram.embed.weight"
    s_name = f"layers.{layer_id}.engram.embed.scale"
    for name in (w_name, s_name):
        if name not in weight_map:
            raise RuntimeError(
                f"{model_dir / INDEX_NAME} has no entry for {name}; is layer {layer_id} an Engram layer of "
                "this checkpoint?"
            )
    if weight_map[w_name] != weight_map[s_name]:
        raise RuntimeError(
            f"engram layer {layer_id}: weight is in {weight_map[w_name]} but scale in {weight_map[s_name]}; "
            "both must be in one shard"
        )
    shard = model_dir / weight_map[w_name]
    if not shard.is_file():
        raise RuntimeError(f"engram layer {layer_id}: shard {shard} is missing")
    data_start, header = _read_header(shard)

    def entry(name: str, dtype: str, cols: int) -> tuple[int, int]:
        info = header.get(name)
        if not isinstance(info, dict):
            raise RuntimeError(f"{shard.name}: header has no tensor {name}")
        shape = list(info.get("shape", []))
        if info.get("dtype") != dtype:
            raise RuntimeError(f"{shard.name}: {name} is {info.get('dtype')}, expected {dtype}")
        if len(shape) != 2 or shape[1] != cols:
            raise RuntimeError(f"{shard.name}: {name} has shape {shape}, expected [N, {cols}]")
        offsets = info.get("data_offsets")
        if not (isinstance(offsets, list) and len(offsets) == 2):
            raise RuntimeError(f"{shard.name}: {name} has no data_offsets")
        if offsets[1] - offsets[0] != shape[0] * cols:
            raise RuntimeError(f"{shard.name}: {name} data_offsets {offsets} do not span {shape[0]} x {cols} bytes")
        return int(shape[0]), data_start + int(offsets[0])

    n_w, w_off = entry(w_name, WEIGHT_DTYPE, ROW_BYTES)
    n_s, s_off = entry(s_name, SCALE_DTYPE, SCALE_BYTES)
    if n_w != n_s:
        raise RuntimeError(f"{shard.name}: {w_name} has {n_w} rows but {s_name} has {n_s}")
    return shard, n_w, w_off, s_off


def _model_dir() -> Path:
    value = os.environ.get(ENV_DIR, "").strip()
    if not value:
        raise RuntimeError(f"{ENV_DIR} is not set; it must name the directory with the Engram shards")
    model_dir = Path(value)
    if not model_dir.is_dir():
        raise RuntimeError(f"{ENV_DIR}={value} is not a directory")
    return model_dir


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name}={raw!r} is not an integer") from None
    if value < minimum:
        raise RuntimeError(f"{name}={value} must be at least {minimum}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"{name}={raw!r} is not a number") from None


# ----------------------------------------------------------------------------------------------------------
# cudart: the two runtime calls torch does not expose, host-function launch and raw async memcpy.
# ----------------------------------------------------------------------------------------------------------


class _Cudart:
    """ctypes view of libcudart: ``cudaLaunchHostFunc`` and ``cudaMemcpyAsync`` on a torch stream handle.

    Torch's bundled runtime is tried first so the handle is the instance torch already loaded; the
    toolkit's copy and the loader's search path follow. The memcpys go through here rather than
    ``Tensor.copy_`` so the capture path holds nothing but memcpy nodes on buffers that are never freed:
    no allocator event bookkeeping inside the graph."""

    def __init__(self) -> None:
        failures: list[str] = []
        lib = None
        for candidate in self._candidates():
            try:
                handle = ctypes.CDLL(candidate)
            except OSError as exc:
                failures.append(f"{candidate}: {exc}")
                continue
            if not hasattr(handle, "cudaLaunchHostFunc") or not hasattr(handle, "cudaMemcpyAsync"):
                failures.append(f"{candidate}: no cudaLaunchHostFunc/cudaMemcpyAsync symbol")
                continue
            lib, self.path = handle, candidate
            break
        if lib is None:
            raise RuntimeError("libcudart.so not found for the Engram host node; tried " + "; ".join(failures))
        void_p = ctypes.c_void_p
        self._launch_host_func = lib.cudaLaunchHostFunc
        self._launch_host_func.argtypes = [void_p, void_p, void_p]
        self._launch_host_func.restype = ctypes.c_int
        self._memcpy_async = lib.cudaMemcpyAsync
        self._memcpy_async.argtypes = [void_p, void_p, ctypes.c_size_t, ctypes.c_int, void_p]
        self._memcpy_async.restype = ctypes.c_int
        self._error_string = lib.cudaGetErrorString
        self._error_string.argtypes = [ctypes.c_int]
        self._error_string.restype = ctypes.c_char_p

    @staticmethod
    def _candidates() -> list[str]:
        site = Path(torch.__file__).resolve().parent.parent
        dirs = [site / "nvidia" / "cuda_runtime" / "lib"]
        dirs += sorted(site.glob("nvidia/cu*/lib"))  # CUDA 13 wheels bundle the runtime as nvidia/cu13
        dirs.append(Path("/usr/local/cuda/lib64"))
        found: list[str] = []
        for d in dirs:
            if d.is_dir():
                found += sorted(str(p) for p in d.glob("libcudart.so*"))
        name = ctypes.util.find_library("cudart")
        if name:
            found.append(name)
        return found

    def _check(self, what: str, rc: int) -> None:
        if rc != 0:
            text = self._error_string(rc)
            raise RuntimeError(f"{what} failed: CUDA error {rc} ({text.decode() if text else 'unknown'})")

    def launch_host_func(self, stream: int, fn: int, user_data: int) -> None:
        self._check("cudaLaunchHostFunc", self._launch_host_func(stream, fn, user_data))

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
        self._check("cudaMemcpyAsync", self._memcpy_async(dst, src, nbytes, kind, stream))


_CUDART: _Cudart | None = None
_CUDART_LOCK = threading.Lock()


def _cudart() -> _Cudart:
    global _CUDART
    if _CUDART is None:
        with _CUDART_LOCK:
            if _CUDART is None:
                _CUDART = _Cudart()
                logger.info("engram_store: CUDA runtime for the host node is %s", _CUDART.path)
    return _CUDART


# ----------------------------------------------------------------------------------------------------------
# One layer's rank-local row store and the fixed buffers its lookups go through.
# ----------------------------------------------------------------------------------------------------------


class _RowStore:
    """The open C store of one layer plus every buffer a lookup touches, allocated once, never freed."""

    def __init__(
        self,
        layer_id: int,
        shard: Path,
        num_rows: int,
        weight_off: int,
        scale_off: int,
        lo: int,
        hi: int,
        threads: int,
        capacity: int,
    ) -> None:
        self.layer_id = layer_id
        self.shard = shard
        self.num_rows = num_rows
        self.lo, self.hi = lo, hi
        self.threads = threads
        self.capacity = capacity
        ctypes.set_errno(0)
        ptr = LIB.engram_rows_open(str(shard).encode(), weight_off, scale_off, lo, hi, threads)
        if not ptr:
            err = ctypes.get_errno()
            raise RuntimeError(
                f"engram layer {layer_id}: engram_rows_open({shard}) failed"
                + (f": {os.strerror(err)}" if err else " (open, fstat or mmap of the shard)")
            )
        self.ptr: int = ptr
        cpu = torch.device("cpu")
        self.ids_host = torch.empty(capacity, dtype=torch.int64, device=cpu, pin_memory=True)
        self.w_host = torch.empty((capacity, ROW_BYTES), dtype=torch.uint8, device=cpu, pin_memory=True)
        self.s_host = torch.empty((capacity, SCALE_BYTES), dtype=torch.uint8, device=cpu, pin_memory=True)
        self.works: dict[int, Work] = {}
        self.w_dev: torch.Tensor | None = None
        self.s_dev: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self._last_stats = (0, 0, 0, 0)

    def work_for(self, count: int) -> Work:
        """The Work struct for a lookup of ``count`` ids; graphs of that size replay through it."""
        work = self.works.get(count)
        if work is None:
            work = Work(count=count, store=self.ptr, w_out=self.w_host.data_ptr(), s_out=self.s_host.data_ptr(),
                        ids=self.ids_host.data_ptr())
            self.works[count] = work
        return work

    def device_staging(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Device staging for the gathered rows and the 0..capacity position ids, made on the first eager
        call: a graph replay must find them at the addresses it captured."""
        if self.w_dev is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"engram layer {self.layer_id}: the first lookup ran under CUDA-graph capture; the device "
                    "staging must be created by an eager (warm-up) call before any capture"
                )
            self.w_dev = torch.empty((self.capacity, ROW_BYTES), dtype=torch.uint8, device=device)
            self.s_dev = torch.empty((self.capacity, SCALE_BYTES), dtype=torch.uint8, device=device)
            self.positions = torch.arange(self.capacity, dtype=torch.int64, device=device)
        return self.w_dev, self.s_dev, self.positions

    def stats(self) -> tuple[int, int, int, int]:
        """Cumulative (calls, rows owned, rows zeroed, nanoseconds) of the C gather."""
        out = (ctypes.c_uint64 * 4)()
        LIB.engram_rows_stats(self.ptr, out)
        return int(out[0]), int(out[1]), int(out[2]), int(out[3])

    def log_stats(self) -> None:
        now = self.stats()
        calls, owned, zeroed, ns = (a - b for a, b in zip(now, self._last_stats))
        self._last_stats = now
        mean_us = ns / calls / 1000.0 if calls else 0.0
        logger.info(
            "engram layer %d: %d lookups, %d owned rows, %d zeroed rows, %.0f us mean gather per lookup",
            self.layer_id, calls, owned, zeroed, mean_us,
        )


_STATS_STARTED = False


def _start_stats_thread() -> None:
    """One daemon thread per process logs every store's gather statistics at the configured interval."""
    global _STATS_STARTED
    if _STATS_STARTED:
        return
    _STATS_STARTED = True
    interval = _env_float(ENV_STATS, DEFAULT_STATS_SECONDS)
    if interval <= 0:
        return

    def loop() -> None:
        while True:
            time.sleep(interval)
            for store in list(_ROW_STORES):
                try:
                    store.log_stats()
                except Exception as exc:  # noqa: BLE001 - statistics must never take the server down
                    logger.warning("engram stats thread stopped: %s", exc)
                    return

    threading.Thread(target=loop, daemon=True, name="spark-engram-stats").start()


# ----------------------------------------------------------------------------------------------------------
# Replacements for EngramEmbedding.__init__ and _owned_rows.
# ----------------------------------------------------------------------------------------------------------


def _local_span(layer_id: int, num_embeddings: int, model_dir: Path) -> tuple[Path, int, int, int, int, int]:
    """(shard, N, weight_off, scale_off, lo, hi) after every check this rank can do on its own."""
    ranges = row_ranges(model_dir)
    if layer_id not in ranges:
        raise RuntimeError(
            f"engram layer {layer_id}: no row range for it in {model_dir / MANIFEST_NAME} or {ENV_ROWS} "
            f"(ranges exist for layers {sorted(ranges)})"
        )
    lo, hi = ranges[layer_id]
    shard, num_rows, w_off, s_off = table_span(model_dir, layer_id)
    if num_rows != num_embeddings:
        raise RuntimeError(
            f"engram layer {layer_id}: {shard.name} holds {num_rows} rows but the model config says "
            f"{num_embeddings}; the shards do not belong to this checkpoint"
        )
    if not 0 <= lo <= hi <= num_rows:
        raise RuntimeError(f"engram layer {layer_id}: row range [{lo}, {hi}) is outside [0, {num_rows}]")
    return shard, num_rows, w_off, s_off, lo, hi


def _check_tiling(
    layer_id: int, span: tuple[int, int] | None, error: str | None, num_rows: int, tp_size: int
) -> tuple[int, int]:
    """Exchange (lo, hi) over the TP group and demand that the ranges tile [0, N) with no gap or overlap.

    A rank whose local checks failed sends its error instead of a range, so every rank stops with the same
    message instead of the healthy ranks waiting on a collective the failed rank will never join."""
    reports: list = []
    if tp_size <= 1:
        if error is not None:
            raise RuntimeError(error)
        reports = [(span, None)]
    else:
        from sglang.srt.distributed import get_tp_group

        group = get_tp_group().cpu_group
        reports = [None] * torch.distributed.get_world_size(group=group)
        torch.distributed.all_gather_object(reports, (span, error), group=group)
    failed = [f"rank {r}: {err}" for r, (_, err) in enumerate(reports) if err is not None]
    if failed:
        raise RuntimeError(f"engram layer {layer_id}: " + "; ".join(failed))
    spans = [s for s, _ in reports]
    cursor = 0
    tiled = True
    for lo, hi in sorted(spans):
        if lo != cursor:
            tiled = False
            break
        cursor = hi
    if not tiled or cursor != num_rows:
        listing = ", ".join(f"rank {r}: [{lo}, {hi})" for r, (lo, hi) in enumerate(spans))
        raise RuntimeError(
            f"engram layer {layer_id}: the TP ranks' row ranges do not tile [0, {num_rows}) exactly "
            f"(gap, overlap or missing rows): {listing}"
        )
    return span


def _shape_check_loader(layer_id: int, what: str, expect: tuple[int, int]):
    """The parameter's ``weight_loader``: the checkpoint tensor must have the table's shape; nothing is
    copied, the rows are read from the shard at lookup time."""

    def weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        got = tuple(int(x) for x in loaded_weight.shape)
        if got != expect:
            raise RuntimeError(
                f"engram layer {layer_id}: checkpoint {what} has shape {list(got)}, expected {list(expect)}"
            )

    return weight_loader


def embedding_init(self, num_embeddings: int, dim: int, layer_id: int) -> None:
    """Replacement ``EngramEmbedding.__init__``: no table in memory, one open row store per layer.

    Keeps every attribute the rest of the class reads (dim, tp_size, row_start, rows, host_table, weight,
    scale) and gives the two parameters zero-size tensors of the checkpoint dtypes, so the loader finds
    them by name and their ``weight_loader`` can validate the shard without copying 101 GB."""
    if _SGL is None:
        raise RuntimeError("engram_store.install() was not called before EngramEmbedding was built")
    torch.nn.Module.__init__(self)
    parallel = _SGL.get_parallel()
    self.dim = dim
    self.tp_size = int(parallel.tp_size)
    tp_rank = int(parallel.tp_rank)
    self.host_table = None
    block = _SGL.block_size
    if dim != ROW_BYTES or dim // block != SCALE_BYTES:
        raise RuntimeError(
            f"engram layer {layer_id}: dim {dim} with {block}-value blocks does not match the {ROW_BYTES}-byte "
            f"rows and {SCALE_BYTES} scales the C gather is built for"
        )
    model_dir = _model_dir()
    span = error = None
    try:
        shard, num_rows, w_off, s_off, lo, hi = _local_span(layer_id, num_embeddings, model_dir)
        span = (lo, hi)
    except RuntimeError as exc:
        error, num_rows = str(exc), num_embeddings
    lo, hi = _check_tiling(layer_id, span, error, num_rows, self.tp_size)
    self.row_start = lo
    self.rows = hi - lo
    self.weight = torch.nn.Parameter(torch.empty((0, dim), dtype=torch.float8_e4m3fn), requires_grad=False)
    self.scale = torch.nn.Parameter(
        torch.empty((0, dim // block), dtype=torch.float8_e8m0fnu), requires_grad=False
    )
    self.weight.weight_loader = _shape_check_loader(layer_id, "weight", (num_embeddings, dim))
    self.scale.weight_loader = _shape_check_loader(layer_id, "scale", (num_embeddings, dim // block))
    threads = _env_int(ENV_THREADS, DEFAULT_THREADS, minimum=1)
    capacity = _env_int(ENV_MAX_IDS, DEFAULT_MAX_IDS, minimum=1)
    store = _RowStore(layer_id, shard, num_rows, w_off, s_off, lo, hi, threads, capacity)
    self._spark_store = store
    _ROW_STORES.append(store)
    STORES.append((layer_id, store.ptr))
    logger.info(
        "engram layer %d: rank %d/%d owns rows [%d, %d) of %d, shard %s, %d gather threads, %d ids per lookup",
        layer_id, tp_rank, self.tp_size, lo, hi, num_rows, shard.name, threads, capacity,
    )
    _start_stats_thread()


def owned_rows(self, indices: torch.Tensor) -> torch.Tensor:
    """Replacement ``EngramEmbedding._owned_rows``: bf16 ``[*indices.shape, dim]`` with this rank's rows
    dequantized and zeros elsewhere, produced by D2H ids -> host-function gather -> H2D rows -> Triton
    dequantize by position. Legal eagerly and under capture: every step is a stream operation on buffers
    that outlive the graph, and nothing here synchronizes."""
    n = indices.numel()
    if n == 0:
        return self._empty(indices)
    if self.rows == 0:
        return self._empty(indices).zero_()
    store: _RowStore = self._spark_store
    if not indices.is_cuda:
        raise RuntimeError(f"engram layer {store.layer_id}: lookup ids must be on the GPU, got {indices.device}")
    if n > store.capacity:
        raise RuntimeError(
            f"engram layer {store.layer_id}: a lookup of {n} ids exceeds {ENV_MAX_IDS}={store.capacity}; "
            "raise it so that chunked-prefill tokens x hash columns fit"
        )
    w_dev, s_dev, positions = store.device_staging(indices.device)
    ids = indices.reshape(-1)
    if ids.dtype != torch.int64:
        ids = ids.to(torch.int64)
    if not ids.is_contiguous():
        ids = ids.contiguous()
    cudart = _cudart()
    stream = torch.cuda.current_stream(indices.device).cuda_stream
    cudart.memcpy_async(store.ids_host.data_ptr(), ids.data_ptr(), n * 8, _MEMCPY_D2H, stream)
    cudart.launch_host_func(stream, _GATHER_FN, ctypes.addressof(store.work_for(n)))
    cudart.memcpy_async(w_dev.data_ptr(), store.w_host.data_ptr(), n * ROW_BYTES, _MEMCPY_H2D, stream)
    cudart.memcpy_async(s_dev.data_ptr(), store.s_host.data_ptr(), n * SCALE_BYTES, _MEMCPY_H2D, stream)
    out = self._empty(indices)
    _SGL.engram_gather(
        w_dev.data_ptr(),
        s_dev.data_ptr(),
        positions[:n],
        out.view(-1, self.dim),
        self.dim,
        _SGL.block_size,
        row_lo=0,
        row_hi=store.capacity,
    )
    return out


# ----------------------------------------------------------------------------------------------------------
# The hasher wrap that feeds engram_prefetch, and the entry point.
# ----------------------------------------------------------------------------------------------------------


def _wrap_hasher(module) -> None:
    """After SGLang's ``EngramHasher.__init__``, hand engram_prefetch CPU copies of the hash tables so the
    scheduler can hash the next prefill chunk on the CPU and warm its rows."""
    cls = module.EngramHasher
    stock = cls.__init__
    if getattr(stock, "_spark_wrapped", False):
        return

    @functools.wraps(stock)
    def __init__(self, *args, **kwargs) -> None:
        stock(self, *args, **kwargs)
        layout = args[0] if args else kwargs.get("layout")
        try:
            import engram_prefetch
        except ImportError as exc:
            logger.warning("engram_prefetch is not importable (%s); next-chunk row warming is off", exc)
            return
        cpu = torch.device("cpu")
        engram_prefetch.HASHER.update(
            pad_id=int(self.pad_id),
            ngram=int(self.max_ngram_size),
            layer_ids=list(layout.layer_ids),
            token_map=self.token_map.detach().to(cpu),
            multipliers=self.multipliers.detach().to(cpu),
            primes=self.primes.detach().to(cpu),
            offsets=self.offsets.detach().to(cpu),
        )

    __init__._spark_wrapped = True
    cls.__init__ = __init__


class _Borrowed:
    """The pieces of the executed engram module a lookup calls into, taken from that module so the hook
    uses exactly the objects SGLang imported."""

    def __init__(self, module) -> None:
        self.module = module
        self.get_parallel = module.get_parallel
        self.engram_gather = module.engram_gather
        self.block_size = int(module.FP8_BLOCK_SIZE)


def install(module) -> None:
    """Install the hook into the executed ``sglang.srt.layers.engram`` module (called once by sitecustomize).

    Replaces ``EngramEmbedding.__init__`` and ``_owned_rows``, leaves every other method as SGLang wrote it,
    and wraps ``EngramHasher.__init__``. The row source is validated here, before the weight load, so a
    misconfigured directory fails in seconds rather than after minutes of loading."""
    global _SGL
    model_dir = _model_dir()
    ranges = row_ranges(model_dir)
    for layer_id in sorted(ranges):
        table_span(model_dir, layer_id)
    _SGL = _Borrowed(module)
    cls = module.EngramEmbedding
    if getattr(cls._owned_rows, "__module__", None) != __name__:
        cls.__init__ = embedding_init
        cls._owned_rows = owned_rows
    _wrap_hasher(module)
    logger.info(
        "engram_store installed: Engram rows of layers %s come from %s through %s",
        sorted(ranges), model_dir, Path(LIB._name).name,
    )


def prefetch_rows(layer_id: int, ids) -> None:
    """Warm this rank's rows among global ``ids`` (int64, any shape) of ``layer_id`` in the page cache:
    ``posix_fadvise(WILLNEED)`` per owned row inside the C store, no copy. Best effort; unknown layers are ignored.
    The scheduler-side prefetch (engram_prefetch.py) calls this for the next prefill chunk."""
    import numpy as np
    store = dict(STORES).get(layer_id)
    if store is None:
        return
    arr = np.ascontiguousarray(np.asarray(ids, dtype=np.int64).reshape(-1))
    if arr.shape[0]:
        LIB.engram_rows_prefetch(store, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), arr.shape[0])
