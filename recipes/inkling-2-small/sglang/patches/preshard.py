"""Pre-sharded weights: each rank's weights written once, then loaded from the rank's own files.

SPARK_PRESHARD_EXPORT=<dir> (a boot that also writes the files; off unless set): every rank writes the weights it
holds to <dir>/rank<R>/ and then serves as usual. SPARK_PRESHARDED=1 (off unless set): the model path is such a rank
folder, and every rank loads its weights from it instead of the official checkpoint.

Where the cut is. The engine's Inkling2ForConditionalGeneration.load_weights (and the draft's) runs a copy loop: each
checkpoint tensor goes to its parameter through the parameter's weight loader, which slices out this rank's part
(with TP_SHARD, a quarter of every expert's width). After the loop the engine checks that every parameter was loaded
and then runs its post-steps (scale swizzles, query scaling, ...); the hooks of patches/ attach their packed copies
after that (dense_p12, dense_i8), and quant methods post-process after load_weights returns (scale_pack, ...). The
files hold the parameters exactly as they stand when the copy loop ends, so every step after the loop still runs, as
on a boot from the official checkpoint:

- export: the weights iterator handed to load_weights is wrapped; when the loop asks for the item after the last
  one, all parameters of the rank are written (2 GiB safetensors parts, keys = parameter names), with the checkpoint
  names and dtypes the loop saw, a checksum of every parameter and the switches that shape the weights;
- reload: the loop gets the recorded checkpoint names with empty tensors of their dtypes (the loop checks names and
  dtypes only) while every weight loader is a no-op, so the engine's own completeness check and post-steps run
  unchanged; when the loop ends, the parameters are filled from the rank's files and their checksum must equal the
  export's, or the boot stops.

The files are tied to what shapes the parameters: the node count (tensor-parallel size), the rank, and the
switches TP_SHARD, TP_SHARD_FP8, TP_SHARD_HALF, FI_MOE and FI_MXFP8. A reload refuses any difference, and any
difference in the parameters' names, shapes or dtypes. With SPARK_PRESHARD_CHECK=1 (validation boots; about 20 s
on GB10) every rank logs a checksum of all parameters after the full load, post-processing included ("[preshard]
after load"): a boot from the rank folders and a boot from the official checkpoint must log the same value.

On GB10 the page cache competes with the GPU for memory, so every file is fsynced and dropped from the page cache
after it is written or read. Copies between the GPU and pageable host memory run at about 150 MB/s on GB10, so the
files are written and read in the safetensors format directly, through two 256 MiB pinned buffers.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import hashlib
import json
import logging
import os
import shutil
import time

import torch

_log = logging.getLogger("spark_preshard")
EXPORT = os.environ.get("SPARK_PRESHARD_EXPORT") or ""
RELOAD = os.environ.get("SPARK_PRESHARDED") == "1"
CHECK = os.environ.get("SPARK_PRESHARD_CHECK") == "1"
PART_BYTES = int(os.environ.get("SPARK_PRESHARD_PART_BYTES", str(2 << 30)))
SHAPING = ("SPARK_TP_SHARD", "SPARK_TP_SHARD_FP8", "SPARK_TP_SHARD_HALF", "SPARK_FI_MOE", "SPARK_FI_MXFP8")
META = "spark-preshard.json"
SMALL_FILES = ("config.json", "hf_quant_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "chat_template.jinja", "generation_config.json")
_S = {"model_path": None}


def _rank_world():
    from sglang.srt.runtime_context import get_parallel
    p = get_parallel()
    return int(p.tp_rank), int(p.tp_size)


def _drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


CHUNK = 1 << 28  # 256 MiB pinned staging buffers
_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64, "U8": torch.uint8,
       "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool,
       "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
_ST = {v: k for k, v in _DT.items()}


class _Staging:
    """Two pinned host buffers: GPU <-> file copies on GB10 run at about 150 MB/s from pageable memory."""

    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.bufs = [torch.empty(CHUNK, dtype=torch.uint8, pin_memory=self.cuda) for _ in range(2)]
        self.views = [memoryview(b.numpy()) for b in self.bufs]
        self.events = [None, None]
        self.i = 0

    def next(self):
        self.i ^= 1
        if self.events[self.i] is not None:
            self.events[self.i].synchronize()
        return self.bufs[self.i], self.views[self.i]

    def mark(self):
        if not self.cuda:
            return
        e = torch.cuda.Event()
        e.record()
        self.events[self.i] = e

    def drain(self):
        for e in self.events:
            if e is not None:
                e.synchronize()


def _bytes(t):
    return t.detach().contiguous().view(-1).view(torch.uint8)


def _save(named, path, staging):
    """A safetensors file written straight from GPU tensors through the pinned buffers."""
    header, off = {"__metadata__": {"format": "pt"}}, 0
    for key in sorted(named):
        t = named[key]
        n = t.numel() * t.element_size()
        header[key] = {"dtype": _ST[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little"))
        f.write(h)
        for key in sorted(named):
            src = _bytes(named[key])
            for start in range(0, src.numel(), CHUNK):
                n = min(CHUNK, src.numel() - start)
                buf, view = staging.next()
                buf[:n].copy_(src[start:start + n])  # synchronous: the bytes are needed on the host now
                f.write(view[:n])
    os.chmod(path, 0o644)
    _drop_cache(path)


def _read_into(path, named, filled, staging):
    """Fill parameters from a safetensors file through the pinned buffers (host -> GPU copies overlap the reads)."""
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
        base = 8 + size
        for key, info in sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                                key=lambda kv: kv[1]["data_offsets"][0]):
            p = named.get(key)
            if p is None or key in filled:
                raise RuntimeError(f"[preshard] {os.path.basename(path)}: unexpected or repeated tensor {key}")
            if tuple(info["shape"]) != tuple(p.shape) or _DT[info["dtype"]] != p.dtype or not p.data.is_contiguous():
                raise RuntimeError(f"[preshard] {key}: file {info['shape']} {info['dtype']}, parameter "
                                   f"{tuple(p.shape)} {p.dtype}")
            dst = p.data.view(-1).view(torch.uint8)
            a, b = info["data_offsets"]
            f.seek(base + a)
            for start in range(0, b - a, CHUNK):
                n = min(CHUNK, b - a - start)
                buf, view = staging.next()
                if f.readinto(view[:n]) != n:
                    raise RuntimeError(f"[preshard] {os.path.basename(path)}: short read in {key}")
                dst[start:start + n].copy_(buf[:n], non_blocking=True)
                staging.mark()
            filled.add(key)
    staging.drain()
    _drop_cache(path)


def _tensor_sum(t, chunk=1 << 26):
    """An order-sensitive checksum of a tensor's bytes, computed on its device in chunks: (sum, weighted sum)."""
    if t.numel() == 0:
        return (0, 0)
    flat = t.detach().contiguous().view(-1).view(torch.uint8)
    s = w = 0
    for start in range(0, flat.numel(), chunk):
        c = flat[start:start + chunk].to(torch.int64)
        idx = torch.arange(start, start + c.numel(), device=c.device, dtype=torch.int64).remainder_(251).add_(1)
        s += int(c.sum())
        w += int((c * idx).sum())
    return (s, w)


def checksum(named):
    """One digest over (name, shape, dtype, byte sums) of every tensor, in name order."""
    h = hashlib.sha256()
    for name in sorted(named):
        t = named[name]
        s, w = _tensor_sum(t)
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|{s}|{w};".encode())
    return h.hexdigest()


def _spec(named):
    return [[n, list(named[n].shape), str(named[n].dtype).removeprefix("torch.")] for n in sorted(named)]


def _shaping_env():
    return {k: os.environ.get(k, "") for k in SHAPING}


def _params(model, draft):
    named = dict(model.named_parameters())
    if draft:  # the draft's load_weights leaves these to the target (shared)
        named = {n: p for n, p in named.items() if not n.startswith(("embed_tokens.", "lm_head."))}
    return named


def _write(named, folder, prefix, single=None):
    """Parameters to safetensors files of at most PART_BYTES (one file named `single` if given); returns the map."""
    weight_map, part, size, n = {}, {}, 0, 0
    staging = _Staging()

    def flush():
        nonlocal part, size, n
        if not part:
            return
        name = single or f"{prefix}-{n + 1:05d}.safetensors"
        _save(part, os.path.join(folder, name), staging)
        for key in part:
            weight_map[key] = name
        part, size, n = {}, 0, n + 1

    for key in sorted(named):
        t = named[key]
        nbytes = t.numel() * t.element_size()
        if single is None and part and size + nbytes > PART_BYTES:
            flush()
        part[key] = t
        size += nbytes
    flush()
    if single is None:  # the engine loads model*.safetensors through the index
        total = sum(named[k].numel() * named[k].element_size() for k in named)
        renamed = {}
        files = sorted(set(weight_map.values()))
        for i, old in enumerate(files):
            new = f"{prefix}-{i + 1:05d}-of-{len(files):05d}.safetensors"
            os.replace(os.path.join(folder, old), os.path.join(folder, new))
            renamed[old] = new
        weight_map = {k: renamed[v] for k, v in weight_map.items()}
        with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1, sort_keys=True)
    return weight_map


def _folder():
    rank, _ = _rank_world()
    folder = os.path.join(EXPORT, f"rank{rank}")
    os.makedirs(folder, exist_ok=True)
    os.chmod(EXPORT, 0o777)
    os.chmod(folder, 0o777)
    return folder


def _export(model, draft, names):
    t0 = time.time()
    rank, world = _rank_world()
    folder = _folder()
    named = _params(model, draft)
    digest = checksum(named)
    if draft:
        _write(named, folder, "mtp", single="mtp.safetensors")
    else:
        _write(named, folder, f"model-spark-rank{rank}")
        src = _S["model_path"]
        if src:
            for f in SMALL_FILES:
                if os.path.isfile(os.path.join(src, f)):
                    shutil.copyfile(os.path.join(src, f), os.path.join(folder, f))
                    os.chmod(os.path.join(folder, f), 0o644)
            if os.path.isdir(os.path.join(src, "tiktoken")):
                shutil.copytree(os.path.join(src, "tiktoken"), os.path.join(folder, "tiktoken"), dirs_exist_ok=True)
    meta_path = os.path.join(folder, META)
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    meta.update({"rank": rank, "world": world, "shaping_env": _shaping_env(), "format": 1})
    meta["draft" if draft else "target"] = {"names": names, "spec": _spec(named), "checksum": digest}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    os.chmod(meta_path, 0o644)
    nbytes = sum(p.numel() * p.element_size() for p in named.values())
    _log.warning("[preshard] rank %d wrote the %s: %d parameters, %.2f GB, checksum %s, %.0f s", rank,
                 "draft" if draft else "target", len(named), nbytes / 1e9, digest[:16], time.time() - t0)


def _check_meta(meta, draft):
    rank, world = _rank_world()
    why = []
    if meta.get("rank") != rank:
        why.append(f"the folder is rank {meta.get('rank')}'s, this is rank {rank}")
    if meta.get("world") != world:
        why.append(f"written for {meta.get('world')} ranks, this fleet has {world}")
    if meta.get("shaping_env") != _shaping_env():
        why.append(f"written with {meta.get('shaping_env')}, this boot has {_shaping_env()}")
    if ("draft" if draft else "target") not in meta:
        why.append("no " + ("draft" if draft else "target") + " in the folder")
    if why:
        raise RuntimeError("[preshard] SPARK_PRESHARDED=1 refused: " + "; ".join(why))


def _fill(model, draft, meta):
    t0 = time.time()
    folder = _S["model_path"]
    part = meta["draft" if draft else "target"]
    named = _params(model, draft)
    if _spec(named) != part["spec"]:
        raise RuntimeError("[preshard] the parameters' names, shapes or dtypes differ from the rank folder's")
    if draft:
        files = {"mtp.safetensors": None}
    else:
        index = json.load(open(os.path.join(folder, "model.safetensors.index.json")))["weight_map"]
        files = {f: None for f in sorted(set(index.values()))}
    filled, staging = set(), _Staging()
    for name in files:
        _read_into(os.path.join(folder, name), named, filled, staging)
    if filled != set(named):
        raise RuntimeError(f"[preshard] {len(set(named) - filled)} parameters missing from the rank folder")
    digest = checksum(named)
    if digest != part["checksum"]:
        raise RuntimeError(f"[preshard] checksum {digest[:16]} differs from the export's {part['checksum'][:16]}")
    rank, _ = _rank_world()
    _log.warning("[preshard] rank %d loaded the %s from its folder: %d parameters, checksum %s (equal to the "
                 "export's), %.0f s", rank, "draft" if draft else "target", len(named), digest[:16], time.time() - t0)


class _NoopLoaders:
    """Every parameter's weight loader, and the module's default one, as a no-op while the copy loop runs."""

    def __init__(self, model, mod):
        self.model, self.mod, self.saved = model, mod, []

    def __enter__(self):
        noop = lambda *a, **k: None  # noqa: E731
        for p in self.model.parameters():
            self.saved.append((p, p.__dict__.get("weight_loader", None), "weight_loader" in p.__dict__))
            p.weight_loader = noop
        self.default = self.mod.default_weight_loader
        self.mod.default_weight_loader = noop
        return self

    def __exit__(self, *exc):
        for p, loader, had in self.saved:
            if had:
                p.weight_loader = loader
            else:
                del p.weight_loader
        self.mod.default_weight_loader = self.default
        return False


def _wrap(cls, mod, draft):
    stock = cls.load_weights

    def load_weights(self, weights):
        if EXPORT:
            names = []

            def recorded():
                for name, w in weights:
                    names.append([name, str(w.dtype).removeprefix("torch.")])
                    yield name, w
                _export(self, draft, names)  # the copy loop has asked for the item after the last one
            return stock(self, recorded())
        if RELOAD:
            meta = json.load(open(os.path.join(_S["model_path"], META)))
            _check_meta(meta, draft)
            noop = _NoopLoaders(self, mod)

            def names_only():
                with noop:
                    for name, dtype in meta["draft" if draft else "target"]["names"]:
                        yield name, torch.empty(0, dtype=getattr(torch, dtype))
                _fill(self, draft, meta)  # every loader is itself again; the loop has seen every name
            return stock(self, names_only())
        return stock(self, weights)

    load_weights.__wrapped__ = stock
    cls.load_weights = load_weights


def install_target(mod):
    _wrap(mod.Inkling2ForConditionalGeneration, mod, draft=False)
    _log.warning("[preshard] %s", f"export to {EXPORT}" if EXPORT else "loading from the rank folder")


def install_draft(mod):
    _wrap(mod.Inkling2ForConditionalGenerationMTP, mod, draft=True)


def install_loader(mod):
    """Remember the model path (the small files go with an export) and log a checksum after every full load."""
    cls = mod.DefaultModelLoader
    stock = cls.load_model

    def load_model(self, *args, **kwargs):
        cfg = kwargs.get("model_config") or (args[0] if args else None)
        path = getattr(cfg, "model_path", None)
        if path and os.path.isdir(path) and _S["model_path"] is None:
            _S["model_path"] = path
        model = stock(self, *args, **kwargs)
        if not CHECK:
            return model
        try:
            rank, _ = _rank_world()
            _log.warning("[preshard] after load: rank %d, %s, checksum of all parameters %s", rank,
                         type(model).__name__, checksum(dict(model.named_parameters()))[:16])
        except Exception as e:  # noqa: BLE001
            _log.warning("[preshard] after load: no checksum (%r)", e)
        return model

    load_model.__wrapped__ = stock
    cls.load_model = load_model
