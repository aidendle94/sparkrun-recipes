"""The per-layer glue of a prefill chunk in fewer passes over memory (SPARK_PREFILL_GLUE=<parts>, off by default).

A 4,096-token chunk spends about a tenth of its time between the big kernels: every one of the 48 layers builds its
inputs, projects, biases and closes with launches that each read and write rows x 4608 values (or the 100 MB
relative-bias tensor) once more. Each part below removes one of those passes and keeps every IEEE operation and its
order, so the values are the same bit for bit; each is compared with stock's own path on a GPU by bench/glue_check.py
and, inside a serving boot, by SPARK_PREFILL_GLUE_CHECK=1.

  epilogue  short_convolution(mlp) + residual_add(partial, attention, conv) of the layer's end in one tiled kernel
            (fused_epilogue.py); chunks of at least PREFILL_MIN_ROWS rows, where stock takes its own tiled kernel.
  stream    attn_res + RMSNorm of the attention input (1 stream on a local layer, 3 on a global one) and of the MLP
            input in one kernel that never writes the FP32 stream (fused_stream.py).
  gate      FA4's output stays BF16: stock asks for an FP32 copy of it (which the SM12x forward makes from its BF16
            result) and gate_sink reads that; with RadixAttention.out_dtype None for the call, gate_sink's own BF16
            kernel reads the result directly. No kernel of ours.
  qkvr      a global layer projects its three sources through the whole qkvr weight (3 x rows x 3792 outputs) and
            keeps k of the second and v of the third; here the first source takes the whole weight and the other two
            only their slice of it. cuBLAS's sums depend on the shape it is given, so every row count is compared
            with the stacked product the first time it is seen and left to stock if a single bit differs.
  relbias   the relative-position bias is written once, already sheared, by fused_relbias.py instead of an einsum
            and the interface's shear: RelLogitsProj.forward hands its input back, and sitecustomize's FA4 wrapper,
            the place that knows the call's sequences and window, writes the sheared workspace and passes it as
            presheared. A call the kernel does not cover (image tokens, no KV cache, ..) gets stock's tensor back.

Only plain extend batches (ForwardMode.EXTEND) outside CUDA graph capture take any of it: decode, verification and
mixed batches run stock's code (inside a graph at 1-8 rows the fused stream is slower than stock's pair). Every
refusal logs its reason once.

Measured on GB10 (2026-11-16). One GPU, bench/glue_check.py: 0 differing values for epilogue, stream, gate and
relbias in every case; the split qkvr product equals the stacked one from 500 rows up and differs at 65-384 rows
(hence the floor and the per-row-count comparison). Four nodes, all five parts: 0 differing values under CHECK
on every rank over a 9,000, four 30,000 and a 120,000-token prefill; cold prefill 2,474 -> 2,756 tok/s at 30,000
tokens (+11%), 2,241 -> 2,469 at 120,000 (+10%); NLL equal to the reference to the last digit; decode unchanged.

SPARK_PREFILL_GLUE_ROWS=part=rows,..   row floors (defaults in ROWS), for the windows that look for a crossover
SPARK_PREFILL_GLUE_CHECK=1             diagnosis windows only: every fused call also runs stock's path and counts
                                       the values that differ (log lines `[prefill_glue] check ..`)

MIT License, Copyright (c) 2026 Aiden Le.
"""
import inspect
import linecache
import logging
import os
import sys
import textwrap
import threading

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.forward_mode import ForwardMode

_log = logging.getLogger("prefill_glue")
ALL = ("epilogue", "stream", "gate", "qkvr", "relbias")
PARTS = {part for part in (os.environ.get("SPARK_PREFILL_GLUE") or "").split(",") if part}
CHECK = os.environ.get("SPARK_PREFILL_GLUE_CHECK") == "1"
ROWS = dict(epilogue=2048, stream=512, gate=2048, qkvr=512, relbias=1)  # the floors of the validated boots
for _item in (os.environ.get("SPARK_PREFILL_GLUE_ROWS") or "").split(","):
    if _item:
        ROWS[_item.partition("=")[0]] = int(_item.partition("=")[2])

TIMED = None  # sitecustomize's _timed when the timers are on
_SEEN = set()
_STATE = threading.local()  # plain: inside a plain-prefill attention forward; defer / stash: the relbias hand-over
_CHECKS = {}  # part -> [calls, calls with a difference, differing values]
_QKVR_ROWS = {}  # row count -> the split product equals the stacked one (its first use)
_STOCK_REL = None  # RelLogitsProj.forward as the image has it


def _timed(category, fn):
    return fn if TIMED is None else TIMED(category, fn)


def _note(part, why):
    if (part, why) not in _SEEN:
        _SEEN.add((part, why))
        _log.warning("[prefill_glue] %s: stock's path, %s", part, why)


def _first(part, what):
    if (part, None) not in _SEEN:
        _SEEN.add((part, None))
        _log.warning("[prefill_glue] %s: first fused call (%s)", part, what)


def _plain(forward_batch, x):
    return (forward_batch.forward_mode == ForwardMode.EXTEND and x.is_cuda
            and not torch.cuda.is_current_stream_capturing())


def _differ(a, b):
    """Values of a whose bits are not b's."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), 1)
    bits = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return int((a.contiguous().view(bits) != b.contiguous().view(bits)).sum())


def _checked(part, pairs):
    differ = sum(_differ(a, b) for a, b in pairs)
    count = _CHECKS.setdefault(part, [0, 0, 0])
    count[0] += 1
    count[1] += differ != 0
    count[2] += differ
    if (differ and count[1] <= 20) or count[0] & (count[0] - 1) == 0 or count[0] % 512 == 0:
        _log.warning("[prefill_glue] check %s: call %d, %d of %d values differ; so far %d calls with a difference, "
                     "%d values", part, count[0], differ, sum(a.numel() for a, _ in pairs), count[1], count[2])
    return differ


# ---- stream and epilogue: Inkling2DecoderLayer.forward

def _stream(state, sources, addend, raw, scaled, norms):
    """A tuple of BF16 streams (one per norm) from one fused launch, or None: stock's attn_res and norms run."""
    import fused_stream

    first = sources[0]
    if first.shape[0] < ROWS["stream"]:
        return None
    why = None
    if not 2 <= len(sources) <= fused_stream.NS:
        why = f"{len(sources)} sources"
    for x in (*sources, *(() if addend is None else (addend,))):
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape != first.shape or x.shape[1] != fused_stream.D:
            why = why or f"a {tuple(x.shape)} {x.dtype} source"
    eps = norms[0].eps
    for norm in norms:
        weight = norm.weight
        if norm.pre_weight is not None or weight is None:
            why = why or "a norm with a pre-affine weight or without a weight"
        elif weight.dtype != torch.bfloat16 or weight.shape != (fused_stream.D,) or not weight.is_contiguous():
            why = why or f"a {tuple(weight.shape)} {weight.dtype} norm weight"
        elif norm.output_dtype != torch.bfloat16 or norm.eps != eps:
            why = why or "norms of another output dtype or of different eps"
    queries = raw.unsqueeze(0) if raw.ndim == 1 else raw
    if queries.shape[0] != len(norms) or queries.shape[0] not in fused_stream.GEOMETRY:
        why = why or f"{queries.shape[0]} queries for {len(norms)} norms"
    if why is not None:
        _note("stream", why)
        return None
    # block_attn_res's own lines
    hidden = first.shape[-1]
    effective = scaled.view(queries.shape) if scaled is not None else queries.float() * (hidden ** -0.5)
    if effective.dtype != torch.float32:
        _note("stream", f"{effective.dtype} queries")
        return None
    _first("stream", f"{len(sources)} sources, {len(norms)} queries, {first.shape[0]} rows")
    return _timed("glue_stream", fused_stream.fused_attn_res_norm)(
        [x.contiguous() for x in sources], None if addend is None else addend.contiguous(), effective.contiguous(),
        [norm.weight.data for norm in norms], eps, normalize_values=state.attn_res.normalize_values)


def attention_streams(layer, state, forward_batch):
    """The layer's attention input (stock: state.attention_input + attn_norm, or the three input norms into one
    stacked buffer), or None."""
    if "stream" not in PARTS or not state.blocks or not _plain(forward_batch, state.partial):
        return None
    res, lid = state.attn_res, layer.layer_id
    norms = (layer.attn_norm,) if layer.is_local else (layer.q_input_norm, layer.k_input_norm, layer.v_input_norm)
    out = _stream(state, [*state.blocks, state.partial], None, res.attn_query(lid), res.scaled_query("attn", lid),
                  norms)
    if out is None:
        return None
    if CHECK:
        streams = state.attention_input(lid)
        _checked("stream", [(got, norm(stream)) for got, norm, stream in zip(out, norms, streams)])
    if layer.is_local:
        return out[0]
    from sglang.srt.models.inkling_common.inkling2_attn import AttentionSources

    return AttentionSources(*out)  # adjacent views of one buffer, as stock's stacked[0..2]


def mlp_stream(layer, state, attention, forward_batch):
    """The layer's MLP input (stock: mlp_norm(state.mlp_input(layer_id, attention))), or None."""
    if "stream" not in PARTS or not _plain(forward_batch, attention):
        return None
    res, lid = state.attn_res, layer.layer_id
    if res.is_block_boundary(lid):  # the attention output is a source of its own
        sources, addend = [*state.blocks, state.partial, attention], None
    else:  # it is added to the partial block
        sources, addend = [*state.blocks, state.partial], attention
    out = _stream(state, sources, addend, res.mlp_queries[lid], res.scaled_query("mlp", lid), (layer.mlp_norm,))
    if out is None:
        return None
    if CHECK:
        _checked("stream", [(out[0], layer.mlp_norm(state.mlp_input(lid, attention)))])
    return out[0]


def finalize_extend(layer, state, attention, mlp, forward_batch):
    """mlp_sconv + state.finalize_layer of a plain prefill chunk in one launch; False = stock's two lines run."""
    if "epilogue" not in PARTS or not isinstance(mlp, torch.Tensor) or not _plain(forward_batch, mlp):
        return False
    import fused_epilogue
    from sglang.kernels.ops.inkling2.sconv import PREFILL_MIN_ROWS
    from sglang.srt.models.inkling_common.kernels.sconv import update_sconv_cache

    if mlp.shape[0] < max(ROWS["epilogue"], PREFILL_MIN_ROWS):
        return False
    conv, lid = layer.mlp_sconv, layer.layer_id
    boundary = state.attn_res.is_block_boundary(lid)
    why = None
    if conv.is_dynamic:
        why = "a dynamic convolution"
    elif mlp.dim() != 2 or mlp.stride(-1) != 1 or mlp.dtype != torch.bfloat16:
        why = f"a {tuple(mlp.shape)} {mlp.dtype} MLP output"
    elif attention.dtype != torch.bfloat16 or attention.shape != mlp.shape:
        why = f"a {tuple(attention.shape)} {attention.dtype} attention output"
    elif not boundary and (state.partial.dtype != torch.bfloat16 or state.partial.shape != mlp.shape):
        why = f"a {tuple(state.partial.shape)} {state.partial.dtype} partial block"
    if why is not None:
        _note("epilogue", why)
        return False
    _first("epilogue", f"{mlp.shape[0]} rows")
    # ShortConvolution.forward's own lines around the kernel
    meta = conv._conv_state(forward_batch)
    cache = conv._sconv_cache()
    weight = conv._weight_2d()
    conv._prepare_extend_sconv_cache(forward_batch, cache, mlp, meta.track_conv_indices)
    reference = None
    if CHECK:
        from sglang.kernels.ops.inkling2.residual import residual_add
        from sglang.kernels.ops.inkling2.sconv import short_convolution

        convolved = short_convolution(mlp, weight, cache, meta.precomputed, dynamic_weights=None, decode=False,
                                      track_mask=None, track_indices=None, window=None)
        reference = attention + convolved if boundary else residual_add(state.partial, attention, convolved)
    fused = _timed("glue_epilogue", fused_epilogue.fused_epilogue)
    state.finalize_layer_with(lid, lambda partial: fused(mlp, weight, cache, meta.precomputed, attention, partial))
    if reference is not None:
        _checked("epilogue", [(state.partial, reference)])
    update_sconv_cache(x=mlp, sconv_cache=cache, cache_indices=meta.cache_indices,
                       has_initial_state=meta.has_initial_state, query_start_loc=meta.query_start_loc)
    return True


def _derive(source):
    """Inkling2DecoderLayer.forward with the three seams: stock's lines stay, each behind `if <ours> is None`."""
    lines = source.rstrip().splitlines()

    def at(text):
        hits = [i for i, line in enumerate(lines) if line == "    " + text]
        if len(hits) != 1:
            raise ValueError(f"{len(hits)} lines read `{text}`")
        return hits[0]

    def inner(block):
        return ["    " + line if line else line for line in block]

    if not lines[0].startswith("def forward(self, state, positions, forward_batch)"):
        raise ValueError(f"signature `{lines[0]}`")
    a = at("streams = state.attention_input(self.layer_id)")
    b = at("attention = self.attn(attn_input, positions, forward_batch)")
    m = at("mlp_input = self.mlp_norm(state.mlp_input(self.layer_id, attention))")
    s = at("mlp = self.mlp_sconv(mlp, positions, forward_batch)")
    f = at("state.finalize_layer(self.layer_id, attention, mlp)")
    if not (a < b < m < s and f == s + 1 == len(lines) - 1):
        raise ValueError("the lines are not in the expected order")
    return "\n".join([
        *lines[:a],
        "    attn_input = _spark_glue.attention_streams(self, state, forward_batch)",
        "    if attn_input is None:",
        *inner(lines[a:b]),
        *lines[b:m],
        "    mlp_input = _spark_glue.mlp_stream(self, state, attention, forward_batch)",
        "    if mlp_input is None:",
        *inner(lines[m:m + 1]),
        *lines[m + 1:s],
        "    if not _spark_glue.finalize_extend(self, state, attention, mlp, forward_batch):",
        *inner(lines[s:]),
    ]) + "\n"


def install_layer(mod):
    """sglang.srt.models.inkling2: the layer forward with the stream and epilogue seams."""
    if not PARTS & {"stream", "epilogue"}:
        return
    cls = mod.Inkling2DecoderLayer
    stock = cls.forward
    try:
        if hasattr(stock, "__wrapped__"):
            raise ValueError("it is wrapped already")
        source = _derive(textwrap.dedent(inspect.getsource(stock)))
        filename = "<prefill_glue: Inkling2DecoderLayer.forward>"
        code = compile(source, filename, "exec")
    except Exception as e:  # another image: nothing is installed
        _log.warning("[prefill_glue] Inkling2DecoderLayer.forward is not the one this was written for (%s: %s): "
                     "stream and epilogue stay stock", type(e).__name__, e)
        return
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    mod._spark_glue = sys.modules[__name__]
    scope = {}
    exec(code, mod.__dict__, scope)
    forward = scope["forward"]
    forward.__qualname__ = stock.__qualname__
    forward.__wrapped__ = stock
    cls.forward = forward
    _log.warning("[prefill_glue] layer forward installed (parts %s, rows %s%s)", sorted(PARTS), ROWS,
                 ", CHECK" if CHECK else "")


# ---- gate, qkvr, relbias: Inkling2Attention

def _plain_gemm(layer):
    """ColumnParallelLinear.forward and UnquantizedLinearMethod.apply would reach F.linear(x, weight) for it."""
    from sglang.srt.layers import linear
    from sglang.srt.layers.quantization import unquant

    backend = unquant.get_bf16_gemm_backend()
    return not (linear.get_forward().sp_active or unquant.use_intel_amx_backend(layer) or unquant._use_aiter
                or backend.is_cutedsl() or backend.is_gemv()
                or not isinstance(layer.quant_method, unquant.UnquantizedLinearMethod))


def _split_qkvr(attn, sources, inner):
    """_project_sources of three distinct sources with one full and two sliced products, or None."""
    q_in, k_in, v_in = sources.q, sources.k, sources.v
    if len({id(q_in), id(k_in), id(v_in)}) != 3:
        return None
    rows = q_in.shape[0]
    known = _QKVR_ROWS.get(rows)
    if rows < ROWS["qkvr"] or known is False:
        return None
    layer = attn.qkvr
    packed = layer.__dict__.get("_p12")  # dense_p12: the BF16 weight may be gone, rebuilt in a scratch buffer
    freed = packed is not None and packed.freed
    why = None
    if layer.bias is not None and not layer.skip_bias_add:
        why = "a bias"
    elif layer.gather_output:
        why = "gathered outputs"
    elif getattr(layer, "_i8", None) is not None:
        why = "int8 codes"
    elif any(x.dim() != 2 or x.dtype != torch.bfloat16 or x.shape != q_in.shape for x in (q_in, k_in, v_in)):
        why = f"{tuple(k_in.shape)} {k_in.dtype} sources"
    elif not freed and layer.weight.dtype != torch.bfloat16:
        why = f"a {layer.weight.dtype} weight"
    elif not _plain_gemm(layer):
        why = "another GEMM backend or sequence parallelism"
    if why is not None:
        _note("qkvr", why)
        return None
    sizes = attn.split_sizes
    k0 = sizes[0]
    k1 = k0 + sizes[1]
    k2 = k1 + sizes[2]

    def project(weight):
        q, _, _, r, gate, sinks = F.linear(q_in, weight).split(sizes, dim=-1)[:6]
        return q, F.linear(k_in, weight[k0:k1]), F.linear(v_in, weight[k1:k2]), r, gate, sinks

    def split():
        if not freed:
            return project(layer.weight)
        import dense_p12

        with dense_p12._unpacked(layer, packed):
            return project(layer.weight)

    result = _timed("glue_qkvr", split)()
    if known is None or CHECK:
        reference = inner(attn, sources)
        pairs = list(zip(result, reference))
        differ = _checked("qkvr", pairs) if CHECK else sum(_differ(a, b) for a, b in pairs)
        if known is None:
            _QKVR_ROWS[rows] = differ == 0
            if differ or len(_QKVR_ROWS) == 1:
                _log.warning("[prefill_glue] qkvr: %d rows, the split product %s", rows,
                             f"differs in {differ} values: stock's stacked product for this row count" if differ
                             else "equals the stacked one bit for bit (checked once per row count)")
        if differ:
            return reference
    return result


def install_attn(mod):
    """sglang.srt.models.inkling_common.inkling2_attn: the plain-prefill bracket around Inkling2Attention.forward
    (gate, and the flags the other two seams read), and the qkvr seam around whatever _project_sources is by now."""
    if not PARTS & {"gate", "qkvr", "relbias"}:
        return
    cls = mod.Inkling2Attention
    stock = cls.forward
    gate, relbias = "gate" in PARTS, "relbias" in PARTS

    def forward(self, hidden_states, positions, forward_batch):
        first = hidden_states if isinstance(hidden_states, torch.Tensor) else hidden_states[0]
        if not _plain(forward_batch, first) or mod.get_model().kv_cache_dtype == "mxfp8":
            return stock(self, hidden_states, positions, forward_batch)
        rows = first.shape[0]
        radix = self.attn
        narrow = False
        if gate and rows >= ROWS["gate"]:
            narrow = radix.out_dtype is torch.float32 and getattr(forward_batch, "_attn_output", None) is None
            if narrow:
                _first("gate", f"{rows} rows")
            else:
                _note("gate", f"out_dtype {radix.out_dtype} or a preallocated attention output")
        _STATE.plain = True
        _STATE.defer = relbias and rows >= ROWS["relbias"]
        _STATE.narrow = narrow
        if narrow:
            radix.out_dtype = None
        try:
            return stock(self, hidden_states, positions, forward_batch)
        finally:
            if narrow:
                radix.out_dtype = torch.float32
            _STATE.plain = _STATE.defer = _STATE.narrow = False
            _STATE.stash = None

    forward.__wrapped__ = stock
    cls.forward = forward

    if "qkvr" in PARTS:
        inner = cls._project_sources

        def _project_sources(self, sources):
            if not getattr(_STATE, "plain", False):
                return inner(self, sources)
            split = _split_qkvr(self, sources, inner)
            return inner(self, sources) if split is None else split

        _project_sources.__wrapped__ = inner
        cls._project_sources = _project_sources

    if gate and CHECK:
        stock_gate = mod.gate_sink

        def gate_sink(output, *args, **kwargs):
            result = stock_gate(output, *args, **kwargs)
            if getattr(_STATE, "narrow", False) and output.dtype == torch.bfloat16:  # today's path: an FP32 copy
                _checked("gate", [(result, stock_gate(output.float(), *args, **kwargs))])
            return result

        mod.gate_sink = gate_sink
    _log.warning("[prefill_glue] attention forward installed (parts %s, rows %s%s)", sorted(PARTS), ROWS,
                 ", CHECK" if CHECK else "")


def install_relproj(mod):
    """sglang.srt.models.inkling_common.attn: RelLogitsProj.forward hands its input back when a plain-prefill
    attention forward asked for it; fa4_relbias (or undefer) takes it from there."""
    global _STOCK_REL
    if "relbias" not in PARTS:
        return
    import fused_relbias

    cls = mod.RelLogitsProj
    _STOCK_REL = stock = cls.forward

    def forward(self, r_out, log_scaling_tau=None):
        if not getattr(_STATE, "defer", False):
            return stock(self, r_out, log_scaling_tau)
        _STATE.defer = False  # one projection per attention forward
        proj = self.proj
        why = None
        if log_scaling_tau is not None:
            why = "a log-scaling tau"
        elif r_out.dim() != 3 or r_out.dtype != torch.bfloat16 or r_out.shape[2] != 16 or r_out.stride(-1) != 1:
            why = f"a {tuple(r_out.shape)} {r_out.dtype} r"
        elif (proj.dtype != torch.bfloat16 or proj.shape[0] != 16 or not proj.is_contiguous() or proj.shape[1] % 128
              or (proj.shape[1] + fused_relbias.PAD) // 128 % fused_relbias.WORKERS):
            why = f"a {tuple(proj.shape)} {proj.dtype} projection"
        if why is not None:
            _note("relbias", why)
            return stock(self, r_out, log_scaling_tau)
        _STATE.stash = (r_out, self)
        return r_out

    forward.__wrapped__ = stock
    cls.forward = forward


class _Took:
    """A deferred projection that fa4_relbias wrote: what the FA4 wrapper needs of it afterwards."""
    __slots__ = ("r", "module", "reference")

    def __init__(self, r, module):
        self.r, self.module, self.reference = r, module, None

    def unsheared(self):
        """Stock's (tokens, heads, rel_extent) tensor (for attn_check, which shears nothing)."""
        return _STOCK_REL(self.module, self.r)

    def verify(self, result):
        if self.reference is not None:
            _checked("relbias", list(zip(result if isinstance(result, tuple) else (result,), self.reference)))


def _fa4_why_not(name, args, kwargs, nested, r, extent):
    if name != "flash_attn_with_kvcache" or nested or args:
        return f"reached through {name}" + (" (nested)" if nested else "") + (" with positional arguments" if args
                                                                              else "")
    if kwargs.get("causal") is not True or kwargs.get("mask_mod") is not None or kwargs.get("score_mod") is not None:
        return "not a plain causal call (image tokens?)"
    if kwargs.get("pack_gqa"):
        return "packed GQA heads"
    window = kwargs.get("window_size") or (-1, -1)
    if tuple(window) not in ((-1, -1), (extent - 1, 0)):
        return f"window {tuple(window)}"
    q = kwargs.get("q")
    if q is None or q.dim() != 3 or q.shape[:2] != r.shape[:2]:
        return f"q {None if q is None else tuple(q.shape)} for r {tuple(r.shape)}"
    for key in ("cu_seqlens_q", "cache_seqlens"):
        x = kwargs.get(key)
        if x is None or x.dtype != torch.int32 or not x.is_contiguous() or x.dim() != 1:
            return f"{key} {None if x is None else (tuple(x.shape), x.dtype)}"
    if kwargs["cu_seqlens_q"].numel() != kwargs["cache_seqlens"].numel() + 1:
        return "cu_seqlens_q and cache_seqlens of different batches"
    return None


def fa4_relbias(name, stock, args, kwargs, nested, timed, glob):
    """Called by sitecustomize's FA4 wrapper before its stock call. When kwargs["rel_bias"] is the input a
    RelLogitsProj handed back, it becomes the sheared workspace (the wrapper then calls presheared; returns a
    _Took) or, for a call the kernel does not cover, stock's tensor (returns None)."""
    stash = getattr(_STATE, "stash", None)
    if stash is None or kwargs.get("rel_bias") is not stash[0]:
        return None
    _STATE.stash = None
    r, module = stash
    proj = module.proj.data
    why = _fa4_why_not(name, args, kwargs, nested, r, proj.shape[1])
    if why is not None:
        _note("relbias", why)
        kwargs["rel_bias"] = _STOCK_REL(module, r)
        return None
    import fused_relbias

    took = _Took(r, module)
    cu, lens = kwargs["cu_seqlens_q"], kwargs["cache_seqlens"]
    if CHECK:
        short = int(((cu[1:] - cu[:-1]) > lens).sum())
        if short:
            _log.warning("[prefill_glue] check relbias: %d sequences with fewer keys than rows", short)
        reference = stock(*args, **{**kwargs, "rel_bias": took.unsheared()})
        took.reference = tuple(x.clone() for x in (reference if isinstance(reference, tuple) else (reference,)))
    window = (kwargs.get("window_size") or (-1, -1))[0]
    _first("relbias", f"{r.shape[0]} rows, window {window}")
    kwargs["rel_bias"] = timed("relbias" + glob, fused_relbias.fused_relbias)(
        r, proj, cu, lens, window if window >= 0 else None)
    return took


def undefer(kwargs):
    """Called by sitecustomize's interface wrapper: a handed-back input that no wrapped call took (a route around
    flash_attn_with_kvcache) becomes stock's tensor before the interface reads it."""
    stash = getattr(_STATE, "stash", None)
    if stash is not None and kwargs.get("rel_bias") is stash[0]:
        _STATE.stash = None
        _note("relbias", "the interface was reached around the FA4 wrapper")
        kwargs["rel_bias"] = _STOCK_REL(stash[1], stash[0])
