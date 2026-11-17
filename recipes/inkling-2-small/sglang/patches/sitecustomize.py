"""GB10 (SM12x) bring-up hooks for Inkling-2-Small on SGLang dev-inkling-2-small, installed at interpreter start.

fa4_sm12x_args (SGLang's SM12x FA4 wrapper flash_attention_v4_sm120 swallows arguments the SM100 wrapper forwards):
  rel_bias_flags: Inkling2 passes rel_bias_scale_after_add=True and, on the padded verify path,
  rel_bias_presheared=True. scale_after_add only names SM100's operation order ((qk + bias / scale) * scale); every
  kernel, SM12x's included, computes qk * scale + bias, so the flag is dropped and the bias passed as is. (Until
  2026-11-13 this hook multiplied the bias by softmax_scale, which left prefill with a bias 11x too weak:
  bench/bias_consistency.py.) presheared is forwarded.
fa4_out_dtype: Inkling2 asks FA4 for a float32 output (gate_sink combines heads with the LSE in fp32). The SM12x FA4
  forward writes only the input dtype, so on SM12x a float32 `out` is replaced by a bf16 result copied into it.
moe_marlin / fi_moe: the routed experts' NVFP4/MXFP8 kernels are CUTLASS SM100 JIT builds; on SM12x the experts take SGLang's
  Hopper path instead (Marlin GEMMs with NVFP4/MXFP8 fake-quantized activations, repacked once at load).
nccl_collectives: Inkling2's collectives (kernels/ops/inkling2/comm.py) use NVLink multicast (torch symmetric memory).
  Four GB10s joined over RoCE have none, so on SM12x all_reduce / multicast_all_reduce go through the process group's
  regular all_reduce (NCCL over the fabric) and the one-shot multicast path is declined (returns None, as the
  module does when multicast is unavailable). The patched names bind in inkling2_attn / inkling2_moe because their
  `from ...comm import` runs after this hook.
served_aliases: SPARK_SERVED_ALIASES (comma-separated) are listed by /v1/models and retrievable by /v1/models/{id}, with
  the real served name as root (SGLang's chat/completion handlers already ignore the requested id). Ported from the
  DeepSeek-V4.1 overlay (overlay/served_aliases.py); the two handlers mirror SGLang's http_server.py (Apache-2.0).
  The Ollama-style discovery routes follow: /api/tags lists the aliases beside the served name, and /api/show takes
  the model under `name` (what older Ollama clients send) as well as `model`. SGLang's handlers stay if that fails.
triton_decode: compact decode/verify attention (rel_bias_partials) uses FA4's SM100 bias-decode kernel off Hopper and
  FA4's SM90 kernel for wide verify on Hopper; neither runs on SM12x, so there every call takes the Triton
  decode_partials kernel (any qlen, causal, compact relative bias), which returns the (o, m, l) partials that
  combine_gate_sink already accepts from the Hopper path.
  That kernel reads a tile of keys from one page-table entry, which at GB10's page size of 1 is only right while a
  request's pool slots are consecutive, so by default patches/paged_decode.py replaces it (the same kernel with the
  page looked up per key; SPARK_PAGED_DECODE=0 goes back to SGLang's).
attn_hopper_dispatch: Inkling2Attention's decode/verify dispatch reads get_platform().is_sm90. Off Hopper, MTP verify
  takes the padded FA4 path with split-KV and a presheared bias laid out for SM100 tiles; the SM12x FA4 forward takes
  one split and other tiles. On SM12x that module sees is_sm90=True, so verify takes the compact path (Triton
  decode_partials through triton_decode) as on Hopper; prefill is unaffected (compact is decode/verify only).
route_smem: Inkling2's CUDA route_metadata kernel (one CTA, all ids in shared memory) needs more than GB10's 99 KB
  from ~3,072 ids (forwards of ~384-512 tokens) and fails with "Cannot set route metadata shared memory", killing
  the scheduler. On SM12x, calls above 2,048 ids take the module's Triton kernel (correct at every size on GB10).
fp8_kv (--kv-cache-dtype fp8_e4m3; a BF16 pool takes none of this): SGLang's fp8_e4m3 pool stores K/V as a plain
  saturating e4m3 cast (one byte per element, no scales; Inkling's K is L2-clipped and V RMS-normed, far below 448).
  The store is stock; SM12x has no reader. fp8_kv_query keeps queries BF16 (stock casts decode queries to the cache
  dtype for FA3's FP8 kernel); triton_decode sends an e4m3 cache to patches/paged_decode.py's e4m3 kernel (a cast
  after the K/V loads); fa4_sm12x_args hands FA4 a BF16 copy of the rows a prefill reads.
roce (SPARK_ROCE_AR=1): b12x one-shot RDMA all-reduce/all-gather for the "tp" GroupCoordinator (patches/
  roce_collectives.py, from the DeepSeek-V4.1 overlay). At TP4/EP4 SGLang's attention-TP and MoE-EP groups are that
  same coordinator, so every Inkling collective up to SPARK_ROCE_AR_MAX skips NCCL.
timers (SPARK_TIMERS=1, diagnosis windows only): CUDA-event timing of Inkling2's modules per eager forward, logged by
  every rank as one line per forward (category: total ms / calls). Inactive under CUDA-graph capture, so run the
  window with --disable-cuda-graph to time decode. Nested categories overlap (layer includes attn, moe, ...).
graph_timers (SPARK_GRAPH_TIMERS=1 or <categories>, diagnosis windows only): the same wrappers under CUDA-graph
  capture: patches/graph_timers.py records an event pair per region into the graph and reads them after every replay.
trace (SPARK_TRACE=<dir>, diagnosis windows only): patches/spark_trace.py writes, on rank 0, every verify step's routed
  expert ids, draft logit features and acceptance (bench/trace_run.py drives it, bench/trace_sim.py reads it).
gate (SPARK_GATE=<tau>, off by default): patches/spark_gate.py gives every request of a DSpark step a verify row
  count K from the draft's own confidence; the rows >= K route to the anchor row's experts (no expert reads of their
  own) and the accept is capped at K - 1 drafts.
dense_fq (SPARK_DENSE_FQ=<bits>, quality windows only): patches/dense_fq.py rounds the dense BF16 weights (attention
  projections, router, unembedding, the draft's layers) to per-row <bits>-bit integers after loading: what an integer
  weight format would cost in quality, measured before any kernel reads one.
accept_stats (SPARK_ACCEPT_STATS=1, diagnosis windows only): patches/accept_stats.py logs, for sampled requests, the
  tokens per step that token-by-token rejection and block verification would each commit in expectation.
sampled_lab (SPARK_SAMPLED_LAB=1 with the gate and SPARK_GATE_RID=1, diagnosis windows only): patches/sampled_lab.py
  lets a request id choose the gate rule and the draft temperature of its request (both leave a sampled request's
  distribution alone), to measure them side by side in one boot.
dense_i8 (SPARK_DENSE_I8=<parts>, off by default): patches/dense_i8.py keeps int8 codes (one scale per output row)
  beside the chosen dense weights and reads them with a Triton kernel on forwards of up to 64 rows, where a linear
  runs at memory bandwidth: half the bytes, half the time. Parts draft and dhead leave the target's arithmetic alone;
  attn, router and head round the target's weights (dense_fq measures what that costs).
dense_p12 (SPARK_DENSE_P12=<parts>, off by default): patches/dense_p12.py packs the target's BF16 attention
  projections (attn) and unembedding (head) into 12 bits a weight without rounding anything (BF16 weights use few
  distinct exponents in a row; the rare others are kept as exceptions) and reads them with a Triton kernel on
  forwards of up to 64 rows: three quarters of the bytes, the same products. free drops the BF16 copies of attn;
  larger forwards rebuild a weight in a scratch buffer and run stock's code on it. Every weight is checked at load.
attn_check (SPARK_ATTN_CHECK=1, diagnosis windows only): patches/attn_check.py recomputes sampled query rows of every
  prefill attention call and of a sample of decode / verify calls in plain torch from the kernel's own inputs and logs
  how far the kernel's output and LSE are from it (decode needs --disable-cuda-graph).
tp_shard (SPARK_TP_SHARD=1 | local, off by default, windows only until measured; needs SPARK_FI_MOE=1):
  patches/tp_shard.py loads the NVFP4 expert layers with every expert on every rank, cut to a quarter of its
  intermediate width, instead of a quarter of the experts a rank: no rank waits for the busiest one. "1" exchanges
  each row's amax between the ranks (the checkpoint's quantized operands, rounding-level differences), "local" does not.
  SPARK_TP_SHARD_HALF=1 (with "1"): the down projection's partial sums as FP16, summed without the BF16 rounding.
  SPARK_TP_SHARD_FP8=1: the MXFP8 expert layers sharded the same way (stock's Hopper code, every expert local).
fi_mxfp8 (SPARK_FI_MXFP8=1, off by default, not bit-identical to Marlin; needs SPARK_FI_MOE=1 and
  SPARK_TP_SHARD_FP8=1): patches/fi_mxfp8.py runs the sharded MXFP8 expert layers through FlashInfer's native MXFP8
  grouped GEMM instead of Marlin (the same operands, another order of sums: not bit-identical).
route_balance (SPARK_ROUTE_BALANCE=1, diagnosis windows only, garbage text): patches/route_diag.py renames the routed
  experts of every forward so the ranks read equal shares: with graph_timers, what the wait for the busiest rank costs.
reasoning_cache (SPARK_REASONING_CACHE=1 or a flag file's path, off by default): patches/reasoning_cache.py keeps the
  reasoning of every answer in the API server and puts it back into history turns a client sends without it (the
  template renders an earlier turn's thinking only when it is sent; without it later questions get earlier answers).
wmark_boost (SPARK_WMARK_BOOST=1, off by default): patches/wmark_boost.py gives the KV pool sizing the host's
  MemAvailable without the kernel's watermark boost (a reclaim target of a few minutes that hides up to 2.24 GB of
  page cache here at the second the pool is sized, on some boots and not on others).
draft_kv_fp8 (SPARK_DRAFT_KV_FP8=1, off by default; with --kv-cache-dtype fp8_e4m3): SGLang gives the DSpark draft a
  BF16 KV pool beside an FP8 target pool, because FA4 (the draft block's attention) reads K/V in the query dtype
  only. With this the draft's pool is e4m3 as well (half its bytes) and fa4_sm12x_args sends the draft block's
  reads to patches/draft_decode.py (the image's Triton decode kernel rebuilt for an e4m3 pool: every valid key, no
  bias). The draft's K/V are rounded when stored; the target's arithmetic is untouched.
  SPARK_DRAFT_KV_STATS=1 (diagnosis windows only): patches/draft_kv_stats.py logs what e4m3 does to those values.
prefill_glue (SPARK_PREFILL_GLUE=<parts>, off by default; parts epilogue,stream,gate,qkvr,relbias):
  patches/prefill_glue.py takes passes over memory out of the per-layer glue of a plain prefill chunk: the layer's
  closing convolution + residual add in one kernel (fused_epilogue.py), attn_res + RMSNorm in one kernel
  (fused_stream.py), FA4's output read as BF16 by gate_sink, a global layer's k and v projected through their own
  slice of the qkvr weight, and the relative-position bias written once, already sheared (fused_relbias.py; the
  FA4 wrapper below hands it over as presheared). Every part keeps stock's operations and their order: the same
  bits. Decode, verification and mixed batches run stock's code. SPARK_PREFILL_GLUE_CHECK=1 (diagnosis windows
  only) runs stock's path beside every fused call and counts differing values. Measured on 4 nodes (all
  five parts, the default row floors): cold prefill +11% at 30,000 tokens, +10% at 120,000; 0 differing values
  under CHECK on every rank.
fa4_bias_far (SPARK_FA4_BIAS_FAR=f1|f2, off by default; f2 saves 2% of a 120,000-token prefill, 3.6% at 300,000):
  patches/fa4_bias_far.py derives the SM120 FA4 forward kernel with the far key blocks of a call with a relative
  bias compiled without the bias code (the same multiplies in the same order: the same bits), and gives the derived
  kernels their own persistent cache directory.
stable_kernels (SPARK_STABLE_KERNELS=tau,gate_sink, off by default): patches/stable_kernels.py binds two of the
  image's Triton kernels with the batch's row count and strides as arguments where the image compiles them in, so a
  prefill of a new size no longer compiles and loads a CUDA module in the middle of serving (the same bits).
kernel_load_guard (SPARK_KERNEL_LOAD_GUARD=<retries>, off by default): patches/kernel_load_guard.py loads a Triton
  kernel again when the driver refuses it with "operation not permitted", where the rank died before.
SPARK_FA_DEBUG=1: also log the tensors of a failing SM12x FA4 call.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import importlib.abc
import importlib.util
import os
import sys

_HOOKS = {}


def _on_import(name):
    def deco(fn):
        _HOOKS.setdefault(name, []).append(fn)
        return fn
    return deco


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _HOOKS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig = spec.loader.exec_module

        def exec_module(module, orig=orig, fullname=fullname):
            orig(module)
            for hook in _HOOKS.get(fullname, []):
                hook(module)
        spec.loader.exec_module = exec_module
        return spec


def _desc(v):
    import torch
    if isinstance(v, torch.Tensor):
        return f"{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
    return None


@_on_import("sglang.kernels.ops.attention.flash_attention_v4_sm120")
def fa4_sm12x_args(mod):
    """fa4_out_dtype + rel_bias_flags (see the module docstring)."""
    import threading
    debug = os.environ.get("SPARK_FA_DEBUG") == "1"
    state = threading.local()
    inner = mod._flash_attn_varlen_func

    def interface_call(*args, **kwargs):
        if _PREFILL_GLUE:  # a projection input that no wrapped call took becomes stock's tensor
            import prefill_glue
            prefill_glue.undefer(kwargs)
        if getattr(state, "presheared", False) and kwargs.get("rel_bias") is not None:
            kwargs["rel_bias_presheared"] = True
        return inner(*args, **kwargs)

    interface_call.__wrapped__ = inner
    mod._flash_attn_varlen_func = interface_call

    for name in ("flash_attn_varlen_func", "flash_attn_with_kvcache"):
        stock = getattr(mod, name)

        def wrapped(*args, __stock=stock, __name=name, **kwargs):
            q = kwargs.get("q", args[0] if args else None)
            scale_after_add = kwargs.pop("rel_bias_scale_after_add", False)
            presheared = kwargs.pop("rel_bias_presheared", False)
            if _DRAFT_KV_FP8 and __name == "flash_attn_with_kvcache" and not args and _e4m3(kwargs.get("k_cache")):
                import draft_decode
                if draft_decode.handles(kwargs):  # draft_kv_fp8: the draft's block reads its e4m3 pool in place
                    return _timed("draft_kv", draft_decode.fa4_call)(**kwargs)
                draft_decode.not_handled(kwargs)  # logged once; the call goes on as any e4m3 cache does
            out = kwargs.get("out")
            widen = out is not None and q is not None and out.dtype != q.dtype
            if widen:
                kwargs["out"] = None
            check = (_ATTN_CHECK and __name == "flash_attn_with_kvcache" and not presheared
                     and not getattr(state, "nested", False) and kwargs.get("rel_bias") is not None)
            if check:  # the pool and page table as the model passed them, before the fp8 view replaces them
                pool = (kwargs["k_cache"], kwargs["v_cache"], kwargs["page_table"])
            # the timers keep the global layers apart (no window): their time grows with the prefix
            glob = "" if (kwargs.get("window_size") or (-1, -1))[0] >= 0 else "_g"
            if _e4m3(kwargs.get("k_cache")):  # fp8_kv: the SM12x FA4 forward reads BF16 K/V only
                import fp8_kv
                kwargs["k_cache"], kwargs["v_cache"], kwargs["page_table"] = _timed("fp8_view" + glob, fp8_kv.bf16_view)(
                    kwargs["k_cache"], kwargs["v_cache"], kwargs["page_table"], kwargs["cache_seqlens"],
                    kwargs.get("cu_seqlens_q"), (kwargs.get("window_size") or (-1, -1))[0],
                    q.shape[0] if q.dim() == 3 else q.shape[0] * q.shape[1])
            took = None
            if _PREFILL_GLUE and not presheared and kwargs.get("rel_bias") is not None:
                import prefill_glue
                # relbias: the projection's input, handed back, becomes the sheared bias (or stock's tensor)
                took = prefill_glue.fa4_relbias(__name, __stock, args, kwargs, getattr(state, "nested", False),
                                                _timed, glob)
                presheared = took is not None
            owner = presheared and not getattr(state, "presheared", False)
            if owner:
                state.presheared = True
            nested = getattr(state, "nested", False)  # flash_attn_with_kvcache calls flash_attn_varlen_func: time once
            state.nested = True
            try:
                result = (__stock if nested else _timed("fa4" + glob, __stock))(*args, **kwargs)
            except Exception:
                if debug:
                    parts = [f"{k}={_desc(v)}" for k, v in kwargs.items() if _desc(v)]
                    parts += [f"{k}={v!r}" for k, v in kwargs.items()
                              if k in ("causal", "window_size", "softcap", "num_splits", "pack_gqa", "max_seqlen_q",
                                       "max_seqlen_k")]
                    parts += [f"scale_after_add={scale_after_add}", f"presheared={presheared}"]
                    print(f"[spark-gb10] {__name} failed: " + " ".join(parts), file=sys.stderr, flush=True)
                raise
            finally:
                state.nested = nested
                if owner:
                    state.presheared = False
            if check:
                import attn_check
                attn_check.prefill(q, *pool, kwargs.get("cache_seqlens"), kwargs.get("cu_seqlens_q"),
                                   kwargs["rel_bias"] if took is None else took.unsheared(),
                                   kwargs.get("softmax_scale"), kwargs.get("window_size"), result)
            if took is not None:
                took.verify(result)
            if not widen:
                return result
            widened = result[0] if isinstance(result, tuple) else result
            _timed("fa4_widen", out.copy_)(widened.view(out.shape))
            return (out, *result[1:]) if isinstance(result, tuple) else out

        wrapped.__wrapped__ = stock
        setattr(mod, name, wrapped)


def _sm12x():
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


# SPARK_PAGED_DECODE=0: decode/verify attention with SGLang's kernel (one page lookup per tile, wrong at page size 1).
_PAGED_DECODE = os.environ.get("SPARK_PAGED_DECODE", "1") == "1"


# SPARK_ATTN_CHECK=1 (diagnosis windows only): compare the attention kernels with a torch reference, patches/attn_check.py.
_ATTN_CHECK = os.environ.get("SPARK_ATTN_CHECK") == "1"


def _e4m3(tensor):
    return tensor is not None and str(tensor.dtype) == "torch.float8_e4m3fn"


# SPARK_DRAFT_KV_FP8=1 (unset = off): the draft's KV pool in e4m3 like the target's, read by patches/draft_decode.py.
_DRAFT_KV_FP8 = os.environ.get("SPARK_DRAFT_KV_FP8") == "1"


if _DRAFT_KV_FP8:
    @_on_import("sglang.srt.mem_cache.kv_cache_dtype")
    def draft_kv_fp8(mod):
        """The draft runner, the pool budget and the pool placement all ask this function for the draft's KV dtype;
        stock answers the model dtype for an fa4 draft. Answer what it gives a draft whose backend reads FP8."""
        import logging
        stock = mod.configure_kv_cache_dtype
        told = []

        def configure_kv_cache_dtype(**kwargs):
            asked = kwargs.get("speculative_draft_kv_cache_dtype") or kwargs.get("server_args_kv_cache_dtype")
            if (kwargs.get("is_draft_worker") and kwargs.get("is_dflash") and asked == "fp8_e4m3"
                    and kwargs.get("speculative_draft_attention_backend") == "fa4" and _sm12x()):
                import torch
                import draft_decode  # noqa: F401  derive the kernel now: a changed image fails the boot
                resolved, dtype = stock(**{**kwargs, "speculative_draft_attention_backend": "fa4+draft_decode"})
                if dtype == torch.float8_e4m3fn:
                    if not told:
                        told.append(1)
                        logging.getLogger("spark.draft_kv_fp8").warning(
                            "[draft_kv_fp8] the draft's KV pool is %s (read by patches/draft_decode.py)", dtype)
                    return resolved, dtype
            return stock(**kwargs)

        configure_kv_cache_dtype.__wrapped__ = stock
        mod.configure_kv_cache_dtype = configure_kv_cache_dtype

    if os.environ.get("SPARK_DRAFT_KV_STATS") == "1":
        @_on_import("sglang.srt.models.inkling2_mtp")
        def draft_kv_stats(mod):
            import draft_kv_stats as stats
            stats.install(mod)


@_on_import("sglang.srt.layers.attention.flashattention_backend")
def fp8_kv_query(mod):
    if not _sm12x():
        return
    import torch
    cls = mod.FlashAttentionBackend
    stock = cls.prepare_paged_mha_query

    def prepare_paged_mha_query(self, q, q_rope, k_rope, layer, **kwargs):
        if self.kv_cache_dtype == torch.float8_e4m3fn and not self.kv_cache_is_mxfp8 and layer.k_scale is None:
            return q, q_rope, k_rope, None, None
        return stock(self, q, q_rope, k_rope, layer, **kwargs)

    prepare_paged_mha_query.__wrapped__ = stock
    cls.prepare_paged_mha_query = prepare_paged_mha_query


@_on_import("sglang.srt.utils.common")
def wmark_boost(mod):
    if os.environ.get("SPARK_WMARK_BOOST") != "1" or not hasattr(mod, "psutil"):
        return
    import wmark_boost as wb
    mod.psutil = wb.Psutil(mod.psutil)


@_on_import("sglang.kernels.ops.inkling2.comm")
def nccl_collectives(mod):
    if not _sm12x():
        return
    import torch

    def all_reduce(x, group, *, multicast=False):
        if group.world_size == 1 or x.numel() == 0:
            return x
        return _timed("allreduce", group.all_reduce)(x.contiguous())

    def multicast_all_reduce_input(group, shape, dtype):
        return torch.empty(shape, dtype=dtype, device=torch.device("cuda", torch.cuda.current_device()))

    def multicast_all_reduce(x, group):
        return _timed("allreduce", group.all_reduce)(x)

    def oneshot_all_reduce_input(group, rows, cols, dtype, tag=""):
        # Only the expert partials above 512 tokens ask with tag "moe" (BF16, as on Hopper); None keeps FP32.
        if tag == "moe" and _MOE_BF16:
            return torch.empty((rows, cols), dtype=dtype, device=torch.device("cuda", torch.cuda.current_device()))
        return None

    def oneshot_all_reduce(x, group, acc_f32=False, tag=""):
        return _timed("allreduce", group.all_reduce)(x)

    for fn in (all_reduce, multicast_all_reduce_input, multicast_all_reduce, oneshot_all_reduce_input,
               oneshot_all_reduce):
        fn.__wrapped__ = getattr(mod, fn.__name__)
        setattr(mod, fn.__name__, fn)


@_on_import("sglang.srt.models.inkling_common.inkling2_moe")
def moe_marlin(mod):
    cls = mod.Inkling2Experts
    stock = cls.__init__

    def __init__(self, *args, **kwargs):
        sharded = False
        if _TP_SHARD and _sm12x():
            if os.environ.get("SPARK_FI_MOE", "0") != "1":
                raise RuntimeError("SPARK_TP_SHARD needs SPARK_FI_MOE=1 (only patches/fi_moe.py runs sharded experts)")
            import tp_shard
            sharded = tp_shard.construct(mod, stock, self, args, kwargs)
        if not sharded:
            stock(self, *args, **kwargs)
        if _sm12x():
            self.sm90 = True
        if _TRACE:
            import spark_trace
            spark_trace.register(self)

    __init__.__wrapped__ = stock
    cls.__init__ = __init__

    # fi_moe (SPARK_FI_MOE=1): NVFP4 expert layers keep the checkpoint weights, swizzle their block scales in place
    # and run patches/fi_moe.py (FlashInfer NVFP4 grouped GEMMs); MXFP8 layers keep the Marlin path.
    if os.environ.get("SPARK_FI_MOE", "0") == "1" and _sm12x():
        import torch
        import fi_moe
        if _TIMERS:
            fi_moe.TIMED = _timed
        stock_swizzle = cls.swizzle_nvfp4_scales

        def swizzle_nvfp4_scales(self):
            if getattr(self, "_spark_fi", False) or getattr(self, "_spark_mx", None) is not None:
                return  # already in the 128x4 layout (same shape as the checkpoint's: never swizzle twice)
            if not self._nvfp4_scale_shapes or set(self._nvfp4_scale_shapes) != {"w13", "w2"}:
                if _FI_MXFP8 and getattr(self, "_spark_shard_fp8", None) is not None:
                    import fi_mxfp8
                    if _TIMERS:
                        fi_moe.TIMED = _timed
                    if fi_mxfp8.adopt(self):  # False (reason logged): stock's Marlin repack below
                        self._identity = torch.arange(1 << 17, dtype=torch.int32, device=self.w13_weight.device)
                        return
                done = stock_swizzle(self)
                if _TP_SHARD and getattr(self, "_spark_shard_fp8", None) is not None:
                    import tp_shard
                    tp_shard.fp8_repacked(self)  # a log line: what stock's repack left of the shard
                return done
            for name, checkpoint_shape in self._nvfp4_scale_shapes.items():
                scale = getattr(self, name + "_scale")
                if scale.is_cuda and tuple(scale.shape) == checkpoint_shape:
                    scale.data = fi_moe.swizzle_scales_128x4(scale.data)
            self._spark_fi = True
            self._identity = torch.arange(1 << 17, dtype=torch.int32, device=self.w13_weight.device)
            if _TP_SHARD and getattr(self, "_spark_shard", None) is not None:
                import tp_shard
                self._spark_half = tp_shard.half_scales(self, fi_moe)  # None unless SPARK_TP_SHARD_HALF=1

        swizzle_nvfp4_scales.__wrapped__ = stock_swizzle
        cls.swizzle_nvfp4_scales = swizzle_nvfp4_scales
        stock_fwd = cls.forward

        def fi_forward(self, x, weights, ids, combine=True):
            if _ROUTE_STATS:
                _route_stats(ids, self.ep_rank * self.w13_weight.shape[0], self.w13_weight.shape[0])
            shard = getattr(self, "_spark_shard", None)  # (rank, world): patches/tp_shard.py loaded this layer
            mx = getattr(self, "_spark_mx", None)  # patches/fi_mxfp8.py adopted this (sharded MXFP8) layer
            if mx is not None and x.shape[0]:
                import fi_mxfp8
                bf16 = _MOE_BF16 and ids.shape[0] > mod.BF16_COMBINE_MIN_TOKENS
                sums = _timed("x_fi_mxfp8", fi_mxfp8.experts_forward)(
                    x, ids, weights, self.ep_rank * self.w13_weight.shape[0], self.w13_weight, mx[0],
                    self.w2_weight, mx[1], (self.gate_cap, self.up_cap),
                    out_dtype=torch.bfloat16 if bf16 else torch.float32)
                combined = mod.multicast_all_reduce(sums, mod.get_parallel().moe_ep_group)
                if not combine:
                    return mod.RoutedOutput(combined, self._identity[: ids.shape[0]], 1)
                return combined.to(x.dtype)
            if not getattr(self, "_spark_fi", False) or x.shape[0] == 0:
                if shard is not None and x.shape[0]:
                    raise RuntimeError("SPARK_TP_SHARD: a sharded expert layer has no path but fi_moe's")
                return stock_fwd(self, x, weights, ids, combine)
            start = self.ep_rank * self.w13_weight.shape[0]  # sharded: every expert is local (ep_rank 0)
            bf16 = _MOE_BF16 and ids.shape[0] > mod.BF16_COMBINE_MIN_TOKENS
            group = mod.get_parallel().moe_ep_group
            exchange = None
            if shard is not None and _TP_SHARD != "local":
                exchange = getattr(self, "_spark_exchange", None)
                if exchange is None:
                    import tp_shard
                    exchange = self._spark_exchange = tp_shard.amax_exchange(
                        group, *shard, reduce=_timed("amax", group.all_reduce))
            sums = _timed("x_fi_experts", fi_moe.experts_forward)(
                x, ids, weights, start, self.w13_weight, self.w13_scale, self.w13_amax,
                self.w13_outer_scale_denominator, self.w2_weight, self.w2_scale, self.w2_amax,
                self.w2_outer_scale_denominator, (self.gate_cap, self.up_cap),
                out_dtype=torch.bfloat16 if bf16 else torch.float32, shard=shard, exchange=exchange,
                half=getattr(self, "_spark_half", None))
            combined = mod.multicast_all_reduce(sums, group)
            if not combine:
                return mod.RoutedOutput(combined, self._identity[: ids.shape[0]], 1)
            return combined.to(x.dtype)

        fi_forward.__wrapped__ = stock_fwd
        cls.forward = fi_forward

    # The Marlin repack runs at the end of load_weights; --load-format dummy (the smoke test) never calls it, so
    # repack on the first forward instead (eager, before any graph capture). A real load has already repacked.
    stock_forward = cls.forward

    def forward(self, x, *args, **kwargs):
        if self.sm90 and not self._marlin and not getattr(self, "_spark_repacked", False):
            self._spark_repacked = True
            self.swizzle_nvfp4_scales()
        if _GATE and len(args) >= 2:
            import spark_gate
            args = (args[0], spark_gate.mask_ids(args[1]), *args[2:])
        if _ROUTE_BALANCE and len(args) >= 2:
            import route_diag
            args = (args[0], _timed("relabel", route_diag.balance)(args[1], self.num_global_experts,
                                                                    mod.get_parallel().moe_ep_size), *args[2:])
        if _TRACE and len(args) >= 2:
            import spark_trace
            spark_trace.note_ids(self, args[1])
        return stock_forward(self, x, *args, **kwargs)

    forward.__wrapped__ = stock_forward
    cls.forward = forward


@_on_import("sglang.kernels.ops.inkling2.attention")
def triton_decode(mod):
    if not _sm12x():
        return
    stock = mod.rel_bias_partials
    if _PAGED_DECODE:
        import paged_decode  # derive the kernel now: a changed image fails the boot, not the first request

    def rel_bias_partials(q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, out, *,
                          window_size_left, verify):
        if _e4m3(k_cache):
            import paged_decode as fp8_decode  # not `paged_decode`: that would shadow the module for the BF16 branch
            decode_partials = fp8_decode.decode_partials_fp8
        elif _PAGED_DECODE:
            decode_partials = paged_decode.decode_partials
        else:
            from sglang.kernels.ops.inkling2.decode_attention_sm90 import decode_partials
        partials = _timed("decode_attn", decode_partials)(q, k_cache, v_cache, page_table, seqused_k, rel_bias,
                                                          softmax_scale, window_size_left=window_size_left)
        if _ATTN_CHECK:
            import attn_check
            attn_check.decode(q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, window_size_left,
                              partials)
        return partials

    rel_bias_partials.__wrapped__ = stock
    mod.rel_bias_partials = rel_bias_partials


@_on_import("sglang.srt.models.inkling_common.inkling2_attn")
def attn_hopper_dispatch(mod):
    if not _sm12x():
        return
    stock = mod.get_platform

    class _Sm90View:
        """The platform as Inkling2Attention's dispatch should see it on SM12x: Hopper-like."""

        def __init__(self, platform):
            self._platform = platform

        is_sm90 = True

        def __getattr__(self, name):
            return getattr(self._platform, name)

    def get_platform():
        return _Sm90View(stock())

    get_platform.__wrapped__ = stock
    mod.get_platform = get_platform


@_on_import("sglang.srt.entrypoints.http_server")
def served_aliases(module):
    import logging
    aliases = [a.strip() for a in os.environ.get("SPARK_SERVED_ALIASES", "").split(",") if a.strip()]
    if not aliases:
        return
    app, ORJSONResponse, ModelCard, ModelList = module.app, module.ORJSONResponse, module.ModelCard, module.ModelList

    def names():
        base = module._global_state.tokenizer_manager.served_model_name
        return [base] + [a for a in aliases if a != base]

    async def available_models():
        tm = module._global_state.tokenizer_manager
        ns = names()
        return ModelList(data=[ModelCard(id=n, root=ns[0], max_model_len=tm.model_config.context_len) for n in ns])

    async def retrieve_model(model: str):
        ns = names()
        if model not in ns:
            return ORJSONResponse(status_code=404, content={"error": {
                "message": f"The model '{model}' does not exist", "type": "invalid_request_error",
                "param": "model", "code": "model_not_found"}})
        tm = module._global_state.tokenizer_manager
        return ModelCard(id=model, root=ns[0], max_model_len=tm.model_config.context_len)

    for r in list(app.router.routes):
        if getattr(r, "path", None) in ("/v1/models", "/v1/models/{model:path}"):
            app.router.routes.remove(r)
    app.add_api_route("/v1/models", available_models, methods=["GET"], response_class=ORJSONResponse)
    app.add_api_route("/v1/models/{model:path}", retrieve_model, methods=["GET"], response_class=ORJSONResponse)
    logging.getLogger(__name__).info("Served-model aliases: %s", ", ".join(aliases))

    # Ollama-style discovery: SGLang's /api/tags lists only the served name and its /api/show rejects a body without
    # `model`. The answers are still built by SGLang's OllamaServing; a failure here leaves its two routes in place.
    tags_path = os.environ.get("SGLANG_OLLAMA_TAGS_ROUTE", "/api/tags")
    show_path = os.environ.get("SGLANG_OLLAMA_SHOW_ROUTE", "/api/show")
    stock = [r for r in app.router.routes if getattr(r, "path", None) in (tags_path, show_path)]
    try:
        Request = module.Request

        async def ollama_tags(raw_request: Request):
            tags = raw_request.app.state.ollama_serving.get_tags()
            one = tags.models[0]
            tags.models = [one.model_copy(update={"name": n, "model": n, "details": {**(one.details or {}), "family": n}})
                           for n in names()]
            return tags

        async def ollama_show(raw_request: Request):
            try:
                body = await raw_request.json()
            except Exception:
                body = None
            model = (body.get("model") or body.get("name")) if isinstance(body, dict) else None
            if not isinstance(model, str) or not model:
                return ORJSONResponse(status_code=400, content={"error": "model is required"})
            return raw_request.app.state.ollama_serving.get_show(model)

        for r in stock:
            app.router.routes.remove(r)
        app.add_api_route(tags_path, ollama_tags, methods=["GET"])
        app.add_api_route(show_path, ollama_show, methods=["POST"])
    except Exception:
        for r in list(app.router.routes):
            if getattr(r, "path", None) in (tags_path, show_path):
                app.router.routes.remove(r)
        app.router.routes.extend(stock)
        logging.getLogger(__name__).exception("Ollama alias routes not installed, SGLang's are kept")


@_on_import("sglang.kernels.ops.inkling2.route")
def route_smem(mod):
    if not _sm12x():
        return
    stock = mod.route_metadata

    def route_metadata(ids, *args, **kwargs):
        if ids.numel() <= 2048:
            return stock(ids, *args, **kwargs)
        saved = mod.ROUTE_CUDA_MIN_IDS
        mod.ROUTE_CUDA_MIN_IDS = mod.ROUTE_MAX_IDS
        try:
            return stock(ids, *args, **kwargs)
        finally:
            mod.ROUTE_CUDA_MIN_IDS = saved

    route_metadata.__wrapped__ = stock
    mod.route_metadata = route_metadata


@_on_import("sglang.srt.distributed.parallel_state")
def roce(mod):
    import roce_collectives
    roce_collectives.install(mod)


@_on_import("sglang.srt.model_executor.model_runner")
def roce_health(mod):
    import roce_collectives
    roce_collectives.install_health(mod)


_TIMERS = os.environ.get("SPARK_TIMERS") == "1"
# SPARK_GRAPH_TIMERS=1 or <categories> (diagnosis windows only): region times inside the captured decode graphs,
# patches/graph_timers.py.
_GRAPH_TIMERS = bool(os.environ.get("SPARK_GRAPH_TIMERS"))
# Expert partials above 512 tokens leave each rank as BF16 (half the all-reduce payload), as the stock Hopper path
# sends them; the sum across ranks is then NCCL's BF16 sum instead of Hopper's FP32 accumulation.
_MOE_BF16 = os.environ.get("SPARK_MOE_BF16", "1") == "1"
# SPARK_ROUTE_STATS=1 (with SPARK_TIMERS=1): per verify forward, log for every layer how many of this rank's experts
# the first K of each request's 8 rows reach, K = 1..8 (what a narrower verify would read).
_ROUTE_STATS = _TIMERS and os.environ.get("SPARK_ROUTE_STATS") == "1"
_T = {"events": [], "depth": 0, "route": []}
# SPARK_TRACE=<dir> (diagnosis windows only): patches/spark_trace.py records every verify step's routed expert ids,
# draft logit features and acceptance on rank 0.
_TRACE = bool(os.environ.get("SPARK_TRACE"))
# SPARK_GATE=<tau> (0 or unset = off): confidence-gated verify rows, patches/spark_gate.py.
_GATE = float(os.environ.get("SPARK_GATE") or 0) > 0
# SPARK_DENSE_FQ=<bits> (0 or unset = off; quality windows only): fake-quantized dense weights, patches/dense_fq.py.
_DENSE_FQ = int(os.environ.get("SPARK_DENSE_FQ") or 0) > 0
# SPARK_DENSE_I8=<parts> (unset or empty = off): int8 dense weights on the decode path, patches/dense_i8.py.
_DENSE_I8 = bool(os.environ.get("SPARK_DENSE_I8"))
# SPARK_DENSE_P12=<parts> (unset or empty = off): exact 12-bit packing of the target's dense weights on the decode
# path, patches/dense_p12.py.
_DENSE_P12 = bool(os.environ.get("SPARK_DENSE_P12"))
# SPARK_TP_SHARD=1 | local (unset or empty = off; needs SPARK_FI_MOE=1): NVFP4 expert layers sharded by intermediate
# width instead of by expert, patches/tp_shard.py.
_TP_SHARD = os.environ.get("SPARK_TP_SHARD") or ""
# SPARK_FI_MXFP8=1 (needs SPARK_FI_MOE=1 and SPARK_TP_SHARD_FP8=1): the sharded MXFP8 expert layers on FlashInfer's
# MXFP8 grouped GEMM instead of Marlin, patches/fi_mxfp8.py.
_FI_MXFP8 = os.environ.get("SPARK_FI_MXFP8") == "1"
# SPARK_ROUTE_BALANCE=1 (diagnosis windows only, the text is garbage): patches/route_diag.py.
_ROUTE_BALANCE = os.environ.get("SPARK_ROUTE_BALANCE") == "1"
# SPARK_ACCEPT_STATS=1 (diagnosis windows only): expected accepted lengths of sampled steps, patches/accept_stats.py.
_ACCEPT_STATS = os.environ.get("SPARK_ACCEPT_STATS") == "1"
# SPARK_SAMPLED_LAB=1 (diagnosis windows only; needs SPARK_GATE and SPARK_GATE_RID=1): a request id chooses the gate
# rule and the draft temperature of its request, patches/sampled_lab.py.
_SAMPLED_LAB = _GATE and os.environ.get("SPARK_SAMPLED_LAB") == "1" and os.environ.get("SPARK_GATE_RID") == "1"
# SPARK_REASONING_CACHE=1 or <flag file> (unset = off): history turns get their reasoning back, patches/reasoning_cache.py.
_REASONING_CACHE = bool(os.environ.get("SPARK_REASONING_CACHE"))
# SPARK_PREFILL_GLUE=<parts> (unset or empty = off): the per-layer glue of plain prefill chunks in fewer passes,
# patches/prefill_glue.py.
_PREFILL_GLUE = bool(os.environ.get("SPARK_PREFILL_GLUE"))


def _route_stats(ids, start, num_local):
    import torch
    rows = ids.shape[0]
    if rows % 8 or rows > 64 or torch.cuda.is_current_stream_capturing():
        return
    local = ids.to(torch.int64) - start
    local = torch.where((local >= 0) & (local < num_local), local, num_local).flatten()
    pos = (torch.arange(rows, device=ids.device) % 8)[:, None].expand(rows, ids.shape[1]).flatten()
    first = torch.full((num_local + 1,), 8, dtype=torch.int64, device=ids.device)
    first.scatter_reduce_(0, local, pos, "amin")
    _T["route"].append((first[:num_local, None] < torch.arange(1, 9, device=ids.device)[None]).sum(0))


def _timed(category, fn):
    """Wrap fn: record CUDA events around each call (when SPARK_TIMERS=1 and not capturing); fn itself otherwise.
    Under capture with SPARK_GRAPH_TIMERS the call goes through graph_timers.call (events inside the graph)."""
    if not _TIMERS and not _GRAPH_TIMERS:
        return fn
    import torch

    def wrapper(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            if _GRAPH_TIMERS:
                import graph_timers
                return graph_timers.call(category, fn, args, kwargs)
            return fn(*args, **kwargs)
        if not _TIMERS:
            return fn(*args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return fn(*args, **kwargs)
        finally:
            end.record()
            _T["events"].append((category, start, end))

    wrapper.__wrapped__ = fn
    return wrapper


def _timed_top(label, fn):
    """The outermost timed call of a forward: synchronize, report every category, reset."""
    import logging
    import torch
    inner = _timed(label, fn)

    def wrapper(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            return inner(*args, **kwargs)
        if not _TIMERS:
            return fn(*args, **kwargs)
        _T["depth"] += 1
        try:
            return inner(*args, **kwargs)
        finally:
            _T["depth"] -= 1
            if _T["depth"] == 0:
                torch.cuda.synchronize()
                agg = {}
                for cat, a, b in _T["events"]:
                    ms, n = agg.get(cat, (0.0, 0))
                    agg[cat] = (ms + a.elapsed_time(b), n + 1)
                _T["events"].clear()
                fb = kwargs.get("forward_batch") or next((a for a in args if hasattr(a, "forward_mode")), None)
                mode = getattr(getattr(fb, "forward_mode", None), "name", "?")
                toks = getattr(getattr(fb, "input_ids", None), "shape", ["?"])[0]
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                body = " ".join(f"{c}={ms:.2f}/{n}" for c, (ms, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]))
                logging.getLogger("spark_timers").warning("[timers] rank=%d %s mode=%s tokens=%s %s", rank, label, mode,
                                                          toks, body)
                if _T["route"]:
                    logging.getLogger("spark_timers").warning("[route] rank=%d tokens=%s %s", rank, toks,
                                                              torch.stack(_T["route"]).tolist())
                    _T["route"].clear()

    wrapper.__wrapped__ = fn
    return wrapper


if _TRACE:
    @_on_import("sglang.srt.speculative.dspark_components.dspark_observability")
    def trace_steps(mod):
        import spark_trace
        stock = mod.DsparkStepObservers.observe_verify_step

        def observe_verify_step(self, **kwargs):
            stock(self, **kwargs)
            spark_trace.observe(kwargs)

        observe_verify_step.__wrapped__ = stock
        mod.DsparkStepObservers.observe_verify_step = observe_verify_step


if _GATE:
    @_on_import("sglang.srt.speculative.dspark_components.dspark_verify")
    def gate_verify(mod):
        import spark_gate
        spark_gate.install_verify(mod)

    @_on_import("sglang.srt.speculative.dspark_components.dspark_draft_sampler")
    def gate_sampler(mod):
        import spark_gate
        spark_gate.install_sampler(mod)

    @_on_import("sglang.srt.speculative.dspark_components.dspark_draft")
    def gate_proposer(mod):
        import spark_gate
        spark_gate.install_proposer(mod)

    @_on_import("sglang.srt.models.inkling2")
    def gate_model(mod):
        import spark_gate
        spark_gate.install_model(mod)


if _SAMPLED_LAB:  # after the gate's hooks: the lab wraps what they installed
    @_on_import("sglang.srt.speculative.dspark_components.dspark_draft")
    def sampled_lab_draft(mod):
        import sampled_lab
        sampled_lab.install_draft(mod)

    @_on_import("sglang.srt.speculative.dspark_components.dspark_draft_sampler")
    def sampled_lab_sampler(mod):
        import sampled_lab
        import spark_gate
        sampled_lab.install_draft(mod)
        sampled_lab.install_gate(spark_gate)


if _REASONING_CACHE:
    @_on_import("sglang.srt.entrypoints.openai.serving_inkling")
    def reasoning_history(mod):
        import reasoning_cache
        reasoning_cache.install(mod)


if _ACCEPT_STATS:
    @_on_import("sglang.kernels.ops.speculative.dspark.dspark_accept")
    def accept_stats_kernel(mod):
        import accept_stats
        accept_stats.install(mod)


if _DENSE_FQ:
    @_on_import("sglang.srt.models.inkling2")
    def dense_fq_target(mod):
        import dense_fq
        dense_fq.install_target(mod)

    @_on_import("sglang.srt.models.inkling2_mtp")
    def dense_fq_draft(mod):
        import dense_fq
        dense_fq.install_draft(mod)


if _DENSE_I8:
    @_on_import("sglang.srt.models.inkling2")
    def dense_i8_target(mod):
        import dense_i8
        dense_i8.install_target(mod)

    @_on_import("sglang.srt.models.inkling2_mtp")
    def dense_i8_draft(mod):
        import dense_i8
        dense_i8.install_draft(mod)

    @_on_import("sglang.srt.layers.quantization.unquant")
    def dense_i8_linear(mod):
        import dense_i8
        dense_i8.install_unquant(mod)

    @_on_import("sglang.srt.models.inkling_common.inkling2_moe")
    def dense_i8_router(mod):
        import dense_i8
        dense_i8.install_router(mod)

    @_on_import("sglang.srt.models.inkling_common.inkling2_attn")
    def dense_i8_attn(mod):
        import dense_i8
        dense_i8.install_attn(mod)


if _DENSE_P12:  # after dense_i8: a module that carries int8 codes is left to it
    @_on_import("sglang.srt.models.inkling2")
    def dense_p12_target(mod):
        import dense_p12
        dense_p12.install_target(mod)

    @_on_import("sglang.srt.layers.quantization.unquant")
    def dense_p12_linear(mod):
        import dense_p12
        dense_p12.install_unquant(mod)

    @_on_import("sglang.srt.models.inkling_common.inkling2_attn")
    def dense_p12_attn(mod):
        import dense_p12
        dense_p12.install_attn(mod)


# SPARK_FA4_BIAS_FAR=f1|f2 (unset or empty = off): the far key blocks of a prefill call with a relative bias,
# patches/fa4_bias_far.py.
_FA4_BIAS_FAR = os.environ.get("SPARK_FA4_BIAS_FAR", "")

if _FA4_BIAS_FAR:
    @_on_import("sglang.kernels.ops.attention.flash_attn.cute.interface")
    def fa4_bias_far(mod):
        if _sm12x():
            import fa4_bias_far  # derive the kernel now: a changed image fails the boot, not the first request
            fa4_bias_far.install(_FA4_BIAS_FAR, mod)


if _PREFILL_GLUE:  # after dense_p12: the qkvr part wraps whatever _project_sources is by then
    @_on_import("sglang.srt.models.inkling_common.attn")
    def prefill_glue_relproj(mod):
        import prefill_glue
        prefill_glue.install_relproj(mod)

    @_on_import("sglang.srt.models.inkling_common.inkling2_attn")
    def prefill_glue_attn(mod):
        import prefill_glue
        prefill_glue.install_attn(mod)

    @_on_import("sglang.srt.models.inkling2")
    def prefill_glue_layer(mod):
        import prefill_glue
        if _TIMERS:
            prefill_glue.TIMED = _timed
        prefill_glue.install_layer(mod)


if _GRAPH_TIMERS:
    @_on_import("sglang.srt.model_executor.runner_backend.full_cuda_graph_backend")
    def graph_timers_backend(mod):
        import graph_timers
        graph_timers.install(mod)


if _TIMERS or _GRAPH_TIMERS:
    def _wrap_method(cls, name, category, top=False):
        fn = getattr(cls, name)
        setattr(cls, name, (_timed_top if top else _timed)(category, fn))

    @_on_import("sglang.srt.models.inkling2")
    def timers_model(mod):
        _wrap_method(mod.Inkling2CausalLLM, "forward", "model", top=True)
        _wrap_method(mod.Inkling2DecoderLayer, "forward", "layer")
        _wrap_method(mod.Inkling2LogitsProcessor, "forward", "logits")

    @_on_import("sglang.srt.models.inkling2_mtp")
    def timers_mtp(mod):
        _wrap_method(mod.Inkling2ForConditionalGenerationMTP, "forward", "draft", top=True)

    @_on_import("sglang.srt.models.inkling_common.inkling2_attn")
    def timers_attn(mod):
        _wrap_method(mod.Inkling2Attention, "forward", "attn")
        for name, cat in (("_project_sources", "a_qkvr"), ("_prologue", "a_prologue"), ("_decode_prologue", "a_dprologue")):
            _wrap_method(mod.Inkling2Attention, name, cat)
        _wrap_method(mod.RelLogitsProj, "forward", "a_relproj")
        _wrap_method(mod.RowParallelLinear, "forward", "a_wo")
        mod.gate_sink = _timed("a_gate_sink", mod.gate_sink)

    @_on_import("sglang.srt.models.inkling_common.inkling2_norm")
    def timers_norm(mod):
        _wrap_method(mod.RMSNorm, "forward", "norm")

    @_on_import("sglang.srt.models.inkling_common.inkling2_moe")
    def timers_moe(mod):
        _wrap_method(mod.Inkling2MoE, "forward", "moe")
        _wrap_method(mod.Inkling2Gate, "forward", "gate")
        _wrap_method(mod.Inkling2Experts, "forward", "experts")

    @_on_import("sglang.kernels.ops.inkling2.marlin_sm90")
    def timers_marlin(mod):
        for name in ("gemm", "gemv", "route", "swiglu_fake_quantize", "swiglu_fake_quantize_mxfp8", "source_partial"):
            setattr(mod, name, _timed("x_" + name, getattr(mod, name)))

    @_on_import("sglang.kernels.ops.inkling2.experts_sm90")
    def timers_experts_sm90(mod):
        for name in ("fake_quantize", "fake_quantize_mxfp8"):
            setattr(mod, name, _timed("x_" + name, getattr(mod, name)))

    @_on_import("sglang.srt.models.inkling_common.inkling2_sconv")
    def timers_sconv(mod):
        _wrap_method(mod.ShortConvolution, "forward", "sconv")

    @_on_import("sglang.srt.models.inkling_common.inkling2_attn_res")
    def timers_attn_res(mod):
        for name in ("attention_input", "mlp_input", "finalize_layer", "finalize_layer_with"):
            _wrap_method(mod.Inkling2AttnResState, name, "attn_res")


_STABLE_KERNELS = os.environ.get("SPARK_STABLE_KERNELS", "")
if _STABLE_KERNELS:
    @_on_import("sglang.kernels.ops.inkling2.attention")
    def stable_kernels_bind(mod):
        import stable_kernels
        stable_kernels.install(mod, stable_kernels.parse(_STABLE_KERNELS))

_KERNEL_LOAD_GUARD = os.environ.get("SPARK_KERNEL_LOAD_GUARD", "")
if _KERNEL_LOAD_GUARD:
    @_on_import("triton.compiler.compiler")
    def kernel_load_guard_wrap(mod):
        import kernel_load_guard
        kernel_load_guard.install(mod, int(_KERNEL_LOAD_GUARD))


sys.meta_path.insert(0, _Finder())
