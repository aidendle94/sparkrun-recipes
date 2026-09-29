"""Wide verify for the lookup drafter: DSpark drafts its trained 5 tokens, the lookup extends the block by up to 10 more,
and only the extended rows pay for the longer verify.

The server runs with a 16-token verify window (--speculative-dspark-block-size 15), so the scheduler reserves 16 KV slots
per request and step, and the target's verify graphs, accept kernels and draft-KV commit all work at width 16. Four
things are narrowed back or made per-row:

  * the draft (SpeculativeAlgorithm width hook, DSparkWorkerV2.__init__ / init_attention_backends /
    _maybe_build_draft_sampler): DSpark's proposer, folded sampler, draft attention backend and draft CUDA graphs run
    at the checkpoint's 5 tokens. At 15 drafted positions its first 5 get worse (its block attention is not causal),
    measured in window 23;
  * the block (lookup_draft.apply_wide): DSpark's 5 tokens, then the lookup continuation of them where one exists;
    sampled rows get one-hot draft distributions at the lookup positions, so sampling stays exact;
  * the verify length (DSparkVerifyPlanner): compact ragged verify with per-row lengths 6 + extension, instead of the
    stock planner, which needs a confidence head this checkpoint does not ship. Steps that may extend copy the lengths
    to the host (one sync) so the step replays the smallest graph that fits; with SPARK_LOOKUP_SYNC_GATE=1 steps with no
    extension demand two steps earlier skip that sync and reuse a cached all-6 layout (stock graphed compact verify has
    no per-step sync either, so an unconditional one costs the overlap scheduler ~1.4 ms per step);
  * the graphs (DecodeCudaGraphRunner): the compact verify graphs are captured at 6b, 6b+10 and 6b+20 tokens for every
    captured batch size b instead of at multiples of 16, so a step without extensions costs what it costs at width 6.

  SPARK_LOOKUP_MODE=wide     selects this path (with SPARK_LOOKUP_DRAFT=1, SGLANG_RAGGED_VERIFY_MODE=compact, block 15)
  SPARK_LOOKUP_BASE          DSpark's own block (default 5)
  SPARK_LOOKUP_EXT_BUDGET    extension tokens per step over the whole batch (default 20; the graph tiers follow it)
  SPARK_LOOKUP_SYNC_GATE     1: sync only on steps whose batch showed extension demand in the last GATE_HOLD steps
                             (read at a lag of two steps, identical on every TP rank); other steps are sync-free
  SPARK_LOOKUP_GATE_HOLD     steps a demand keeps the gate open (default 32)
  SPARK_LOOKUP_GATE_OVERRIDE 1: a request with custom_params {"spark_lookup_force_sync": 1} forces the synced path
                             (measurement only)
  SPARK_LOOKUP_NARROW_SLOTS  1: capture compact verify graphs with tokens // 6 request slots instead of min(tokens, 16),
                             so the in-graph epilogue scatters and argmaxes 16 rows per real request, not per token
  SPARK_LOOKUP_FUSED_C2      1 (needs NARROW_SLOTS): graphs of 6b tokens (every row exactly 6, uniform by routing) use
                             production's fused ratio-2 compressor with draft_len 6 instead of the general torch path

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import inspect
import logging
import os
import textwrap

import lookup_draft as ld

logger = logging.getLogger(__name__)

ACTIVE = ld._ENABLED and ld._MODE == "wide"
EXT_BUDGET = int(os.environ.get("SPARK_LOOKUP_EXT_BUDGET", "20"))
EXT_TIERS = tuple(sorted({*range(0, EXT_BUDGET + 1, 10), EXT_BUDGET})) if EXT_BUDGET >= 10 else (0, EXT_BUDGET)
SYNC_GATE = os.environ.get("SPARK_LOOKUP_SYNC_GATE", "0") == "1"
GATE_HOLD = int(os.environ.get("SPARK_LOOKUP_GATE_HOLD", "32"))
GATE_OVERRIDE = os.environ.get("SPARK_LOOKUP_GATE_OVERRIDE", "0") == "1"
NARROW_SLOTS = os.environ.get("SPARK_LOOKUP_NARROW_SLOTS", "0") == "1"
FUSED_C2 = os.environ.get("SPARK_LOOKUP_FUSED_C2", "0") == "1"
if FUSED_C2 and (not NARROW_SLOTS or any(t and t % (ld._BASE + 1) == 0 for t in EXT_TIERS)):
    logger.warning("lookup wide: SPARK_LOOKUP_FUSED_C2 needs SPARK_LOOKUP_NARROW_SLOTS=1 and extension tiers that are not "
                   "multiples of %d; fused ratio-2 compressor stays off", ld._BASE + 1)
    FUSED_C2 = False


def _base() -> int:
    return ld._BASE


def install_spec_info(module) -> None:
    """speculative.spec_info: the DSpark draft worker's per-request width is DSpark's own block."""
    if not ACTIVE:
        return
    cls = module.SpeculativeAlgorithm
    stock = cls.get_num_tokens_per_req_for_target_verify

    def get_num_tokens_per_req_for_target_verify(self, num_draft_tokens, is_draft_worker):
        if self.is_dspark() and is_draft_worker:
            return _base()
        return stock(self, num_draft_tokens, is_draft_worker)

    cls.get_num_tokens_per_req_for_target_verify = get_num_tokens_per_req_for_target_verify


def install_planner(module) -> None:
    """dspark_components.dspark_planner: compact ragged verify without a confidence head; the layout comes from
    lookup_draft.apply_wide."""
    if not ACTIVE:
        return
    cls = module.DSparkVerifyPlanner
    stock_init, stock_layout = cls.__init__, cls.schedule_layout
    compact = module.RaggedVerifyMode.COMPACT
    static = module.RaggedVerifyMode.STATIC

    def __init__(self, *a, **kw):
        real = module.read_ragged_verify_mode
        module.read_ragged_verify_mode = lambda: static      # the stock init refuses compact without the head
        try:
            stock_init(self, *a, **kw)
        finally:
            module.read_ragged_verify_mode = real
        if real() is compact:
            self._ragged_verify_mode = compact
        else:                               # diagnosis only: static verify at the full width, draft still split
            logger.warning("lookup wide without SGLANG_RAGGED_VERIFY_MODE=compact: every row verifies the full block")

    def schedule_layout(self, **kw):
        layout = ld._S.layout
        ld._S.layout = None
        if layout is not None:
            return layout
        return stock_layout(self, **kw)

    def compute_confidence_tensor(self, **kw):
        return None                        # the lengths come from the lookup, not from DSpark's confidence head

    cls.__init__ = __init__
    cls.schedule_layout = schedule_layout
    cls.compute_confidence_tensor = compute_confidence_tensor


def install_worker(module) -> None:
    """dspark_components.dspark_worker_v2: proposer, folded sampler and draft attention at DSpark's own block."""
    if not ACTIVE:
        return
    cls = module.DSparkWorkerV2
    stock_init, stock_attn = cls.__init__, cls.init_attention_backends

    def __init__(self, *a, **kw):
        stock_init(self, *a, **kw)
        ld._S.vocab = int(self.target_worker.model_runner.model_config.vocab_size)
        base = _base()
        if self.gamma <= base:                  # diagnosis only: compact verify at DSpark's own width, no extension
            logger.warning("lookup wide at gamma=%d <= %d: no extension, the draft is not split", self.gamma, base)
            ld._S.wide = {"gamma": self.gamma, "base": self.gamma, "epilogue": self._verify_epilogue,
                          "runner": self.model_runner, "budget": 0,
                          "max_bs": max(module.get_exec().graph.cuda_graph_config.decode.bs)}
            return
        query = base if self.sample_from_anchor else base + 1
        self._draft_block_spec_info = module.make_draft_block_spec_info(draft_token_num=query, device=self.device)
        self._proposer = module.DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=base,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            tp_sync=self._tp_sync,
            dp_moe_sync=self._draft_is_moe and module.get_parallel().enable_dp_attention,
        )
        ld._S.wide = {"gamma": self.gamma, "base": base, "epilogue": self._verify_epilogue,
                      "runner": self.model_runner, "budget": EXT_BUDGET,
                      "max_bs": max(module.get_exec().graph.cuda_graph_config.decode.bs),
                      "gate": SYNC_GATE, "hold": GATE_HOLD, "override": GATE_OVERRIDE, "split_grid": FUSED_C2}
        if self.ps.tp_rank == 0:
            logger.warning("lookup wide: DSpark drafts %d tokens, verify window %d, extension budget %d tokens/step; "
                           "sync gate %s (hold %d, override %s), narrow slots %s, fused ratio-2 compressor %s",
                           base, self.verify_num_draft_tokens, EXT_BUDGET, SYNC_GATE, GATE_HOLD, GATE_OVERRIDE,
                           NARROW_SLOTS, FUSED_C2)

    def init_attention_backends(self):
        stock_attn(self)
        width = _base() + 1
        runner = self.draft_model_runner
        backends = {id(b): b for b in [runner.attn_backend] + list(getattr(runner, "decode_attn_backend_group", []) or [])}
        for b in backends.values():
            for inner in [b] + [getattr(b, n) for n in ("full_attn_backend", "decode_backend", "prefill_backend")
                                if getattr(b, n, None) is not None]:
                if getattr(inner, "speculative_num_draft_tokens", None) is not None:
                    inner.speculative_num_draft_tokens = width
                    buf = getattr(inner, "extend_seq_lens_buffer", None)
                    if buf is not None:
                        buf.fill_(width)

    def _maybe_build_draft_sampler(self, *, available_memory_gb):
        return module.maybe_build_draft_sampler(
            draft_model=self.draft_model,
            gamma=_base(),
            max_bs=max(module.get_exec().graph.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self.ps.tp_rank,
            tp_sync=self._tp_sync,
            available_memory_gb=available_memory_gb,
            confidence_fn=None,
            out=None,                     # the epilogue's buffer is 15 wide; apply_wide copies the full block into it
        )

    cls.__init__ = __init__
    cls.init_attention_backends = init_attention_backends
    cls._maybe_build_draft_sampler = _maybe_build_draft_sampler


def _rewrite(module, fn, replacements):
    src = textwrap.dedent(inspect.getsource(fn))
    for old, new, count in replacements:
        if src.count(old) != count:
            raise RuntimeError(f"lookup wide: {fn.__qualname__} changed upstream ({old!r} found {src.count(old)}x)")
        src = src.replace(old, new)
    ns: dict = {}
    exec(compile(src, f"<lookup_wide {fn.__qualname__}>", "exec"), module.__dict__, ns)
    return ns[fn.__name__]


def install_graph_runner(module) -> None:
    """model_executor.runner.decode_cuda_graph_runner: compact verify graphs at 6b, 6b+10, 6b+20 tokens."""
    if not ACTIVE:
        return
    cls = module.DecodeCudaGraphRunner

    def _build_ragged_verify_token_buckets(self):
        narrow = _base() + 1
        limit = self.max_bs * self.captured_req_width            # rows of the shared logits buffer
        return sorted({b * narrow + e for b in self.capture_bs for e in EXT_TIERS if b * narrow + e <= limit})

    def _spark_capture_sizes(self):
        return self.capture_num_tokens if self.ragged_verify_mode else self.capture_bs

    def _spark_capture_num_tokens(self, size):
        return size if self.ragged_verify_mode else size * self.captured_req_width

    if os.environ.get("SPARK_LOOKUP_EAGER_VERIFY", "0") == "1":   # diagnosis only: target verify without graphs
        stock_can_run = cls.can_run_graph

        def can_run_graph(self, forward_batch):
            if forward_batch.forward_mode.is_target_verify() and not self.model_runner.is_draft_worker:
                return False
            return stock_can_run(self, forward_batch)

        cls.can_run_graph = can_run_graph
        logger.warning("lookup wide: target verify runs eagerly (SPARK_LOOKUP_EAGER_VERIFY=1, diagnosis)")
    if NARROW_SLOTS:
        stock_slots = cls._ragged_capture_slots
        narrow = _base() + 1

        def _ragged_capture_slots(self, num_tokens):
            if (not self.ragged_verify_mode or self.model_runner.is_draft_worker
                    or module.envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.get()):
                return stock_slots(self, num_tokens)
            return narrow_capture_slots(num_tokens, narrow, self.max_bs)

        cls._ragged_capture_slots = _ragged_capture_slots
    cls._build_ragged_verify_token_buckets = _build_ragged_verify_token_buckets
    cls._spark_capture_sizes = _spark_capture_sizes
    cls._spark_capture_num_tokens = _spark_capture_num_tokens
    cls.capture_one_shape = _rewrite(module, cls.capture_one_shape, [
        ("num_tokens = size * self.captured_req_width", "num_tokens = self._spark_capture_num_tokens(size)", 1)])
    cls._capture_one_stream = _rewrite(module, cls._capture_one_stream, [
        ("self.capture_bs", "self._spark_capture_sizes()", 2)])


def _ragged_rows(forward_batch, num_tokens: int):
    """(layout, row [T], starts [bs]) of a compact target-verify batch, or None. Fixed shapes (graph-capturable): the
    row of token t is found in the layout's qo_indptr; tokens past the covered total land on the last row."""
    import torch
    from sglang.srt.speculative.ragged_verify import resolve_ragged_verify_layout
    if not forward_batch.forward_mode.is_target_verify():
        return None
    layout = resolve_ragged_verify_layout(forward_batch)
    if layout is None:
        return None
    bs = int(forward_batch.req_pool_indices.shape[0])
    indptr = layout.qo_indptr_device.to(torch.int64)
    nb = min(int(indptr.numel()) - 1, bs)
    t = torch.arange(num_tokens, device=indptr.device)
    row = torch.searchsorted(indptr[1:nb + 1].contiguous(), t, right=True).clamp(max=nb - 1)
    return layout, row, indptr[:nb]


def install_dsv41_sparse(module) -> None:
    """layers.attention.dsv4.dsv41_sparse: token -> request map for ragged verify (stock repeats a uniform block)."""
    if not ACTIVE:
        return
    stock = module.token_req_indices

    def token_req_indices(forward_batch, *, num_tokens=None):
        if forward_batch.forward_mode.is_target_verify():
            n = num_tokens if num_tokens is not None else int(forward_batch.positions.shape[0])
            ragged = _ragged_rows(forward_batch, n)
            if ragged is not None:
                return forward_batch.req_pool_indices.to(ragged[1].dtype)[ragged[1]]
        return stock(forward_batch, num_tokens=num_tokens)

    module.token_req_indices = token_req_indices


def install_engram(module) -> None:
    """layers.engram: hash ids of a ragged verify batch through the extend path (per-token row + run starts), with no
    history commit (verify commits after acceptance, in commit_after_verify)."""
    if not ACTIVE:
        return
    cls = module.EngramHasher
    stock = cls.forward

    def forward(self, input_ids, forward_batch):
        num_tokens = input_ids.shape[0]
        ragged = _ragged_rows(forward_batch, num_tokens) if num_tokens else None
        if ragged is None:
            return stock(self, input_ids, forward_batch)
        _, row, starts = ragged
        req_slots = forward_batch.req_pool_indices.to(module.torch.int64)
        if input_ids.is_cuda and module.torch.version.cuda is not None:
            hash_ids, _ = module.engram_hash_ids(
                input_ids, forward_batch.positions, mode=module.MODE_EXTEND, history=self.history,
                token_map=self.token_map, multipliers=self.multipliers, primes=self.primes, offsets=self.offsets,
                pad_id=self.pad_id, num_real=num_tokens, req_slots=req_slots, block=1, row=row, starts=starts,
                image_token_id=self.image_token_id, mm_pad_shift=module.MM_PAD_SHIFT_VALUE)
        else:
            hash_ids, _ = self._torch_hash_ids(input_ids, forward_batch.positions, module.MODE_EXTEND,
                                               self.history[req_slots], num_tokens, 1, row, starts)
        return hash_ids

    cls.forward = forward


def narrow_capture_slots(num_tokens: int, narrow: int, max_bs: int) -> int:
    """Request slots of a compact verify graph when every live row verifies at least `narrow` tokens: a batch of bs
    rows totals >= narrow * bs tokens, so a graph of T tokens never serves more than T // narrow rows."""
    return max(1, min(num_tokens // narrow, max_bs))


def fused_c2_applies(*, forward_batch, rows: int, layer, is_dspark_draft: bool, narrow: int) -> bool:
    """True when a compact target-verify forward is uniform (every slot exactly `narrow` tokens, request-major,
    consecutive positions), so production's fused ratio-2 compressor with draft_len=narrow is exact for it. In a graph
    this is decided at capture from the capture layout: with narrow slots the 6b graphs capture [6]*b, and routing in
    lookup_draft.apply_wide sends only all-6 steps to them."""
    from sglang.srt.speculative.ragged_verify import resolve_ragged_verify_layout
    if not (forward_batch.forward_mode.is_target_verify() and not is_dspark_draft and layer.compress_ratio == 2
            and layer.compressor.use_fused_compress):
        return False
    if rows != forward_batch.batch_size * narrow:
        return False
    layout = resolve_ragged_verify_layout(forward_batch)
    lens = getattr(layout, "verify_lens_cpu", None) if layout is not None else None
    return lens is not None and len(lens) == forward_batch.batch_size and all(v == narrow for v in lens)


def install_backend(module) -> None:
    """layers.attention.deepseek_v4_backend: fused ratio-2 compressor on uniform compact verify graphs."""
    if not (ACTIVE and FUSED_C2):
        return
    cls = module.DeepseekV4AttnBackend
    stock = cls._low_ratio_compress
    narrow = _base() + 1

    def _low_ratio_compress(self, layer, x, req, pos, forward_batch):
        if (module.read_ragged_verify_mode() is module.RaggedVerifyMode.COMPACT
                and fused_c2_applies(forward_batch=forward_batch, rows=x.shape[0], layer=layer,
                                     is_dspark_draft=self.is_dspark_draft, narrow=narrow)):
            return self._low_ratio_compress_fused(layer, x, req, pos, draft_len=narrow)
        return stock(self, layer, x, req, pos, forward_batch)

    cls._low_ratio_compress = _low_ratio_compress
