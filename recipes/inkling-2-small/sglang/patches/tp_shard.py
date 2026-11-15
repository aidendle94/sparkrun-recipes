"""SPARK_TP_SHARD=1 | local (off by default; needs SPARK_FI_MOE=1): Inkling-2's NVFP4 expert layers sharded by width.

Stock places the 384 experts 96 a rank (expert parallelism): a decode step's routed experts fall unevenly on the
ranks, every rank waits in the all-reduce for the one that reads most, and the step pays the busiest rank's bytes.
Here every rank holds all 384 experts, each cut to a quarter of its intermediate width (w13: its rank's quarter of the
gate/up row pairs, w2: the matching columns; the same bytes of weights a rank as before), so every rank reads the
same bytes whatever the routing. patches/fi_moe.py runs the layer (shard=...).

Stock refuses NVFP4 experts under tensor parallelism because the intermediate activation is quantized with its row's
amax over the full width. "1" keeps that arithmetic: one small all-reduce a layer hands every rank each row's amax
over all shards (`amax_exchange`: each rank adds its values into its own row of a zeroed (world, pairs) FP32 buffer,
so the sum is exact, and the packing kernel takes the largest of the rows); the quantized operands are then those of
the unsharded path, and what differs is rounding: each rank's partial accumulator is rounded to BF16 where stock
rounds the whole one. "local" quantizes each shard with its own amax: no collective, a different draw of the
quantization noise (not the checkpoint's arithmetic). Never bit-identical to expert parallelism either way.

SPARK_TP_SHARD_HALF=1 (with "1"; off by default): the partial accumulators of the down projection leave the GEMM as
FP16 and are summed without the BF16 rounding (fi_moe's `half`): the shards' sum then carries less rounding than
stock's own BF16 row.

The MXFP8 layers (40-47) keep expert parallelism unless SPARK_TP_SHARD_FP8=1 (off by default): they then load the
same way and run stock's own Hopper code with every expert local (Marlin at a shard's width; the per-rank sums meet in
the same all-reduce). Their activations are quantized block by block (32 wide, no row amax), so every shard's
operands are the unsharded ones without an exchange; each rank's partial rows are rounded to BF16 by Marlin as the
whole rows are today. Stock sizes Marlin's row blocks from tokens / local experts, an estimate made for a rank that
holds 1/world of the experts: `route_blocks` hands it the size expert parallelism would pick (the rows an expert gets
are the same), which only matters above 48 tokens (prefill), and keeps its routing kernel the size of stock's.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import inspect
import logging
import os
import types

_log = logging.getLogger("spark_tp_shard")
MODE = os.environ.get("SPARK_TP_SHARD") or ""
HALF = MODE == "1" and os.environ.get("SPARK_TP_SHARD_HALF") == "1"
FP8 = bool(MODE) and os.environ.get("SPARK_TP_SHARD_FP8") == "1"
_S = {"probed": False, "layers": 0, "fp8": 0, "half": 0, "repacked": set()}


class _Config:
    """The model config with the intermediate width of one shard."""

    def __init__(self, config, intermediate):
        self._config, self.intermediate_size = config, intermediate

    def __getattr__(self, name):
        return getattr(self._config, name)


def construct(mod, stock, self, args, kwargs):
    """Inkling2Experts.__init__ for a layer whose w13 and w2 are NVFP4 (or, with SPARK_TP_SHARD_FP8=1, MXFP8) under
    expert parallelism: every expert, 1/world of its width. False (nothing done) for any other layer: the caller runs
    stock's constructor."""
    bound = inspect.signature(stock).bind(self, *args, **kwargs)
    config, prefix, quant = (bound.arguments[k] for k in ("config", "checkpoint_prefix", "quant_config"))
    parallel = mod.get_parallel()
    world, rank = parallel.moe_ep_size, parallel.moe_ep_rank
    names = [prefix + name for name in ("w13_weight", "w2_weight")]
    nvfp4 = quant is not None and all(name in quant.nvfp4_weights for name in names)
    fp8 = (FP8 and quant is not None and not any(name in quant.nvfp4_weights for name in names)
           and all(quant.quantizes_input(name) for name in names))
    if not (nvfp4 or fp8) or world == 1 or parallel.moe_tp_size != 1:
        return False
    if config.intermediate_size % (32 * world):
        raise ValueError(f"SPARK_TP_SHARD: intermediate width {config.intermediate_size} does not cut into {world} "
                         f"shards of whole 32-wide scale tiles")
    bound.arguments["config"] = _Config(config, config.intermediate_size // world)
    real = mod.get_parallel
    # Stock sizes the parameters from the parallel context and refuses NVFP4 under tensor parallelism: construct as
    # one rank holding every expert at the shard's width, then name the cut for _load_shard (the weight loader).
    mod.get_parallel = lambda: types.SimpleNamespace(moe_tp_rank=0, moe_tp_size=1, moe_ep_rank=0, moe_ep_size=1)
    try:
        stock(*bound.args, **bound.kwargs)
    finally:
        mod.get_parallel = real
    self.tp_size, self.tp_rank = world, rank
    if fp8:  # stock's Hopper path runs it (every expert local: ep_rank 0 of 1); fi_moe never sees the layer
        self._spark_shard_fp8 = (rank, world)
        route_blocks(mod.marlin_sm90, world, self.num_global_experts)
        _S["fp8"] += 1
        if _S["fp8"] == 1:
            _log.warning("[tp_shard] MXFP8 expert layers hold %d experts at width %d (shard %d of %d)",
                         self.num_global_experts, config.intermediate_size // world, rank, world)
        return True
    self._spark_shard = (rank, world)
    _S["layers"] += 1
    if _S["layers"] == 1:
        _log.warning("[tp_shard] mode %s: NVFP4 expert layers hold %d experts at width %d (shard %d of %d)", MODE,
                     self.num_global_experts, config.intermediate_size // world, rank, world)
    return True


def route_blocks(marlin, world, experts):
    """marlin_sm90.route picks its row block from min(tokens, rows) / local experts. With every expert local that is
    1/world of what expert parallelism computes for the same rows an expert: wrap it (once) so that a call over all
    `experts` gets expert parallelism's block size. Its routing kernel is a (blocks a program) x (experts, rounded up
    to a power of two) table: those calls also get 1/world of the blocks a program (64 for 256), so the kernel
    Triton builds stays the size of stock's (at 256 x 512 it is megabytes of PTX). Calls over fewer experts pass
    through."""
    stock = marlin.route
    if getattr(stock, "_spark_shard_blocks", False):
        return
    pow2 = lambda v: 1 << (v - 1).bit_length()  # noqa: E731
    shrink = max(1, pow2(experts) // pow2(max(1, experts // world)))

    def route(counts, offsets, rows, tokens, dst2src=None, block_size=None):
        if counts.numel() != experts:
            return stock(counts, offsets, rows, tokens, dst2src, block_size)
        if block_size is None:
            block_size = marlin._block_size(min(tokens * world, rows) / experts)
        keep = marlin.ROUTE_BLOCKS
        marlin.ROUTE_BLOCKS = max(16, keep // shrink)
        try:
            return stock(counts, offsets, rows, tokens, dst2src, block_size)
        finally:
            marlin.ROUTE_BLOCKS = keep

    route._spark_shard_blocks = True
    route.__wrapped__ = stock
    marlin.route = route


def fp8_repacked(self):
    """After stock's swizzle_nvfp4_scales on a sharded MXFP8 layer: say (once a process for each outcome) whether it
    left the layer with Marlin tensors, the only path the shards are meant to run."""
    marlin = getattr(self, "_marlin", None) or {}
    ok = set(marlin) == {"w13", "w2"}
    if ok in _S["repacked"]:
        return
    _S["repacked"].add(ok)
    if ok:
        _log.warning("[tp_shard] MXFP8 shard repacked for Marlin: w13 %s, w2 %s", tuple(marlin["w13"][0].shape),
                     tuple(marlin["w2"][0].shape))
    else:
        _log.warning("[tp_shard] MXFP8 shard not repacked for Marlin (has %s): not the path the shards are for",
                     sorted(marlin))


def half_scales(self, fi_moe):
    """(alpha, shift) for fi_moe's `half` on a sharded NVFP4 layer whose weights are loaded (called once at the end of
    the load, never under a graph capture), None when SPARK_TP_SHARD_HALF is off."""
    if not HALF or getattr(self, "_spark_shard", None) is None:
        return None
    import torch
    experts, _, half_width = self.w2_weight.shape
    shift = fi_moe.half_shift(2 * half_width)
    alpha = torch.full((experts,), 2.0 ** -shift, dtype=torch.float32, device=self.w2_weight.device)
    _S["half"] += 1
    if _S["half"] == 1:
        _log.warning("[tp_shard] half: down-projection partials as FP16 scaled by 2^-%d (width %d, %d experts)", shift,
                     2 * half_width, experts)
    return alpha, shift


def amax_exchange(group, rank, world, reduce=None):
    """parts (world, pairs) FP32 with this rank's row filled and the others zero -> every rank's row. The first eager
    call checks the collective on a known pattern (a sum of one value and zeros must come back exact)."""
    import torch
    reduce = reduce or group.all_reduce

    def exchange(parts):
        if not _S["probed"] and not torch.cuda.is_current_stream_capturing():
            _S["probed"] = True
            want = torch.arange(1, world + 1, dtype=torch.float32, device=parts.device) * 1.2345678
            probe = torch.zeros((world, 8), dtype=torch.float32, device=parts.device)
            probe[rank] = want[rank]
            got = group.all_reduce(probe)
            if not torch.equal(got, want[:, None].expand(world, 8)):
                raise RuntimeError(f"SPARK_TP_SHARD: the amax exchange is not exact on rank {rank}: "
                                   f"{got[:, 0].tolist()} != {want.tolist()}")
            _log.warning("[tp_shard] amax exchange exact on rank %d of %d", rank, world)
        return reduce(parts)

    return exchange
