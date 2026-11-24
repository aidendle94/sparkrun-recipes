"""Live-row all-reduce for the RoCE one-shot runtime (SPARK_ROCE_LIVE=1, with SPARK_ROCE_AR and SPARK_GATE).

The gate (patches/spark_gate.py) verifies K of a request's 8 rows, but the verify graph keeps all 8, so each of its 96
all-reduces (48 attention outputs, BF16; 48 expert combines, FP32) ships 8 rows a request: 74 KB and 147 KB for one
request, where K = 3 needs 28 and 55 KB. The time of a one-shot all-reduce grows with its bytes (13 us at 9 KB, 28 us
at 74 KB, 39 us at 147 KB inside a captured chain, bench/ar_bench.py).

This module is a fork of the b12x one-shot kernel (FlashInfer `flashinfer/experimental/b12x/comm/roce/_oneshot_cute.py`,
local-inference-lab/b12x, Apache-2.0; see NOTICE) that takes two more arguments: the address of a device table and the
16-byte packs a row holds. The table (int32: [0] = live rows, [2 + r] = 1 + the compact index of row r, 0 = not
live) is written by the gate at the start of a verify forward, inside the graph, from the same K the accept is capped
with. The kernel
  stages   only the live rows, packed in row order, into the send slot,
  rings    the doorbell with the live bytes (the proxy thread already posts a per-operation byte count),
  reduces  the live rows in fixed rank order from the peers' compact payloads (the same sums, the same bits as the
           stock kernel gives those rows) and copies this rank's own input to the rows that are not live.
Rows that are not live therefore hold each rank's own partial sum: finite, but not the same on every rank. They are the
rows the gate already calls garbage: their routed expert ids are their request's anchor row's, the accept is capped
below them, their KV positions are rewritten before they are read. Every rank must stage the same rows: K is the same
on every rank (the head's K is exchanged every step; the gate's own rule runs on all-gathered logits).

`Live.build(rt)` compiles the three launchers (BF16, FP16, FP32) before any graph capture; patches/roce_collectives.py
votes on the result so that no rank runs the stock kernel beside a rank that stages compact payloads.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import os

import torch

ON = os.environ.get("SPARK_ROCE_LIVE") == "1"
PACK_BYTES = 16
HEADER = 2                                # table words before the row entries
_log = logging.getLogger("spark_roce_live")
_ctx = {"table": None, "rows": 0, "tables": {}}


# ---- the table (written by the gate, read by the kernel) -----------------------------------------------------------
def table_for(rows, device):
    """The persistent table of a verify forward with `rows` rows (one per row count: a graph is captured per batch
    size), every row live until a step says otherwise."""
    t = _ctx["tables"].get(rows)
    if t is None:
        t = torch.zeros(HEADER + rows, dtype=torch.int32, device=device)
        t[0] = rows
        t[HEADER:] = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        _ctx["tables"][rows] = t
    return t


def fill(table, live):
    """live [rows] bool (at least one True) -> the table. Three small launches, inside the verify graph."""
    c = torch.cumsum(live.to(torch.int32), 0, dtype=torch.int32)
    table[HEADER:].copy_(torch.where(live, c, 0))
    table[0:1].copy_(c[-1:])


def begin(live):
    """Called by the gate at the start of a verify forward (eager warm-up, capture and so every replay)."""
    if not ON:
        return
    rows = live.shape[0]
    table = table_for(rows, live.device)
    fill(table, live)
    _ctx["table"], _ctx["rows"] = table, rows


def end():
    _ctx["table"], _ctx["rows"] = None, 0


def current(x):
    """The table for this tensor if it is a per-row tensor of the verify forward in progress, else None."""
    table = _ctx["table"]
    if table is None or x.dim() != 2 or x.shape[0] != _ctx["rows"]:
        return None
    if (x.shape[1] * x.element_size()) % PACK_BYTES != 0:
        return None
    return table


# ---- the kernel ------------------------------------------------------------------------------------------------------
def _make_launch_class():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint32
    from b12x.comm.roce import _oneshot_cute as stock
    from b12x.comm.roce._cute_intrinsics import (
        atomic_add_relaxed_gpu_u32, fence_sc_gpu, fence_sc_sys, ld_global_v4_u32, ld_relaxed_gpu_u32,
        ld_relaxed_sys_u32, ld_relaxed_sys_v4_u32, spin_until_eq_acquire_sys, st_global_v4_u32, st_release_gpu_u32,
        st_relaxed_sys_u32)

    class LiveLaunch(stock._RoceOneshotLaunch):
        """The stock launch with a row table: see the module's notes. Steps 2 (doorbell), 3 (wait) and 5 (epoch) are
        the stock kernel's, with the live byte count in the doorbell."""

        @cute.jit
        def __call__(self, input_ptr: cute.Pointer, output_ptr: cute.Pointer, size_packs: Int32, row_packs: Int32,
                     live_ptr: Int64, recv_base: Int64, flag_base: Int64, send_base: Int64, ctrl_base: Int64,
                     slot_bytes: Int64, epoch_ptr: Int64, stage_counter_ptr: Int64, tail_counter_ptr: Int64,
                     poison_ptr: Int64, spin_limit: Uint32, grid_x: Int32, stream: cuda.CUstream) -> None:
            self.kernel(input_ptr, output_ptr, size_packs, row_packs, live_ptr, recv_base, flag_base, send_base,
                        ctrl_base, slot_bytes, epoch_ptr, stage_counter_ptr, tail_counter_ptr, poison_ptr,
                        spin_limit).launch(grid=(grid_x, 1, 1), block=[self._threads, 1, 1], cluster=(1, 1, 1),
                                           stream=stream)

        @cute.kernel
        def kernel(self, input_ptr: cute.Pointer, output_ptr: cute.Pointer, size_packs: Int32, row_packs: Int32,
                   live_ptr: Int64, recv_base: Int64, flag_base: Int64, send_base: Int64, ctrl_base: Int64,
                   slot_bytes: Int64, epoch_ptr: Int64, stage_counter_ptr: Int64, tail_counter_ptr: Int64,
                   poison_ptr: Int64, spin_limit: Uint32) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            gdim, _, _ = cute.arch.grid_dim()
            input_base = Int64(input_ptr.toint())
            output_base = Int64(output_ptr.toint())
            epoch = ld_relaxed_gpu_u32(epoch_ptr)
            seq = epoch + Uint32(1)
            slot = Int64(seq & Uint32(1))
            send_slot = send_base + slot * slot_bytes
            entries = live_ptr + Int64(4 * HEADER)

            index = Int32(bidx) * Int32(self._threads) + Int32(tidx)
            stride = Int32(gdim) * Int32(self._threads)

            poisoned = ld_relaxed_gpu_u32(poison_ptr)
            if poisoned == Uint32(0):
                # 1. stage the live rows, packed, into the pinned send slot
                stage_index = index
                while stage_index < size_packs:
                    row = stage_index // row_packs
                    entry = ld_relaxed_gpu_u32(entries + Int64(row) * Int64(4))
                    if entry != Uint32(0):
                        compact = (Int32(entry) - Int32(1)) * row_packs + (stage_index - row * row_packs)
                        words = ld_global_v4_u32(input_base + Int64(stage_index) * Int64(PACK_BYTES))
                        st_global_v4_u32(send_slot + Int64(compact) * Int64(PACK_BYTES),
                                         words[0], words[1], words[2], words[3])
                    stage_index += stride
                cute.arch.sync_threads()

                # 2. the last block to finish staging rings the proxy doorbell with the live bytes
                if Int32(tidx) == Int32(0):
                    fence_sc_sys()
                    prior = atomic_add_relaxed_gpu_u32(stage_counter_ptr, Uint32(1))
                    if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                        live_bytes = ld_relaxed_gpu_u32(live_ptr) * Uint32(row_packs) * Uint32(PACK_BYTES)
                        st_relaxed_sys_u32(ctrl_base + Int64(4), live_bytes)
                        st_relaxed_sys_u32(ctrl_base + Int64(16) + slot * Int64(4), live_bytes)
                        fence_sc_sys()
                        st_relaxed_sys_u32(ctrl_base, seq)

                # 3. wait for every peer's payload-stripe flags
                if Int32(tidx) < Int32(self._world_size * self._hca_count):
                    peer = Int32(tidx) // Int32(self._hca_count)
                    hca = Int32(tidx) - peer * Int32(self._hca_count)
                    if peer != Int32(self._rank):
                        flag_addr = flag_base + ((Int64(peer) * Int64(self._slots) + slot) * Int64(self._hca_count)
                                                 + Int64(hca)) * Int64(self._flag_stride)
                        timed_out = spin_until_eq_acquire_sys(flag_addr, seq, spin_limit)
                        if timed_out != Uint32(0):
                            st_relaxed_sys_u32(ctrl_base + Int64(12), Uint32(peer))
                            st_relaxed_sys_u32(ctrl_base + Int64(24), Uint32(hca))
                            st_relaxed_sys_u32(ctrl_base + Int64(8), seq)
                            st_release_gpu_u32(poison_ptr, seq)
                cute.arch.sync_threads()
                failed = ld_relaxed_gpu_u32(poison_ptr)
                if failed == Uint32(0):
                    # 4. live rows: reduce in fixed rank order from the compact payloads; other rows: this rank's input
                    reduce_index = index
                    while reduce_index < size_packs:
                        row = reduce_index // row_packs
                        entry = ld_relaxed_gpu_u32(entries + Int64(row) * Int64(4))
                        offset = Int64(reduce_index) * Int64(PACK_BYTES)
                        if entry != Uint32(0):
                            compact = (Int32(entry) - Int32(1)) * row_packs + (reduce_index - row * row_packs)
                            compact_offset = Int64(compact) * Int64(PACK_BYTES)
                            accumulator = cute.make_rmem_tensor((self._pack_elems,), cutlass.Float32)
                            for source in cutlass.range_constexpr(self._world_size):
                                if cutlass.const_expr(source == self._rank):
                                    words = ld_global_v4_u32(input_base + offset)
                                else:
                                    peer_slot = recv_base + (Int64(source) * Int64(self._slots) + slot) * slot_bytes
                                    words = ld_relaxed_sys_v4_u32(peer_slot + compact_offset)
                                self._accumulate_words(accumulator, words, source == 0)
                            self._store_accumulator(output_base + offset, accumulator)
                        else:
                            own = ld_global_v4_u32(input_base + offset)
                            st_global_v4_u32(output_base + offset, own[0], own[1], own[2], own[3])
                        reduce_index += stride

                # 5. the last block to finish reduction publishes the next epoch
                fence_sc_gpu()
                cute.arch.sync_threads()
                if Int32(tidx) == Int32(0):
                    prior = atomic_add_relaxed_gpu_u32(tail_counter_ptr, Uint32(1))
                    if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                        fence_sc_gpu()
                        if ld_relaxed_sys_u32(ctrl_base + Int64(8)) == Uint32(0):
                            st_release_gpu_u32(epoch_ptr, seq)

    return LiveLaunch


def _compile(dtype_name, world, rank, threads, slots, flag_stride, hca_count):
    """One compiled launcher: run(input, output, size_packs, row_packs, live_ptr, ..., grid_x) on the current stream."""
    import cutlass
    import cutlass.cute as cute
    from b12x._lib.utils import current_cuda_stream, make_ptr

    launch = _make_launch_class()(dtype_name, world, rank, threads, slots, flag_stride, hca_count)
    dummy = lambda: make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16)  # noqa: E731
    raw = cute.compile(launch, dummy(), dummy(), 1, 1, 16, 16, 16, 16, 16, 4096, 16, 16, 16, 16, 1, 1,
                       current_cuda_stream())

    def run(input_address, output_address, size_packs, row_packs, live_address, recv_base, flag_base, send_base,
            ctrl_base, slot_bytes, epoch_address, stage_counter, tail_counter, poison_address, spin_limit, grid_x):
        raw(make_ptr(cutlass.Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            int(size_packs), int(row_packs), int(live_address), int(recv_base), int(flag_base), int(send_base),
            int(ctrl_base), int(slot_bytes), int(epoch_address), int(stage_counter), int(tail_counter),
            int(poison_address), int(spin_limit), int(grid_x), current_cuda_stream())

    return run


class Live:
    """The live-row launchers of one runtime (`rt`: b12x.comm.roce AllReduce)."""

    DTYPES = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}

    def __init__(self, rt):
        self.rt = rt
        self.calls = 0
        lay = rt._layout
        hcas = int(rt.rail_count)                         # the stock launcher's key: rails, not device names
        self.launchers = {dt: _compile(name, int(rt.world_size), int(rt.rank), int(rt._threads), int(lay.slots),
                                       int(lay.flag_stride), hcas) for dt, name in self.DTYPES.items()}

    @classmethod
    def build(cls, rt):
        """(Live, '') or (None, why not): never raises, every rank must reach the vote."""
        if not ON:
            return None, "SPARK_ROCE_LIVE is not 1"
        try:
            return cls(rt), ""
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    def all_reduce(self, x, table):
        """x [rows, width] (a per-row tensor of the verify forward) -> the sum on the table's live rows, x itself on
        the others. The stock launch path (roce_oneshot._run_prepared_all_reduce) with the forked kernel."""
        from b12x.comm.roce.roce_oneshot import _grid_blocks
        rt = self.rt
        with rt._lock:
            rt.check_health()
            nbytes = x.numel() * x.element_size()
            with torch.cuda.device(rt.device):
                capturing = torch.cuda.is_current_stream_capturing()
                out = torch.empty_like(x)
                src = x
                if x.data_ptr() % PACK_BYTES != 0:
                    src = rt._aligned_scratch(0, x)
                    src.copy_(x)
                dst = out if out.data_ptr() % PACK_BYTES == 0 else rt._aligned_scratch(1, out)
                grid = _grid_blocks(nbytes // PACK_BYTES, rt._threads, rt._blocks)
                stage_counter, tail_counter = rt._counter_addresses(grid)
                rt._order_stream(capturing)
                self.launchers[x.dtype](src.data_ptr(), dst.data_ptr(), nbytes // PACK_BYTES,
                                        (x.shape[1] * x.element_size()) // PACK_BYTES, table.data_ptr(),
                                        rt._recv_base, rt._flag_base, rt._send_base, rt._ctrl_base, rt._slot_bytes,
                                        rt._epoch_address, stage_counter, tail_counter, rt._poison_address,
                                        rt.spin_limit, grid)
                if dst is not out:
                    out.copy_(dst)
                rt._mark_stream(capturing)
        if not capturing:
            rt.check_health()
        self.calls += 1
        return out
