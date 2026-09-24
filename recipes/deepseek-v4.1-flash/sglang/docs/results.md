# Results

Measurements of this stack on the author's four DGX Sparks, taken on 2026-09-17, and the facts of the production
boots. Every number says what it is (one run, a range over runs, or a range over boots). Nothing here is a projection.

Units: tok/s throughout; GiB (2^30 bytes) for memory, which is how `free -g` and SGLang's memory log count;
GB (10^9 bytes) for files. "M tokens" is millions of KV-cache tokens.

## Configuration measured

| item | value |
|---|---|
| nodes | 4 × DGX Spark (GB10, 128 GB unified memory, arm64), one container per node |
| fabric | switched RoCE, two ConnectX-7 functions per node; NCCL over RoCE v2, buffers trimmed (1 MiB, no LL128, 8 channels) |
| image | this repository's Dockerfile on `lmsysorg/sglang:dev-dsv41` (the digest in [upstream.md](upstream.md)) |
| parallelism | TP4 / EP4, the launcher's `--tp 4 --ep-size 4 --nnodes 4` |
| launcher defaults | chunked prefill 2,048 tokens, DSpark block 5, memory fraction 0.80, 8 running requests, 524,288 context, KV pool auto-sized, MXFP8 backend `b12x`, RDMA collectives on (all-reduce ≤ 1 MB, all-gather shard ≤ 16 MB), Engram store with 64 threads and next-chunk prefetch on |
| prompts | `bench/needle.py` (unique random-word filler, salted per run, one needle at depth 0.5; prefill tok/s = prompt tokens / time to first token), `bench/prefill_repetitive.py` (one word repeated, unique tag, same needle), a three-regime single-stream decode probe (counting / code / prose; `bench/decode_bench.py` is its current form) and a prose-story request at 1–4 concurrent streams |
| sampling | greedy, thinking off per request, 400 completion tokens for the decode regimes |
| baseline | the author's previous vLLM deployment of the same checkpoint on the same nodes, same prompts, measured the same day; its prefill figures are ranges over its best boots |

The needle was found in every run listed here and the counting probe was exact.

## Prefill

| input | launcher defaults (chunk 2,048, persistent pool, prefetch) | chunk 1,024, per-call threads, no prefetch | previous vLLM deployment |
|---|---|---|---|
| unique random words, 32K, needle at 0.5 | **3,068** (10.7 s) | 2,245 (14.5 s) | 2,094–2,362 (14.2–15.5 s) |
| unique random words, 128K | **2,580** (51.1 s) | 2,077 (63.3 s) | 2,152 (60.5 s) |
| repetitive filler, 32K | **4,063** | 3,404 | 2,652 |
| repetitive filler, 128K | **3,435** | 2,868 | 2,550 |

One run per cell, one boot per column, both columns with RDMA collectives on. Seconds are time to first token.
The gain from the left column over the middle one is +37% at 32K and +24% at 128K on unique text; the Engram gather
sits on the critical path of every chunk, and the next section shows where the time went.

## Decode

| regime | launcher defaults | chunk 1,024, per-call threads | previous vLLM deployment |
|---|---|---|---|
| counting, one stream | 94.4 | 94.5 | 93.0 |
| code, one stream | 67.8 | 67.9 | 66.6 |
| prose, one stream | 32.4 | 32.6 | 31.3 |
| prose story, 1 / 2 / 3 / 4 streams, aggregate | 30.0 / 49.6 / 62.2 / 72.3 | 30.0 / 47.0 / 58.9 / 74.3 | 30.0 / – / – / 69.6 |

One run of the probe per cell. Decode is unchanged between the two configurations
within noise: a decode step gathers of the order of a hundred owned Engram rows per layer per rank, and after the
RDMA collectives went in the step is dominated by the model itself. Both configurations run DSpark block 5, like the
vLLM baseline.

## Engram gather cost

Per layer, rank 0, from the row store's own timer (cumulative gather nanoseconds, read every stats period).
Ranges are the spread seen over one boot each.

| phase | per-call thread spawn, chunk 1,024 | persistent pool + next-chunk prefetch, chunk 2,048 |
|---|---|---|
| prefill, per call (one chunk) | 10.9–15.9 ms | 7.7–9.9 ms |
| prefill, per prompt token | 10.7–15.5 µs | 3.8–4.8 µs |
| decode step, per call | 1.17–1.38 ms | 0.86–1.09 ms |

Per prompt token the gather became 2.5–3× cheaper: the pool removes 64 thread spawns per call, the prefetch has the
next chunk's rows in the page cache when the host-function node runs (it still copies them; only the page faults are
gone), and the larger chunk halves the number of calls.

## NCCL versus RDMA collectives

Same configuration otherwise (chunk 1,024, per-call threads, no prefetch), two consecutive boots.

| bench | NCCL | RDMA (b12x one-shot runtime) |
|---|---|---|
| decode counting / code / prose, one stream | 86.2 / 59.5 / 30.9 | 94.5 / 67.9 / 32.6 |
| prose story, 1 / 2 / 3 / 4 streams, aggregate | 29.9 / 43.2 / 51.7 / 66.0 | 30.0 / 47.0 / 58.9 / 74.3 |
| prefill 32K / 128K unique, time to first token | 14.0 s / 63.1 s | 14.5 s / 63.3 s |
| prefill 32K / 128K repetitive | 3,329 / 2,770 | 3,404 / 2,868 |

Single-stream decode +6–14%, four streams +13%, from moving the roughly eighty tensor-parallel all-reduces and the
logits all-gather of each step off NCCL. Prefill is unchanged by construction: its 64 MB all-reduces stay on NCCL.
The runtime's log line on this fabric reads `world=4 hcas=<two devices> gid_index=3 max_size=1048576`.

## Memory

| what | value |
|---|---|
| free for the KV pool after weights, per rank, memory fraction 0.80 | 7.2 GiB and 9.0 GiB on the two production boots (SGLang's `DSV4 memory calculation` line; its "GB" are GiB) |
| KV pool, auto-sized | 4.26 M and 5.43 M tokens on the two production boots (it depends on the page cache at boot); 4.4–5.8 M over the earlier boots |
| KV pool, previous vLLM deployment | 3.42 M tokens (pinned) |
| host memory floor during a 128K prefill, every node, chunk 2,048 | 11–14 GiB free (`free -g`, sampled every 15 s) |
| the same with chunk 1,024 | 16–20 GiB free |
| host memory at idle, production, per node | 19–21 GiB free |

The OOM guard on the author's nodes fires at 2.4 GiB free; larger chunks than 2,048 are untested.

## Two Engram designs, measured against each other

Both serve the rows from the same node-local copies; they differ in *when* the rows move. `SPARK_ENGRAM_MODE` selects
one (launcher knob `ENGRAM_MODE`).

| | host-node (default, `engram_store` + `engram_rows.c`) | staged (`engram_staged`, ported from the author's vLLM implementation) |
|---|---|---|
| mechanism | a host-function node inside the CUDA graph gathers each layer's rows through the C store while the GPU waits | a hook before every forward hashes the batch on the GPU, syncs once, gathers both layers' rows with a Python thread pool and stages them; the model reads them by position |
| prefill 32K / 128K, unique text | **3,068 / 2,580 tok/s** | 2,639 / 2,071 tok/s |
| prefill 32K / 128K, repetitive filler | 4,063 / 3,435 tok/s | 4,157 / 3,478 tok/s |
| decode counting / code / prose (single stream, `bench/decode_bench.py`, best of 2) | 101 / 82 / 35 tok/s | **110 / 89 / 38 tok/s** |
| dependencies | gcc at image build (the C store) | none beyond the base image |

Same nodes, same day (2026-09-21), one boot each. The staged design loses the CPU/GPU overlap on prefill: the sync
before each 2,048-token chunk waits for the previous chunk, then the gather and the launch of the next chunk's kernels
run with the GPU idle. On decode the graph replays cost little to launch and the pre-step staging is cheaper than two
in-graph host nodes. The host-node figures are from the production boot of the previous C pool (a thread per call);
the persistent pool cuts the in-graph stall from ~1.3 ms to under 0.1 ms per layer per step.

## production-1.1 validation boot (2026-09-21 08:40–08:53)

The image published as `production-1.1` (this tree: clean-room hooks, the persistent-pool C store at ABI 2, both Engram
modes) booted on the test port with the launcher defaults; production ran the previous image meanwhile.

| | production-1.1 | previous boots of the host-node design |
|---|---|---|
| boot to healthy | 563 s | 582–604 s |
| KV pool | 6,604,544 tokens (10.9 GiB free per rank after weights) | 4.2–5.4 M |
| count to 20 | exact | exact |
| prefill 32K unique text | **9.8 s · 3,308 tok/s** | 10.7 s · 3,068 |
| prefill 128K unique text | 59.3 s · 2,208 tok/s | 51.1 s · 2,580 |
| prefill 32K / 128K repetitive filler | 3,470 / 3,380 tok/s | 4,063 / 3,435 |
| decode counting / code / prose (`bench/decode_bench.py`) | **107 / 87 / 36 tok/s** | 101 / 82 / 35 (previous pool, same day) |
| Engram gather, per layer per call | prefill 7.2–13.4 ms; decode 0.74–0.79 ms | prefill 7.7–9.9 ms; decode 0.86–1.09 ms |
| host memory floor during the 128K prefill | 13–16 GB free per node | 11–14 GB |

Reading: decode gained 5–6 % from the pool (the in-graph stall per step is shorter). The 32K needle is the best on this
fleet, the 128K needle and the 32K filler are 14–15 % below the best previous boot while the 128K filler is level; the
cold-row gather of the fixed-stripe pool is slower in the first minute of a long prefill (13 ms per call against 10)
and the rest is boot-to-boot swing (GB10 clock state), which this stack shows at ±10–15 % on prefill. A dynamic
chunk queue in the pool is the next thing to try for cold rows.

## Concurrency (2026-09-22)

tonyd2wild's fixed-prompt bench (prompt set v1, eight categories, counting ceiling excluded), aggregate and mean
per-stream tok/s. The 8-request cap was the launcher default until this sweep; the KV pool allows far more.

| streams | cap 8, block 5 (production until 2026-09-22) | **cap 16, block 5 (default since)** | cap 16, block 3 | vLLM production-1.0 (2026-09-10) |
|---|---|---|---|---|
| 1 | 55.0 / 61.3 | | | 52.8 / 58.6 |
| 4 | 137.5 / 39.9 | 130.1 / 38.8 | 123.2 / 36.6 | 146.5 / 43.1 |
| 6 | 172.6 / 33.4 | | | 185.2 / 35.7 |
| 8 | | **207.7 / 30.2** | 206.2 / 29.5 | |
| 12 | | **272.6 / 27.9** | 262.3 / 25.8 | |
| 16 | | **329.4 / 25.3** | 309.1 / 22.9 | |

Decode is bound by reading the MoE weights, so a larger batch amortizes each read: 16 streams deliver 1.9x the
aggregate of the old ceiling at 25 tok/s per stream. Block 3 wins only on prose (16.2 vs 14.7 tok/s per stream at 16)
and loses everywhere else. Host memory held 12 GB free or more on every node at 16 streams.

## Outside the MoE: decode-step breakdown and early Engram staging (2026-09-23)

`overlay/step_timers.py` (SPARK_STEP_TIMERS=1) records capture-safe CUDA events around attention, MoE and Engram inside the
decode graphs. Per target-verify step, rank 0 (all ranks within 1 %):

| per step | 1 stream (6 tokens) | 8 streams (48) | 16 streams (96) |
|---|---|---|---|
| target graph | 54-57 ms | 140 ms | 157 ms |
| MoE | 32 | 90 | 102 |
| attention (43 layers, with the indexer) | 15 | 35 | 33 |
| Engram (2 layers) | 4.4-7.0 | 9.0 | 12.2 |
| rest of the graph | 3 | 6 | 10 |
| draft graph | 4.3 | 8.2 | 10.5 |

The Engram lookups stalled the GPU while the host gathered rows, although the ids of a step exist before layer 0 runs.
Early staging (`SPARK_ENGRAM_EARLY=1`, default since production-1.2) starts both layers' host gathers on a side stream as
soon as the hasher has produced the ids; the dequantize, the TP all-reduce and the projection stay at the layer, so no
collective runs on the side stream. Engram time per single-stream step fell to 1.8-3.9 ms. Verified with
`SPARK_ENGRAM_EARLY_VERIFY=1`: every staged gather on eager steps compared against the inline one, byte-identical on all
four ranks (0 differences), and 12/12 greedy answers identical to the previous production.

| fixed prompt set, aggregate / per stream (tok/s) | before (cap 16) | early staging (production-1.2) |
|---|---|---|
| 1 stream | 55.0 / 61.3 | **57.2 / 63.6** |
| 8 streams | 207.7 / 30.2 | **211.2 / 30.9** |
| 16 streams | 329.4 / 25.3 | **337.4 / 25.7** |
| single-stream decode, counting / code / prose | 105 / 88 / 35 | **110 / 94 / 37** |

The same window found that a request for prompt-token log-probabilities (e.g. `/v1/completions` with `echo` and
`logprobs`) raised inside the model under decoder bounded replay and stopped the whole server; `overlay/request_guard.py`
now answers such requests with HTTP 400 and the server keeps serving (tested live).

## MoE kernel: b12x versus the stock CUTLASS kernel (2026-09-23)

The b12x fused MoE gave the author's vLLM stack +9 % single-stream decode, so it was tried here. One MoE layer with
V4.1's dimensions (384 experts, top-6, hidden 5120), synthetic weights, head GPU, median of 40 runs, microseconds:

| tokens | same weights (96 experts x 2304): CUTLASS | b12x | per-rank work: CUTLASS, experts EP4 (production) | b12x, experts TP4 (576-wide slices) |
|---|---|---|---|---|
| 6 | 2,892 | 4,239 | **849** | 2,605 |
| 24 | 6,807 | 7,908 | **3,191** | 4,416 |
| 48 | 7,919 | 8,809 | **4,375** | 6,391 |
| 96 | 8,427 | 9,349 | **6,819** | 8,713 |
| 512 | 9,282 | 10,331 | **8,648** | 10,835 |
| 2,048 | 13,320 | 12,950 | **9,536** | 12,309 |

Outputs agreed to 1.3-2.7 % relative error (MXFP8 activation rounding). At decode sizes the stock kernel already reads
the expert weights at ~250 GB/s against the GB10's ~273 GB/s peak, so there is nothing for a different kernel to win,
and the expert-parallel layout the stock path uses beats b12x's tensor-parallel slices by 1.3-3x. The b12x route was
dropped; its fleet boots also showed that repacking 576-wide slices needs a padded copy that does not fit the
headroom. `tools/moe_microbench.py` in this repository reproduces the table.

## Attention: compressor overlap in verify (2026-09-23)

38 of the 43 attention layers (compression ratio 1 or 2) run a compressor and, for ratio 2, an indexer before the
sparse attention. SGLang can start them on a side stream in parallel with the Q/KV projections ("early sources"), but
in speculative verify only with FlashInfer's CuTe-DSL MXFP8 GEMMs, which do not exist for SM12x, because other backends
share one GEMM workspace. With the workspace made per stream and the gate lifted (the b12x dense GEMMs take no
workspace at all), outputs stayed identical (12/12 greedy, 32K needle passed), but nothing got faster:

| per verify step, rank 0 | 1 stream | 8 streams | 16 streams |
|---|---|---|---|
| attention, before / overlap | 14.7 / 14.8 ms | 35 / 34.0 ms | 33 / 32.3 ms |
| aggregate tok/s, production-1.2 / overlap | 57.2 / 57.6 | 211.2 / 203.1 | 337.4 / 333.0 |

The GEMMs on both streams read weights from the same memory, so running them together splits the bandwidth rather
than adding to it, and the extra side-stream work delayed Engram's staged gathers (Engram 3.9 -> 10.6 ms per
single-stream step). The overlap was not kept.

## Attention: where the time goes, and wo_a in MXFP8 (2026-09-23, not adopted)

`SPARK_STEP_TIMERS_ATTN=1` (with `SPARK_STEP_TIMERS=1`) splits the attention time of a verify step, rank 0 (all ranks
within 5 %):

| per verify step | 1 stream (6 tokens) | 8 streams (48) | 16 streams (96) |
|---|---|---|---|
| attention | 15.2 ms | 32.8 | 34.2 |
| Q/KV projections | 4.6 | 5.2 | 5.7 |
| compressor | 0.25 | 0.3 | 0.3 |
| indexer | 0.5 | 3.1 | 6.3 |
| attention kernel | 2.2 | 9.6 | 5.0 |
| output: inverse RoPE, wo_a, wo_b, TP all-reduce | 7.7 | 14.6 | 16.9 |

The output path was the largest piece. SGLang runs `wo_a` in FP8 only through DeepGEMM's `fp8_einsum` with 128x128
block scales; this checkpoint's are 32x32, so it dequantizes `wo_a` to BF16 at load, which doubles its bytes and
sends more than eight rows to a cuBLAS BF16 `bmm`. An experimental hook requantized it to MXFP8 at load (exact: the
checkpoint's scales are powers of two, and every one of the 40 layers verified bit for bit) and ran each head group on
the same dense MXFP8 GEMM as `wo_b`, releasing 0.62 GB of BF16 weight per rank. Output path: 7.7 -> 6.6 ms
(1 stream), 14.6 -> 12.5 (8), 16.9 -> 14.3 (16).

| aggregate / per stream (tok/s) | production-1.2 | wo_a MXFP8 |
|---|---|---|
| 1 stream | 57.2 / 63.6 | **58.8 / 65.9** |
| 8 streams | 211.2 / 30.9 | 210.2 / 30.7 |
| 16 streams | 337.4 / 25.7 | **339.8 / 26.4** |
| single-stream decode, counting / code / prose | 110 / 94 / 37 | **112.7 / 96.4 / 39.1** |

The activation into `wo_a` is now quantized per 1x32 segment, as for every other dense projection, so greedy answers
are no longer identical to BF16 `wo_a` (5/12 identical; the rest diverge at near-ties into equally fluent, correct
text, and the same 5/12 on two separate boots). The 32K needle passes, and speculative acceptance, which drops when
the target's distribution drifts from the one the draft was trained on, did not fall (mean accept length 3.56 against
3.49 on BF16 `wo_a`). The gain (about 3 % single-stream) was judged not worth giving up BF16 numerics in this
projection, so `wo_a` stays BF16 and the hook is not part of the overlay.

## Rank crashes under long prefills: the page cache (2026-09-24)

Twice (2026-09-22 01:18 and 2026-09-24 05:42) rank 1 died during a burst of long prefills with
`CUBLAS_STATUS_INTERNAL_ERROR` in the ratio-1/2 compressor's BF16 GEMM, and the other three ranks hung in the next
collective until the watchdog relaunched the fleet (13 minutes of downtime the second time). Seconds before each crash
the kernel logged `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`. The kernel logs showed the
same driver message dozens to hundreds of times on every node since 09-17, nearly all harmless: PyTorch's allocator
answers a failed allocation by emptying its cache and retrying, cuBLAS allocating for itself does not.

On a Spark the GPU allocates from the host's 128 GB, and the driver fails an allocation instead of evicting the page
cache; the kernel reclaims cache only when free memory reaches its watermarks (high = 1.5 GB). The cgroup limit was not
involved (the containers never reached it; GPU memory is not charged to them). Measured with `mincore`, two files filled
the free memory, both ours:

- the checkpoint, read once at boot: 15.5 GB of it still cached on rank 0, 16.6 GB on rank 1, 12 GB on rank 3;
- the Engram rows, read during every prefill and never released: 12-14 GB after twelve 48K-token prompts.

Free memory sat at 2-6 GB on a serving node. Two fixes, both in the overlay: `page_cache_release` drops the checkpoint
from the page cache right after the KV pool is allocated (MemFree on rank 0 at that point: 5.0 -> 24.7 GB; the pool
keeps the size the engine chose), and the row store keeps its cached rows under `SPARK_ENGRAM_CACHE_GB` (4 GB by
default): a thread measures them with `mincore` every 5 s and above the budget unmaps and drops the whole shard (the
kernel keeps file data in large folios and skips any folio reaching outside a range, so a range-limited drop left pages
behind in testing).

`bench/long_prefill_stress.py` replays the load (four clients, twelve unique 48K-token prompts, eight decode streams).
Driver out-of-memory events during the stress, summed over the four ranks, with MemFree at its start and end:

| configuration | driver OOM events | MemFree at start / end of the stress (GB, per rank) |
|---|---|---|
| no fix (production-1.2) | 13 | 3.9-13.4 / 3.5-4.9 |
| checkpoint drop only | 2 | 14.0-15.2 / 3.9-12.3 |
| checkpoint drop + Engram row budget (production-1.3), 20 prompts instead of 12 | **0** | 14.8-15.8 / 13.0-17.8 |

With only the checkpoint dropped the Engram rows still ate 9-11 GB of free memory during the stress. With the budget,
ranks 1-3 released their rows 10-15 times in the five minutes of the stress, 42-66 GB per rank in total, which is the
page cache the stress would otherwise have left behind; the peak stayed at 4.3-4.9 GB (the check runs every 5 s) and
MemFree never went below 13 GB. The prefills were not slower (15.2 s per 48K prompt, against 18.3 s without the fix
and 16.0 s with the checkpoint drop alone), and the per-lookup gather times during the bench matched the run without
the budget within 5-14 %. Decode and the fixed-prompt bench:

| | single stream, counting / code / prose | aggregate at 1 / 8 / 16 streams (tok/s) |
|---|---|---|
| production-1.2 (2026-09-23) | 110 / 94 / 37 | 57.2 / 211.2 / 337.4 |
| checkpoint drop only | 110.5 / 93.3 / 37.0 | 57.3 / 207.7 / 333.8 |
| production-1.3 | 108.2 / 93.3 / 37.0 | 57.2 / 199.0 / 328.3 |

The 8-stream figure of the same unchanged code has ranged from 199 to 211 tok/s across this week's windows, and the
budget was idle during the bench (no releases on ranks 1-3), so the 1.3 row is read as run-to-run spread; a cost of a
few percent at 8 streams cannot be excluded from one run.

## production-1.4: real attention heads on SM12x, and a YOCO-style prefill cut (2026-09-24)

**Decode.** Up to 64 query tokens (every verify step up to 10 streams) SGLang pads each rank's 16 attention heads to
64 before the sparse decode kernel, because datacenter FlashMLA only has 64- and 128-head builds. On SM12x the call goes
to FlashInfer's `decode_dsv4`, which is built for 8, 16, 32, 64 and 128 heads, so the pad quadrupled the work: a
48-token verify step spent 9.6 ms in the attention kernel, a 96-token one (already unpadded, on the prefill kernel)
5.0 ms. `SPARK_SM120_REAL_HEADS=1` keeps the real heads (`overlay/sm120_prefill_pages.py`). A first attempt that moved
SGLang's decode/prefill kernel switch instead crashed graph capture: FlashInfer's paged prefill kernel refuses 64 tokens
or fewer and routes them to `decode_dsv4`, which needs caller scratch the prefill wrapper does not pass.

| attention kernel per verify step (rank 0) | 1 stream (6 tokens) | 8 streams (48) | 16 streams (96) |
|---|---|---|---|
| padded to 64 heads | 2.16 ms | 9.58 | 5.03 |
| real heads | 1.61 | **3.39** | 4.17 |

**Prefill.** V4.1 builds its long-range KV only at the kv_source layers (2, 8, 14, 20); SGLang's decoder SWA bounded
replay runs the later 22 layers on each chunk's last 128 tokens only. Decode reads those layers' window KV for the
prompt's last 128 positions alone, so for every chunk that ends before them the tail pass is unread work: timed at 80 ms
of a 550-580 ms 2,048-token chunk (14 %). `SPARK_LATE_TAIL_SKIP=1` (`overlay/late_tail.py`) runs one token through the
late layers on such chunks and leaves every chunk that holds one of the last 128 positions exactly as SGLang runs it.
The late section fell to 32-38 ms per chunk (a one-token pass through 22 layers is launch- and collective-bound in
eager mode), the chunk to 523-529 ms.

| | production-1.3 | production-1.4 |
|---|---|---|
| 32K / 128K needle, time to first token | 9.1 / 52.2 s | 8.8 / 45.9 s |
| 12 unique 48K-token prompts with 8 decode streams (stress) | 174 s | 167 s |
| aggregate at 1 / 8 / 16 streams (tok/s) | 57.2 / 199.0 / 328.3 | 55.8 / **220.3** / 331.3 |

The single-stream figure is within the week's spread (54-59 tok/s); the real-heads kernel measured 20 % faster at one
stream in the timer run.

**Outputs.** Greedy answers are not identical to 1.3's: the 16-head kernel orders its floating-point work differently
from the 64-head one, and the prefill cut changes which prompt positions DSpark's draft sees, which changes the verify
batches. The divergent answers were checked by hand: the same facts in different words. All eight long prompts (4K-47K
tokens, including final chunks of 4 and 59 tokens) answered the fact buried in them correctly, both needles passed, and
the stress run had no failures and no driver out-of-memory event. A cross-boot comparison turned out to be a poor
identity test: a boot with only the prefill cut, which cannot touch single-chunk prompts, still changed one of twelve
short answers against the production boot.

## production-1.5: the late layers skipped outright; W8A16 wo_a measured and not adopted (2026-09-24)

**Prefill.** In 1.4 a chunk that ends before the prompt's last 128 tokens still ran the 22 late layers on one token, and
that pass cost 32-38 ms per chunk: each layer launches its kernels and joins its collectives whatever the token count.
`SPARK_LATE_TAIL_SKIP_ALL=1` skips them when every request of the chunk is such a chunk (the one tail row then carries
the last kv_source layer's state into the discarded logits and into DSpark's captured rows for that one position, which
the draft never attends to). The late section fell to 3.5-4.3 ms per chunk. Checks on the candidate: all eight long
prompts (4K-47K tokens) answered their buried fact correctly, both needles passed, the stress run had no failures and no
driver out-of-memory event.

**W8A16 `wo_a`.** Exact FP8 weights with BF16 activations and FP32 accumulation, in a Triton kernel (`overlay/wo_a_w8a16.py`):
the boot self-check matched the BF16 reference (relative error 0.0000) but timed it slower on every size, and the bench
agreed where it mattered:

| per layer, warm cache | W8A16 kernel | BF16 bmm |
|---|---|---|
| 6 tokens | 40 us | 21 us |
| 48 tokens | 58 us | 19 us |
| 96 tokens | 82 us | 39 us |
| 2,048 tokens | 1,466 us | 592 us |

| aggregate tok/s, 1 / 8 / 16 streams | production-1.4 | with W8A16 |
|---|---|---|
| fixed prompt set | 55.8 / 220.3 / 331.3 | 57.6 / 208.3 / 329.5 |

Halving the weight bytes does not pay for decoding them in the kernel on this part; the hook ships off.

Production-1.5 (W8A16 off), salted needles on the serving fleet: 32K in 8.0 s and 128K in 45.0 s to first token (1.4: 8.8 / 45.9 s; 1.3: 9.1 / 52.2 s).

## Bring-up

Seven boots of this stack, in order. The first three were fix-one-thing boots and no benchmark numbers were kept for
them; from boot 4 on the numbers are from the run logs.

| boot | configuration | what broke | what fixed it |
|---|---|---|---|
| 1 | first overlay, stock SM120 paths | model construction failed with "Only dense CPU tensors can be pinned": the pinned staging buffers were created under the CUDA default-device context the model is built in | the Engram hook allocates its host staging with an explicit CPU device; the zero-size parameters stay on the model device |
| 2 | + pinned staging on the CPU | CUDA-graph capture failed with a `schedule_meta` mismatch: V4.1's ratio-1/2 indexer sources call the DeepGEMM kernel, which needs a plan SGLang does not build on SM120 | `overlay/indexer_schedule.py` builds the plan for those two ratios |
| 3 | + indexer plan | the server's warm-up prefill was rejected by the SM12x sparse-MLA kernel: the ratio-2 KV source comes in 128-token pages (`extra_page_block_size=128`) | `overlay/sm120_prefill_pages.py` re-pages that source to 64-token pages. From this boot on, FlashInfer's `b12x` MXFP8 kernel logged one warning for the K = 576 projections and `overlay/mxfp8_kernel.py` kept those shapes on the stock kernel, as designed |
| 4 | chunk 2,048, NCCL, per-call gather threads | 32K prefill 2,774 tok/s (11.8 s); the 128K prefill took every node to zero free within a minute, the OOM guard killed two ranks and the other two hung in the TP all-reduce | `overlay/prefill_flush.py` returns the allocator cache after long chunks; chunk back to 1,024 for the next boot |
| 5 | chunk 1,024, NCCL, per-call threads; first working boot | nothing; 604 s to healthy, 32K prefill 2,333 (14.0 s), 128K 2,087 (63.1 s), decode 86.2 / 59.5 / 30.9, host floor 16–20 GiB. Decode below the vLLM baseline, the step dominated by NCCL all-reduces | `overlay/roce_collectives.py`: the TP group's small collectives as one-shot RDMA writes |
| 6 | + RDMA collectives | nothing; 584 s to healthy, decode 94.5 / 67.9 / 32.6. The now-instrumented Engram gather cost 11 ms per layer per 1,024-token chunk and 1.3 ms per decode step, spent largely on spawning 64 threads per call and on page faults | a persistent worker pool in `engram_rows.c` and `overlay/engram_prefetch.py`; chunk 2,048 again, now safe |
| 7 | + pool, prefetch, chunk 2,048 = the launcher defaults | nothing; 583 s to healthy, prefill 3,068 / 2,580, gather 3.8–4.8 µs per prompt token, host floor 11–14 GiB | shipped as production |

## Production boots

`launch/production.sh` on port 8210 with the published image, 2026-09-17 evening.

| fact | first boot | second boot |
|---|---|---|
| time to healthy | 582 s | 582 s |
| KV pool | 4,263,680 tokens | 5,426,688 tokens |
| thinking | off by default | on by default (`--default-chat-template-kwargs '{"thinking":true}'`); `"thinking": false` per request still overrides |
| checks | aliases listed on `/v1/models`, counting exact, an image described, a tool call with its round trip, 32K salted needle found in 9.9 s = 3,318 tok/s | the same checks |
| host memory at idle | 19–21 GiB free per node | 19–21 GiB free per node |

The KV pool differs between the two boots because SGLang sizes it from the memory it finds free at boot, which the
page cache affects; `KVTOK` (`--max-total-tokens`) pins it.
