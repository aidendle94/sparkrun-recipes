# Upstream: the pinned base image and how to rebase the overlay

## What the image is (production-1.8, checked 2026-10-18)

| item | value |
|---|---|
| tag | `lmsysorg/sglang:nightly-dev-cu13-20261018-b234b5c3` |
| digest | `sha256:bfef46e5a45a815f8c2f88a722980baf47b71a8689b89c6cc738f596ceeaf47c` |
| built | 2026-10-18 00:43 UTC, arm64, 36.3 GB |
| SGLang | `0.0.0.dev1+gb234b5c32`, `main` at commit `b234b5c32e0d65e04a61912f85fc09e95e74a328` (2026-10-17); source at `/sgl-workspace/sglang/python` |
| sglang-kernel | 0.4.10 |
| FlashInfer | 0.7.2, with b12x bundled (`flashinfer/experimental/b12x`, imported as `b12x`) |
| DeepGEMM | `sgl-deep-gemm` 0.2.2 |
| PyTorch / Triton | 2.14.1+cu130 / 3.8.0 |
| NCCL | `nvidia-nccl-cu13` 2.30.7 |

The Dockerfile's `BASE` pins the digest above; the published overlay image
`aidendle94/sparkrun-sglang-dsv41-gb10:production-1.8` is the Dockerfile built on it.

Until production-1.7.1 the base was `lmsysorg/sglang:dev-dsv41` (`sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860`,
built 2026-09-11, the `dsv4.1` branch at `da64c5cbb`, FlashInfer 0.6.18, PyTorch 2.13.0), with b12x 1.3.0 installed
from `vendor/`.

## What the rebase onto `main` changed (2026-10-18)

| where | upstream change | what the overlay does now |
|---|---|---|
| `EngramEmbedding.__init__` | gained `column_bounds` | accepted (unused: each rank owns whole rows) |
| `Req` | `extend_range` removed | `engram_prefetch` reads `extend_end`, and refuses to install if `Req.__init__` no longer sets it |
| `ForwardBatch` | `ngram_history` renamed `engram_history` | `engram_staged` reads the new name |
| `late_layer_tail_layout` | gained `whole_extend`; `forward_hc_pre_from_prev` removed | the tail layout passes unknown arguments and whole-extend batches to the stock function; `SKIP_ALL` refuses to install |
| V4.1 candidate indexer | the factory picks DeepGEMM's sparse-table backend from SM100 up, which SGLang enables on SM100 only | new hook `candidate_indexer.py`: dense-blocks backend on SM12x, decode on SGLang's paged scoring branch |
| b12x | moved into FlashInfer 0.7.2; the RoCE `AllReduce` needs a prepared plan, passed as `plan=` on every call | no wheel installed; `roce_collectives` publishes the plan once (`PreparationSession`) and votes it across ranks |
| `_kernel_num_heads` | decode attention on the rank's real heads is native | `sm120_prefill_pages.install_real_heads` removed |
| prompt-logprob requests under bounded replay | served upstream (#44850) | `request_guard.py` removed |
| — | — | `wo_a_w8a16.py` (never adopted) removed |

## What moved upstream after the 1.7.1 image was cut (state on 2026-09-17, kept for the record)


- The `dsv4.1` branch was reconstructed on top of `main` after the image was built. Its head (`00d7d516`,
  2026-09-17) shares no commits with the fifteen V4.1 commits in the image, requires sglang-kernel 0.4.7, and its
  `EngramEmbedding` no longer has the `_owned_rows` method this overlay hooks. The integration pull request calls the
  branch "being refactored and unstable".
- Things on that branch a Spark would want once it settles: a native 16-head attention path for small TP4 decode
  batches (#39674, reported +7–11% at batch 1 on 4×GB300), the top-k v2 cluster rework, and an indexer memory fix on
  `main` (#36534) that attacks the per-chunk transient this overlay's allocator flush works around.
- `main` carries the V4.1 kernels, top-k, communication and Engram modules (landed 2026-09-15 to 17) but not the
  model and runtime integration, nor the branch's Blackwell fast paths for small batches.

## How to rebase the overlay

1. **Pin the new base by digest.** Pull the tag on one node and read its digest:
   `docker pull lmsysorg/sglang:<nightly tag> && docker image inspect lmsysorg/sglang:<nightly tag> --format '{{index .RepoDigests 0}}'`.
   Put that `lmsysorg/sglang@sha256:...` value in the Dockerfile's `BASE` and in the table above, with the date.
   Never build from the bare tag: two nodes pulling on different days would run different engines in one fleet.

2. **Check the hook points in the new source** before building. Each overlay module assumes the following; read the
   file in the new image (`docker run --rm --entrypoint bash <base> -c 'sed -n 1,200p /sgl-workspace/sglang/python/sglang/...'`).

   | module | what the overlay relies on |
   |---|---|
   | `srt/layers/engram.py` | `EngramEmbedding.__init__(num_embeddings, dim, layer_id, column_bounds=None)`; `_lookup` calling `self._owned_rows(indices)` and all-reducing over TP; the attributes `dim`, `tp_size`, `row_start`, `rows`, `host_table`, `weight`, `scale` and the per-parameter `weight_loader`; the Triton `engram_gather(weight_ptr, scale_ptr, indices, out, dim, block_size, row_lo=, row_hi=)`; `EngramHasher.__init__` and `compute_engram_hash_ids` with the same argument order |
   | `srt/layers/quantization/fp8_utils.py` | `flashinfer_mxfp8_blockscaled_linear` and FlashInfer's `mm_mxfp8` still offering a `b12x` backend (the Dockerfile asserts the latter) |
   | `srt/layers/attention/dsv4/metadata.py` | `PagedIndexerMetadata` with `compress_ratio`, `force_deep_gemm_metadata` honoured by `__post_init__`, and the module flag `_IS_SM120` |
   | `kernels/ops/attention/flash_mla_sm120.py` | `_flash_mla_sm120_prefill(q, k_cache, indices, topk_length, attn_sink, head_dim_v, softmax_scale, extra_k_cache, extra_indices, extra_topk_length)`, `_PBS_SRC`, `_PBS_DST`, `_page_split_kernel`, `_page_mark_kernel` and the stride constants |
   | `srt/model_executor/model_runner.py` | `ModelRunner.forward(forward_batch, ...)`; `ForwardBatch.forward_mode.is_extend()` and host-side sequence lengths |
   | `srt/distributed/parallel_state.py` | `GroupCoordinator.__init__` receiving `group_name` (`"tp"` for the TP group), `all_reduce`, `all_gather(input_, dim, output_tensor_list)`, `graph_capture`, `destroy`, `cpu_group`, `device`, `rank_in_group`, `world_size` |
   | `srt/managers/scheduler.py` | `Scheduler.run_batch(batch)`, `self.chunked_prefill_size`; requests with `origin_input_ids`, `extend_end`, `prefix_indices` |
   | `srt/entrypoints/http_server.py` | `app`, `ORJSONResponse`, `ModelCard`, `ModelList`, `_global_state.tokenizer_manager` and the two `/v1/models` routes |

   | `srt/layers/attention/deepseek_v4_backend.py` and `dsv4/v41_indexer` | `make_candidate_indexer(token_to_kv_pool=, req_to_token=, page_size=, candidate_topk_blocks=, candidate_block_size=)`, `DenseBlocksBackend`, `dense_blocks.decode_scores`, `scoring.fp4_index_logits_paged`, `scoring.plan_topk_v2`, `scoring.PagedDecodeScores` |

   Also check the launcher's flags still exist in `srt/server_args.py`: `--attention-backend dsv4`,
   `--moe-runner-backend flashinfer_mxfp4`, `--fp8-gemm-backend flashinfer_cutlass`, `--speculative-algorithm DSPARK`,
   `--speculative-dspark-block-size`, `--enable-decoder-swa-bounded-replay`, `--tool-call-parser deepseekv41`,
   `--reasoning-parser deepseek-v41`, `--default-chat-template-kwargs`, `--min-free-slots-delay`; and the environment
   variables `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`, `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE`, `SGLANG_DSV41_REASONING_EFFORT`.

3. **Build and run the CPU tests.** `docker build -t sglang-dsv41-spark:local .` compiles the C library and runs
   `tests/test_engram_rows.py` (numpy against the library on a synthetic sparse shard) and asserts the FlashInfer
   backend and the bundled b12x RoCE API version. Then run the hook test inside the built image, on the CPU, with a real
   Engram directory mounted (only the manifest and the shard headers are read):

   ```bash
   docker run --rm -v ~/dsv41-engram-local:/engram-local:ro -e SPARK_ENGRAM_DIR=/engram-local \
     -e SPARK_SERVED_ALIASES=alias-check --entrypoint python3 sglang-dsv41-spark:local /opt/dsv41-spark/tests/test_hooks_cpu.py
   ```

   It must end with `HOOKS CPU TEST PASS`: the Engram constructor and gather bound to `engram_store`, the MXFP8
   wrapper to `mxfp8_kernel`, the row ranges inside the table, the indexer metadata wrapped only on SM120, the
   prefill flush on `ModelRunner.forward`, the `/v1/models` routes rebuilt, and every module in the hook table
   imported without a hook error (a module a CPU-only box cannot import for another reason is reported and skipped).
   A hook whose target changed shape fails here, not on the fleet.

4. **Boot once in a test window, not in production.** `launch/window-sgl.sh` stops production, launches the new
   image on a separate port with a deadline, runs the count smoke and the 32K needle (`BENCH=1` adds the 128K needle,
   the repetitive prefill and the decode bench), collects every rank's log and restores production. Check the log
   lines the README lists (Engram layer lines, `RoCE collectives on`, the single MXFP8 shape warning, `DSV4 memory
   calculation`) and compare the numbers with [results.md](results.md), taken on the same boot within the same
   minutes, since GB10 has a slow clock state.

5. **b12x comes with FlashInfer.** Since production-1.8 the RoCE runtime is the copy bundled in the base's FlashInfer;
   the Dockerfile asserts `roce.API_VERSION == 1` and compiles its proxy. Never pip-install the standalone b12x over it
   (both provide the top-level `b12x` package). A FlashInfer bump that changes the runtime's API needs the adapter in
   `overlay/roce_collectives.py` changed with it. (`vendor/` keeps the wheel and snapshot production-1.7.1 used.)

6. **Publish** the rebuilt image under a new tag and update it everywhere it is written: `launch/production.sh`
   (`IMAGE` default), the README ("Run it" step 1 and the CPU check), the Dockerfile and launcher headers,
   `tests/test_hooks_cpu.py`'s docstring; then record the digest, date and versions in the table at the top of this page.
