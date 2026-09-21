# Upstream: the pinned base image and how to rebase the overlay

## What the image is (checked 2026-09-21)

| item | value |
|---|---|
| tag | `lmsysorg/sglang:dev-dsv41` |
| digest | `sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` |
| built | 2026-09-11 18:23 UTC, arm64, 33.2 GB |
| SGLang | `0.0.0.dev1+gda64c5cbb`, the `dsv4.1` branch at commit `da64c5cbb8cf6bfd39be19da43573fdfd484c43a`; source at `/sgl-workspace/sglang/python` |
| sglang-kernel | 0.4.6.post1 |
| FlashInfer | 0.6.18 (`flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` cu130) |
| DeepGEMM | `sgl-deep-gemm` 0.1.7 |
| PyTorch | 2.13.0+cu130 |
| NCCL | `nvidia-nccl-cu13` 2.30.7 (the wheel torch links); a system `libnccl.so.2.28.3` also sits in `/usr/lib/aarch64-linux-gnu` |
| origin | the build the SGLang team's DeepSeek-V4.1 integration pull request (#38798) tells users to run |

The tag is a moving tag. The Dockerfile's `BASE` pins the digest above, so a rebuild reproduces the same base; the
published overlay image `aidendle94/sparkrun-sglang-dsv41-gb10:production-1.1` is the Dockerfile built on it.

## What moved upstream after the image was cut (state on 2026-09-17)

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
   `docker pull lmsysorg/sglang:dev-dsv41 && docker image inspect lmsysorg/sglang:dev-dsv41 --format '{{index .RepoDigests 0}}'`.
   Put that `lmsysorg/sglang@sha256:...` value in the Dockerfile's `BASE` and in the table above, with the date.
   Never build from the bare tag: two nodes pulling on different days would run different engines in one fleet.

2. **Check the hook points in the new source** before building. Each overlay module assumes the following; read the
   file in the new image (`docker run --rm --entrypoint bash <base> -c 'sed -n 1,200p /sgl-workspace/sglang/python/sglang/...'`).

   | module | what the overlay relies on |
   |---|---|
   | `srt/layers/engram.py` | `EngramEmbedding.__init__(num_embeddings, dim, layer_id)`; `_lookup` calling `self._owned_rows(indices)` and all-reducing over TP; the attributes `dim`, `tp_size`, `row_start`, `rows`, `host_table`, `weight`, `scale` and the per-parameter `weight_loader`; the Triton `engram_gather(weight_ptr, scale_ptr, indices, out, dim, block_size, row_lo=, row_hi=)`; `EngramHasher.__init__` and `compute_engram_hash_ids` with the same argument order |
   | `srt/layers/quantization/fp8_utils.py` | `flashinfer_mxfp8_blockscaled_linear` and FlashInfer's `mm_mxfp8` still offering a `b12x` backend (the Dockerfile asserts the latter) |
   | `srt/layers/attention/dsv4/metadata.py` | `PagedIndexerMetadata` with `compress_ratio`, `force_deep_gemm_metadata` honoured by `__post_init__`, and the module flag `_IS_SM120` |
   | `kernels/ops/attention/flash_mla_sm120.py` | `_flash_mla_sm120_prefill(q, k_cache, indices, topk_length, attn_sink, head_dim_v, softmax_scale, extra_k_cache, extra_indices, extra_topk_length)`, `_PBS_SRC`, `_PBS_DST`, `_page_split_kernel`, `_page_mark_kernel` and the stride constants |
   | `srt/model_executor/model_runner.py` | `ModelRunner.forward(forward_batch, ...)`; `ForwardBatch.forward_mode.is_extend()` and host-side sequence lengths |
   | `srt/distributed/parallel_state.py` | `GroupCoordinator.__init__` receiving `group_name` (`"tp"` for the TP group), `all_reduce`, `all_gather(input_, dim, output_tensor_list)`, `graph_capture`, `destroy`, `cpu_group`, `device`, `rank_in_group`, `world_size` |
   | `srt/managers/scheduler.py` | `Scheduler.run_batch(batch)`, `self.chunked_prefill_size`; requests with `origin_input_ids`, `extend_range`, `prefix_indices` |
   | `srt/entrypoints/http_server.py` | `app`, `ORJSONResponse`, `ModelCard`, `ModelList`, `_global_state.tokenizer_manager` and the two `/v1/models` routes |

   Also check the launcher's flags still exist in `srt/server_args.py`: `--attention-backend dsv4`,
   `--moe-runner-backend flashinfer_mxfp4`, `--fp8-gemm-backend flashinfer_cutlass`, `--speculative-algorithm DSPARK`,
   `--speculative-dspark-block-size`, `--enable-decoder-swa-bounded-replay`, `--tool-call-parser deepseekv41`,
   `--reasoning-parser deepseek-v41`, `--default-chat-template-kwargs`, `--min-free-slots-delay`; and the environment
   variables `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`, `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE`, `SGLANG_DSV41_REASONING_EFFORT`.

3. **Build and run the CPU tests.** `docker build -t sglang-dsv41-spark:local .` compiles the C library and runs
   `tests/test_engram_rows.py` (numpy against the library on a synthetic sparse shard) and asserts the FlashInfer
   backend and the vendored b12x API version. Then run the hook test inside the built image, on the CPU, with a real
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

5. **b12x is pinned separately.** `vendor/` holds the wheel and the source snapshot at `b58f34ea`, the last revision
   of the `comm/roce` runtime with `API_VERSION == 1`, which `overlay/roce_collectives.py` requires; the Dockerfile
   asserts it. A newer runtime revision changes the API; bump it only together with the adapter.

6. **Publish** the rebuilt image under a new tag and update it everywhere it is written: `launch/production.sh`
   (`IMAGE` default), the README ("Run it" step 1 and the CPU check), the Dockerfile and launcher headers,
   `tests/test_hooks_cpu.py`'s docstring; then record the digest, date and versions in the table at the top of this page.
