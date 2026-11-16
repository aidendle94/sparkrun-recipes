# Inkling-2-Small on four DGX Sparks with SGLang (experimental)

This recipe serves [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(594B parameters, 17B active per token, the official NVFP4 checkpoint, 394 GB) on four NVIDIA DGX Spark nodes behind
one OpenAI-compatible endpoint. The engine is the SGLang team's own image for the model,
`lmsysorg/sglang:dev-inkling-2-small`, unmodified. That image targets Hopper and datacenter Blackwell; on a Spark's
GB10 (SM12x) it does not serve the model as it is. The MIT-licensed Python hooks in `patches/` are copied to every
node and put on `PYTHONPATH`; they supply what GB10 needs (attention arguments, expert kernels, collectives across
four boxes, a correct decode kernel at page size 1) and then the speed: about 61 tok/s for one stream, where the
image's own decode step on the same hardware gave 37.

**Status: experimental.** It has served the author's fleet since 2026-11-12, which is four days at the time of
writing, and most of those days were spent changing it. What that means in practice:

- The base is a development image, not an SGLang release. The hooks bind to one build of it (digest below). One of
  them rebuilds a kernel from the image's own source and refuses to boot if that source has changed.
- The numbers are single boots on one fleet. The defaults changed several times a day; the last change (sharded
  experts) became the default on 2026-11-15 and has hours of serving behind it, not weeks. The larger pool (half the
  windowed share, FP8 draft pools) has served since 2026-11-16 and is not a launcher default.
- There are open problems, listed under [Known problems](#known-problems). One of them is a scheduler hang that
  happened once and was never reproduced.

If you want a recipe that has soaked, use [`deepseek-v4.1-flash/sglang`](../../deepseek-v4.1-flash/sglang/).

Glossary. **TP4/EP4**: tensor parallel over the four nodes, with the experts split four ways. **DSpark**: the
speculative decoding built into the engine: a small draft model proposes up to 7 tokens, the target verifies them in
one forward. **NVFP4 / MXFP8**: 4- and 8-bit floating-point weight formats with shared block scales; 40 of the
model's expert layers are NVFP4, 8 are MXFP8. **RoCE**: RDMA over Converged Ethernet, between the Sparks' ConnectX-7
ports. **Pool**: the KV cache, counted in tokens, shared by all running and cached requests.

## What you get

Measured on four Sparks over a switched RoCE fabric, 2026-11-13 to 2026-11-16, with the launcher's defaults unless
the row says otherwise. Every row is one boot; conditions and the history of each number are in
[docs/results.md](docs/results.md).

| what | measured |
|---|---|
| decode, one stream, greedy, coding questions of 400 tokens | 60–61 tok/s (37 with the image's own decode step, on another set of such questions) |
| decode, one stream, temperature 0.7 | 56.6 tok/s |
| decode, four / eight streams at a time, greedy | 106 / 130 tok/s for the batch (about 26 / 16 a stream) |
| prefill, ~15K-token prompts | 2,400–2,500 tok/s |
| cold prefill of 130K / 260K / 500K tokens | 61 / 142 / 348 s |
| decode behind a long context (130K–500K) | 52–55 tok/s on the 600K configuration; 68 at 130K and 60 at 500K on the 793K one |
| context length | 262,144 by default; 524,288 run through the environment (see [Run it](#run-it)) |
| KV pool | 493K–805K tokens, sized at each boot; 545K–600K with the 600K cap of the 500K configuration; 793,000 with the settings the author's fleet serves with since 2026-11-16 (see [Run it](#run-it)) |
| concurrent requests | 8 |
| tool calls, 283 cases over five wire formats (chat, Anthropic messages, each also streamed, and responses), two trials | 250 right, 20 soft, 13 failed (10 are `tool_choice: none`, 3 slips on the sampled trial); 12 of 12 agent loops |
| boot to healthy | about 5 min with a warm kernel cache, about 17 min the first time |
| request features | tool calls, reasoning with effort levels; OpenAI chat completions and responses, Anthropic `/v1/messages`; extra model names on `/v1/models` and the Ollama-style `/api/tags` and `/api/show` |

Decode tok/s is tokens divided by decode seconds from the server's own timing. For comparison, the hosted model on
OpenRouter answered the same long lookups at 190–320 tok/s depending on the provider; this is a 128 GB desktop
cluster, and its decode step is set by memory bandwidth.

## What is changed against the stock image

Nothing in the image is modified; every change is a hook in `patches/` behind an environment switch. The ones that
make it work on GB10 at all:

- **Attention**: SGLang's SM12x wrapper dropped two arguments Inkling passes; and at page size 1 (what GB10 runs)
  the image's decode kernel read keys from other requests' cache slots as soon as two requests shared a batch. The
  result was loops and cross-talk that throughput numbers do not show. `patches/paged_decode.py` looks the page up
  per key. Read [docs/results.md](docs/results.md#garbage-under-concurrency-found-and-fixed-2026-11-13) before
  trusting any multi-stream number from a GB10 setup without it.
- **Experts**: the image's NVFP4 kernels are SM100-only. `patches/fi_moe.py` runs the 40 NVFP4 expert layers on
  FlashInfer's SM120 grouped GEMM with the checkpoint's weights (prefill from about 440 to about 2,000 tok/s).
- **Collectives**: the model's all-reduces assume NVLink; here they go over RoCE (`patches/roce_collectives.py`).

The ones that make it faster or larger, all on by default, each with its own off switch:

| launcher variable (default) | what it does | cost |
|---|---|---|
| `KV_DTYPE=fp8_e4m3` | FP8 KV cache with a reader for SM12x: 1.9x the pool at the same speed | prefill NLL within 0.02 of BF16 |
| `GATE=0.4` | verifies only the draft rows the draft is confident of: +15 % decode | greedy text moves at near-ties only, by less than a second request in the batch moves it |
| `DENSE_I8=draft,dhead,dfree` | the draft reads int8 copies of its own weights: +5 % | none for the target (held-out NLL identical); about 5 % of the pool |
| `DENSE_P12=attn,head,free` | the target's attention weights packed into 12 bits without rounding: −4.5 % step time for one stream | none: exact |
| `TP_SHARD=half,fp8` | every node holds all 384 experts at a quarter of their width instead of 96 whole ones, so no node waits for the busiest: −18 % step time | not bit-identical to the stock placement (rounding only; mean NLL unchanged) |
| `REASONING_CACHE=1` | the server puts an answer's reasoning back into history turns that arrive without it | prompts grow by that reasoning |

Each is switched off by setting it empty on the relaunch (`TP_SHARD= DENSE_P12=` is the stack as it served before
2026-11-15); `KV_DTYPE=auto` is the BF16 pool.

Off unless set, and what the author's fleet has served with since 2026-11-16. They grow the pool in the same
bytes, and none of them touches the target's arithmetic:

| setting | what it does | cost |
|---|---|---|
| `--swa-full-tokens-ratio 0.05` (in `EXTRA_ARGS`) | halves the pool of the 40 windowed layers, which was the slack: a pool token costs 6,349 bytes a node instead of 7,578 | about 35 cached conversation ends keep their window instead of 53–58; one that lost it is prefilled again when it comes back |
| `SPARK_WMARK_BOOST=1` (in `NCCL_EXTRA`) | sizes the pool from the memory a node has without the kernel's temporary watermark boost, so that every boot gets the same pool (`patches/wmark_boost.py`) | none |
| `SPARK_DRAFT_KV_FP8=1` (in `NCCL_EXTRA`) | the draft's KV pools as FP8 with a reader of their own (`patches/draft_decode.py`): 5,734 bytes a token, 793,000 tokens where 716,000 fitted | about 1 % of step time for one short stream, nothing at 4 and 8 streams; accepted tokens a step unchanged; the draft is faster behind a long context |

`NCCL_EXTRA` is the launcher's pass-through for extra `-e NAME=value` arguments of the containers. The numbers are
in [docs/results.md](docs/results.md#kv-cache-what-a-token-costs-and-what-grows-the-pool-2026-11-1516).

## What you need

- Four DGX Spark nodes (GB10, 128 GB unified memory, arm64). One of them is rank 0: you run every script there.
- A RoCE fabric between the nodes, switched or direct, with the RDMA devices visible in `/sys/class/infiniband` on
  every node.
- The checkpoint on the local disk of **every** node, 394 GB each (the launcher mounts each node's own copy; the
  default place is the HuggingFace cache of the launching user, `INKLING_REPO_REL` in `launch/fleet.env` moves it).
  A download on a node that is serving something else should run under a memory limit (for example
  `systemd-run --user -p MemoryMax=4G hf download ...`): on GB10 the page cache competes with the GPU for memory.
- About 37 GB per node for the image.
- Docker with the NVIDIA container runtime; the launching user in the `docker` group on every node; key-based ssh
  from rank 0 to the other three; permission to pass `/dev/infiniband` into a container and to lift the memlock
  limit.
- On rank 0: `bash`, `ssh`, `scp`, `curl`, `flock`, `nvidia-smi`, and Python 3 (standard library only) for `bench/`.
- Nothing else on the GPUs. The weights take 92–93 GB of each node and the pool takes what is left.

## Run it

```bash
# 0. The checkpoint, on every node (the revision this recipe was measured on).
hf download thinkingmachines/Inkling-2-Small --revision b0e7afb8892a07897c78ce1c5b801a0d9cbd5e7b

# 1. The image, on every node (arm64). The hooks were written against this build of the tag:
#    lmsysorg/sglang@sha256:424c1da97f6d6eef1612006c77b2ca791657a8184c1fbdbc682ebf9485d3b32c  (2026-11-12)
#    If the tag has moved since and a boot fails in a hook, pull that digest and set INKLING_IMAGE to it.
docker pull lmsysorg/sglang:dev-inkling-2-small

# 2. The site file: nodes, users, devices. Every line of the example says how to find its value.
cp launch/fleet.env.example launch/fleet.env

# 3. Print the four docker run commands without starting anything. This already needs ssh to every rank:
#    the RoCE-v2 GID index is probed on each node.
launch/launch-inkling2.sh --dry-run

# 4. Launch: disarms the watchdog, copies patches/ to every node, removes a previous inkling2 container, starts
#    ranks 3..0, waits for /health, arms the watchdog.
launch/relaunch-inkling2.sh

# 5. First request (port 8210 on rank 0).
curl -s http://<rank-0>:8210/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"inkling-2-small","messages":[{"role":"user","content":"Is 91 prime?"}],"max_tokens":400}'

# 6. Read the text of overlapping requests. Do this after every boot of a changed configuration: a broken
#    decode path looks fast and healthy in every counter.
mkdir -p logs && python3 bench/concurrency_check.py && python3 bench/reasoning_cache_check.py --expect on
```

The 500K configuration the author's fleet serves since 2026-11-16 (not the launcher's defaults; the three pool
settings are in the table above, and `--prefill-decode-interval 2` is a scheduling setting still under test):

```bash
CTX=524288 MEMFRAC=0.94 NCCL_EXTRA="-e SPARK_WMARK_BOOST=1 -e SPARK_DRAFT_KV_FP8=1" \
  EXTRA_ARGS="--max-total-tokens 793000 --swa-full-tokens-ratio 0.05 --prefill-decode-interval 2" \
  launch/relaunch-inkling2.sh
```

The cap goes with the settings: the same bytes hold 716,000 tokens without `SPARK_DRAFT_KV_FP8=1` and 600,000
without the ratio either (the configuration most rows of the table above were measured on:
`CTX=524288 MEMFRAC=0.94 EXTRA_ARGS="--max-total-tokens 600000 --prefill-decode-interval 2"`). A cap above what
the bytes hold is sized down by the engine. The bytes are what the rank-0 node has free at boot, so the cap that
fits is a property of the site.

Stop: kill the watchdog (the pid is in the last `watchdog started` line of `~/inkling2-watchdog.log`), then
`launch/launch-inkling2.sh --stop`. The watchdog probes `/health` and a three-token chat completion every 60 s and
relaunches the fleet after three failures with the environment it was started in; before it does, it saves the
state, logs and Python stacks of every rank under `$LOG_DIR/postmortem/`.

The other launcher variables (`PORT`, `MAXREQ`, `CHUNK`, `SERVED`, `SERVED_ALIASES`, `MTP`, `ROCE_AR`, `FI_MOE`,
`PAGED_DECODE`) are documented at the top of `launch/launch-inkling2.sh`. Leave `PAGED_DECODE=1`.

## Using it as a client

- **Temperature.** A chat request without a `temperature` is sampled at 0.7 (the server's default; the model ships
  no generation config). For lookups in long contexts and for agent work, send 0.
- **History.** The model's template shows an earlier turn's thinking only if the client sends it back
  (`reasoning_content`, or the thinking blocks on `/v1/messages`). Without it, over a long context, the model
  answers a later question with an earlier answer. The reasoning cache covers clients that drop it, for chat
  completions and `/v1/messages`; the Responses API is not covered, and a restart forgets the cache.
- **Output limit.** Give long-context turns 4,000 output tokens or more: the reasoning uses them. Treat
  `finish_reason: length` with an empty answer as no answer.
- **Reasoning effort** is a number in the prompt, not a budget: `none`, `minimal`, `low`, `medium`, `high` (the
  default), `xhigh`, `max`, or a float on chat completions. `thinking.budget_tokens` is accepted and not enforced;
  `max_tokens` is the only hard cap. The task sets the amount of thinking more than the level does.
- **`tool_choice: none`** does not work: the model writes the call as text. Leave tools out of the request instead.

## Known problems

- **A silent scheduler hang, once** (2026-11-13, under concurrent agent loops): the server stopped answering with
  every container up. It was not reproduced. The watchdog recovers from it and now captures the stacks.
- **A short request waits behind a long prefill** for the whole prefill (338 s behind a 500K one).
- **The reasoning loop on long lists.** At 130K tokens of near-identical record lines, about 1 lookup in 10 runs
  out of tokens copying the list in its reasoning (0 at 60K, 0 on source code). It is the model: the hosted
  endpoints do the same on the same questions (29 of 200 across four providers against 4 of 40 here). No server
  setting fixes it; a search tool is the right way to look things up in such lists.
- **The pool differs from boot to boot** (493K–805K) because it is sized from what the head node has free at that
  moment, and the kernel's watermark boost moves that by gigabytes. `SPARK_WMARK_BOOST=1` corrects the sizing for it (off
  unless set).
- **Log-probability requests over long spans can take a node down** (the full-vocabulary logits of every scored
  position). Keep scored spans to a few hundred tokens.
- **No GPU work beside the serving fleet**, however small: there is no memory for it, and the driver fails
  allocations instead of evicting page cache.
- **Heat.** Long all-node load reached 88–89 °C on the hottest node.

## Files

| file | what |
|---|---|
| `launch/launch-inkling2.sh` | one container per node (`inkling2`), RoCE/NCCL environment, the hooks on PYTHONPATH; `--dry-run`, `--stop` |
| `launch/relaunch-inkling2.sh` | disarm the watchdog, launch, wait for /health, re-arm the watchdog (use this to restart) |
| `launch/watchdog-inkling2.sh` | probes and relaunches (log `~/inkling2-watchdog.log`) |
| `launch/fleet.env.example` | the site file; copy to `launch/fleet.env` (git-ignored) |
| `patches/sitecustomize.py` | installs the hooks at interpreter start in every rank; the other files in `patches/` are the hooks |
| `bench/concurrency_check.py` | reads the text of overlapping requests against single-stream references. **Run it after any change to a decode path** |
| `bench/reasoning_cache_check.py` | whether the server restores a history turn's reasoning, over six wire formats |
| `bench/toolcall_check.py` | the tool-call suite (wire formats and agent loops) |
| `bench/streams.py`, `bench/step_time.py` | decode tok/s against the number of streams; step time of a boot, comparable between boots whose text differs |
| `bench/longctx.py`, `bench/longctx_edges.py`, `bench/stress.py` | long contexts end to end, their edges, mixed load |
| `bench/swa_fill.py`, `bench/prefix_fork.py` | the windowed pool run full, then old and recent conversations asked again; what a continued or a forked prompt reuses of the cache |
| `bench/stale_answer.py`, `bench/recital_rate.py`, `bench/path_compare.py` | multi-turn lookups over a long context; how often a lookup recites; decode path against prefill path |
| `bench/effort_tokens.py` | thinking tokens and right answers at each reasoning effort |
| `bench/*_test.py`, `bench/compile_check*.py`, `bench/emu/` | kernel tests (idle GPU only, their docstrings say so) and the offline checks that run without a GPU |
| `docs/results.md` | the lab notes: every hook, lever and number, in the order they were found |

The tools that talk to the server use ordinary requests and say in their docstring whether they are safe on a
serving fleet. Most write their results to `logs/` in this directory (create it once). `bench/window.py` also
reads `logs/code_prompt.json`, a chat request body with a source file as its first message, which you supply.

## Licence

MIT (see `LICENSE` at the top of this repository). Third-party material and what is not included are listed in
`NOTICE`. The image and the model weights are not part of this repository and come under their own terms.
