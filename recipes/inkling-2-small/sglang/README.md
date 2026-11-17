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
- The numbers are single boots on one fleet. The defaults changed several times a day, and the newest have days of
  serving behind them, not weeks: the sharded experts became the default on 2026-11-15; the 500K configuration
  (524,288 tokens of context, the 793,000-token pool, the three prefill switches) served for a day through the
  environment and became the launcher's default on 2026-11-17.
- There are open problems, listed under [Known problems](#known-problems). One of them is a scheduler hang that
  happened once and was never reproduced.

If you want a recipe that has soaked, use [`deepseek-v4.1-flash/sglang`](../../deepseek-v4.1-flash/sglang/).

Glossary. **TP4/EP4**: tensor parallel over the four nodes, with the experts split four ways. **DSpark**: the
speculative decoding built into the engine: a small draft model proposes up to 7 tokens, the target verifies them in
one forward. **NVFP4 / MXFP8**: 4- and 8-bit floating-point weight formats with shared block scales; 40 of the
model's expert layers are NVFP4, 8 are MXFP8. **RoCE**: RDMA over Converged Ethernet, between the Sparks' ConnectX-7
ports. **Pool**: the KV cache, counted in tokens, shared by all running and cached requests.

## What you get

Measured on four Sparks over a switched RoCE fabric, 2026-11-13 to 2026-11-16. Every row is one boot. The
launcher's defaults are the configuration of the last of those days; a row measured on an earlier one says so.
Conditions and the history of each number are in [docs/results.md](docs/results.md).

| what | measured |
|---|---|
| decode, one stream, greedy, coding questions of 400 tokens | 60–61 tok/s (37 with the image's own decode step, on another set of such questions) |
| decode, one stream, temperature 0.7 | 56.6 tok/s |
| decode, four / eight streams at a time, greedy | 106 / 130 tok/s for the batch (about 26 / 16 a stream) |
| prefill, ~15K-token prompts | 2,400–2,500 tok/s (before the prefill switches; not measured again with them) |
| cold prefill of 30K / 120K tokens | 3,110 / 2,786 tok/s (2,474 / 2,241 without the three prefill switches) |
| cold prefill of 130K / 260K / 500K tokens | 61 / 142 / 348 s before the prefill switches; with them 120K takes 43 s (54 without) and 300K 135 s |
| decode behind a long context (130K–500K) | 52–55 tok/s on the 600K configuration; 68 at 130K and 60 at 500K on the 793K one |
| context length | 524,288 (the default since 2026-11-17; 262,144 before) |
| KV pool | 793,000 tokens with the defaults (a cap; see [Run it](#run-it)). Before them: 545K–600K with the 600K cap most rows here were measured on, 493K–805K when it was sized at each boot without a cap |
| concurrent requests | 8 |
| tool calls, 283 cases over five wire formats (chat, Anthropic messages, each also streamed, and responses), two trials | 250 right, 20 soft, 13 failed (10 are `tool_choice: none`, 3 slips on the sampled trial); 12 of 12 agent loops. With the prefill switches (the defaults): 252 / 20 / 11, and no case of the temperature-0 trial changed its verdict |
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

The ones that make it faster or larger. All are launcher variables, all on by default, each with its own off
switch:

| launcher variable (default) | what it does | cost |
|---|---|---|
| `KV_DTYPE=fp8_e4m3` | FP8 KV cache with a reader for SM12x: 1.9x the pool at the same speed | prefill NLL within 0.02 of BF16 |
| `GATE=0.4` | verifies only the draft rows the draft is confident of: +15 % decode | greedy text moves at near-ties only, by less than a second request in the batch moves it |
| `DENSE_I8=draft,dhead,dfree` | the draft reads int8 copies of its own weights: +5 % | none for the target (held-out NLL identical); about 5 % of the pool |
| `DENSE_P12=attn,head,free` | the target's attention weights packed into 12 bits without rounding: −4.5 % step time for one stream | none: exact |
| `TP_SHARD=half,fp8` | every node holds all 384 experts at a quarter of their width instead of 96 whole ones, so no node waits for the busiest: −18 % step time | not bit-identical to the stock placement (rounding only; mean NLL unchanged) |
| `REASONING_CACHE=1` | the server puts an answer's reasoning back into history turns that arrive without it | prompts grow by that reasoning |

Three more grow the pool in the same bytes (defaults since 2026-11-17, served with since 2026-11-16). None of them
touches the target's arithmetic:

| launcher variable (default) | what it does | cost |
|---|---|---|
| `SWA_RATIO=0.05` (`--swa-full-tokens-ratio`) | halves the pool of the 40 windowed layers, which was the slack: a pool token costs 6,349 bytes a node instead of 7,578 | about 35 cached conversation ends keep their window instead of 53–58; one that lost it is prefilled again when it comes back |
| `WMARK_BOOST=1` (`SPARK_WMARK_BOOST`) | sizes the pool from the memory a node has without the kernel's temporary watermark boost, so that every boot gets the same pool (`patches/wmark_boost.py`) | none |
| `DRAFT_KV_FP8=1` (`SPARK_DRAFT_KV_FP8`) | the draft's KV pools as FP8 with a reader of their own (`patches/draft_decode.py`): 5,734 bytes a token, 793,000 tokens where 716,000 fitted | about 1 % of step time for one short stream, nothing at 4 and 8 streams; accepted tokens a step unchanged; the draft is faster behind a long context |

`POOL` is the cap on the pool (`--max-total-tokens`). Unset, it follows these three: 793,000, 716,000 with
`DRAFT_KV_FP8=`, 600,000 with `SWA_RATIO=` as well, the same bytes each time. The numbers are in
[docs/results.md](docs/results.md#kv-cache-what-a-token-costs-and-what-grows-the-pool-2026-11-1516).

And three for prefill (defaults since 2026-11-17, served with since 2026-11-16 17:20 UTC). Together they take a
cold prefill from 2,474 to 3,110 tok/s at 30K tokens (+26 %) and from 2,241 to 2,786 at 120K (+24 %); decode is
unchanged. The first two keep every value of the model bit for bit; the third does not:

| launcher variable (default) | what it does | cost |
|---|---|---|
| `PREFILL_GLUE=epilogue,stream,gate,qkvr,relbias` (`SPARK_PREFILL_GLUE`) | the small steps between a layer's big kernels (residual adds, norms, the closing convolution, the projections of a global layer, the relative bias) in fewer passes over memory, same operations in the same order (`patches/prefill_glue.py`): +11 % at 30K, +10 % at 120K | none: exact (0 differing values against the stock path on every node, NLL equal token by token) |
| `BIAS_FAR=f2` (`SPARK_FA4_BIAS_FAR`) | the SM120 attention kernel rebuilt from the image's source so that key blocks beyond the reach of the relative bias skip the bias bookkeeping (`patches/fa4_bias_far.py`): 2 % of a 120K prefill, 3.6 % of a 300K one | none: exact (output and log-sum-exp equal to the stock kernel's bit for bit); refuses to boot if that source has changed |
| `FI_MXFP8=1` (`SPARK_FI_MXFP8`) | the 8 MXFP8 expert layers on FlashInfer's FP8 grouped GEMM instead of the BF16 Marlin one (`patches/fi_mxfp8.py`): +11 % at 30K, +10 % at 120K | not bit-identical: the same products summed in another order. Held-out NLL 0.9840 against 0.9840; per token the difference is 0.006 on average, above 0.1 for 0.5 % of the tokens |

Set `FI_MXFP8=` to keep the target's arithmetic as it was; the other two are worth 12 % on their own. The numbers
and what was tried and dropped are in
[docs/results.md](docs/results.md#prefill-the-glue-the-far-bias-kernel-and-mxfp8-off-marlin-2026-11-16).

Each of the twelve is switched off by setting it empty on the relaunch (`KV_DTYPE=auto` for the BF16 pool). The
switches are independent of each other, with two exceptions: `FI_MXFP8` does nothing without `fp8` in `TP_SHARD`,
and `DRAFT_KV_FP8` does nothing without the FP8 KV cache. `TP_SHARD= DENSE_P12=` is the stack as it served before
2026-11-15, `PREFILL_GLUE= BIAS_FAR= FI_MXFP8=` as it served before 2026-11-16 17:20 UTC.

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

With nothing set, the launcher starts the 500K configuration the author's fleet serves since 2026-11-16: 524,288
tokens of context, `MEMFRAC=0.94`, a pool capped at 793,000 tokens, the three prefill switches and one scheduling
setting, `--prefill-decode-interval 2` (`PREFILL_DECODE_INTERVAL`): after a prefill batch the scheduler runs two
decode rounds before it takes the next one, so requests that are already decoding keep getting tokens through a
long prefill. A new request still waits for the whole prefill ahead of it. The setting has been on the fleet's line
since the 500K stress test and was not measured against a boot without it; `PREFILL_DECODE_INTERVAL=` leaves the
scheduling to the engine.

The cap goes with the pool settings: the same bytes hold 716,000 tokens with `DRAFT_KV_FP8=` and 600,000 with
`SWA_RATIO=` as well, and the launcher picks the cap accordingly. A cap above what the bytes hold is sized down by
the engine. The bytes are what the rank-0 node has free at boot, so the cap that fits is a property of the site:
`POOL=<tokens>` sets another, `POOL=` none (the engine then sizes the pool at boot). For a combination no cap was
measured for (`KV_DTYPE=auto`, another ratio) the launcher sets none and says so.

Earlier configurations with today's launcher:

```bash
# 600,000-token pool, no prefill switches (what most rows of the table above were measured on)
SWA_RATIO= WMARK_BOOST= DRAFT_KV_FP8= PREFILL_GLUE= BIAS_FAR= FI_MXFP8= launch/relaunch-inkling2.sh

# the defaults before 2026-11-17: 262,144 tokens of context, the pool sized at each boot, the engine's scheduling
CTX=262144 MEMFRAC=0.93 POOL= SWA_RATIO= WMARK_BOOST= DRAFT_KV_FP8= PREFILL_GLUE= BIAS_FAR= FI_MXFP8= \
  PREFILL_DECODE_INTERVAL= launch/relaunch-inkling2.sh
```

`NCCL_EXTRA` (extra `-e NAME=value` arguments of the containers) and `EXTRA_ARGS` (extra server arguments) come
after the defaults on the command line, so a value given there wins.

Stop: kill the watchdog (the pid is in the last `watchdog started` line of `~/inkling2-watchdog.log`), then
`launch/launch-inkling2.sh --stop`. The watchdog probes `/health` and a three-token chat completion every 60 s and
relaunches the fleet after three failures with the environment it was started in; before it does, it saves the
state, logs and Python stacks of every rank under `$LOG_DIR/postmortem/`.

Every launcher variable, these and the others (`PORT`, `MAXREQ`, `CHUNK`, `SERVED`, `SERVED_ALIASES`, `MTP`,
`ROCE_AR`, `FI_MOE`, `PAGED_DECODE`), is documented at the top of `launch/launch-inkling2.sh`. Leave
`PAGED_DECODE=1`.

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
- **The pool depends on what the rank-0 node has free at boot.** Sized without a cap it differed from boot to boot
  (493K–805K), because the kernel's watermark boost moves the free memory by gigabytes. `WMARK_BOOST=1` (the
  default since 2026-11-17) corrects the sizing for that, and the default cap of 793,000 is what the author's fleet
  gets; a node with less memory free gets a smaller pool, and the boot log of rank 0 says how large.
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
| `bench/glue_check.py`, `bench/bias_far_window.py`, `bench/fi_mxfp8_probe.py`, `bench/fi_mxfp8_test.py` | the prefill switches against the stock path on one idle GPU: differing values, bits, timings |
| `bench/accept_sim.py`, `bench/sampled_arms.py`, `bench/sampled_lab_check.py`, `bench/timers_report.py` | lab tools: gate rules and draft temperatures scored on dumped steps, arms of one boot side by side, the timers' log lines as a table |
| `bench/*_test.py`, `bench/compile_check*.py`, `bench/emu/` | kernel tests (idle GPU only, their docstrings say so) and the offline checks that run without a GPU |
| `docs/results.md` | the lab notes: every hook, lever and number, in the order they were found |

The tools that talk to the server use ordinary requests and say in their docstring whether they are safe on a
serving fleet. Most write their results to `logs/` in this directory (create it once). `bench/window.py` also
reads `logs/code_prompt.json`, a chat request body with a source file as its first message, which you supply.

## Licence

MIT (see `LICENSE` at the top of this repository). Third-party material and what is not included are listed in
`NOTICE`. The image and the model weights are not part of this repository and come under their own terms.
