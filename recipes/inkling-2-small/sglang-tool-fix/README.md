# Inkling-2-Small on 4× DGX Spark — Tool-Call & Long-Horizon System-Prompt Fix (experimental)

This recipe serves `thinkingmachines/Inkling-2-Small` on four DGX Spark (GB10) nodes with the stock weights, the 4-bit (NVFP4) KV cache (1,409,000 tokens), and a server-side hook (`patches/system_reinforce.py`) that restores long-horizon adherence to system-prompt instructions such as parallel tool-call batching and tool-selection policies.

## Why stock Inkling-2-Small stops obeying system-prompt tool rules

40 of Inkling-2-Small's 48 layers use 1,024-token sliding-window attention (SWA). Only 8 global layers (`5, 11, 17, 23, 29, 35, 41, 47`) can attend to the leading system prompt once a prompt grows past 1,024 tokens. With seven tool definitions declared, even Turn 1 of a coding agent session exceeds 2,700 tokens, and decode telemetry on the 8 global layers shows median attention to the system prompt dropping to 0.8–2.2% over long sessions. As context grows past 100K tokens, the model reverts to single-call `bash` habits even when the system prompt explicitly requires parallel tool calls.

## What this recipe changes

The model weights and attention kernels remain 100% stock (`aidendle94/Inkling-2-Small-DGX-Spark`). One MIT-licensed request hook (`../sglang/patches/system_reinforce.py`, enabled by `SYSTEM_REINFORCE=user` → `SPARK_SYSTEM_REINFORCE=user`) wraps `render_inkling_messages`, covering `/v1/chat/completions`, `/v1/messages`, and `/v1/responses`:

- **Gated for zero impact on short prompts:** Single-turn requests with no assistant history and under 3,600 prompt characters (about 1,000 tokens), requests without a `system`/`developer` message, and assistant prefills pass through bit-for-bit untouched.
- **100% general directive extraction:** When a request crosses the 1,024-token SWA boundary or enters multi-turn history, the hook extracts the caller's active system-prompt directives (up to 1,200 characters, verbatim if already shorter) with no task-specific keywords or hardcoded rules.
- **Transient `role: user` reinforcement:** The extracted `<system-reminder>` block is appended transiently to a shallow copy of the message list right before tokenization, after `patches/reasoning_cache.py` hashes and fills history. Because the reminder is never stored in conversation history, SGLang keeps full radix prefix-cache hits across prior turns.

## Measured results

Measured on a four-node DGX Spark fleet (2026-11-27, Pi coding scaffold, `thinking=high`, `temperature=1.0`, system-prompt-only tool-batching policy, direct against port 8210 with no proxy):

### 70-turn repository translation benchmark (`DSH-lite`, 15,589 files)

| Configuration | Max context | CJK chars cleaned | T41–50 batch rate | T51–60 batch rate | **T61–70 batch rate** | Shell-lookup violations (all 70 / T31–70) |
|---|---:|---:|---:|---:|---:|---:|
| Stock weights (`sglang-nvfp4-kv`) | 118.9K | 2.89M | 0.0% (1.00/t) | 0.0% (1.00/t) | **0.0% (1.00/t)** | 67 / 39 |
| Weight edit tested & rejected (`log_scaling` + 1.5× 33 retrieval heads) | 143.7K | 2.91M | 0.0% (1.00/t) | 0.0% (1.00/t) | **0.0% (1.00/t)** | 51 / 30 |
| **This recipe (`SYSTEM_REINFORCE=user`, stock weights)** | **248.8K** | **2.89M** | **70.0% (1.90/t)** | **70.0% (2.40/t)** | **60.0% (2.20/t)** | **27 / 3** |

At Turns 61–70 (205K–249K tokens of context), this recipe issued batches including `read×4`, `edit×5`, `edit×3`, and `grep×2`, with only 1 shell-lookup violation in the final 10 turns.

### Unit-tested coding regression check (`ledger_service`, 24 pytest tests)

| Configuration | Tests | Turns | All batchable turns | T16–35 batchable turns | Shell-lookup violations |
|---|---:|---:|---:|---:|---:|
| Stock weights | 24/24 | 25 | 78.3% (3.04/t) | 75.0% (3.00/t) | 2 |
| Weight edit tested & rejected | 24/24 | 27 | 10.5% (1.10/t) | 0.0% (1.00/t) | 16 |
| **This recipe (`SYSTEM_REINFORCE=user`, stock weights)** | **24/24** | **26** | **77.8% (3.89/t)** | **83.3% (4.17/t)** | **1** |

## Run it

```bash
git clone https://github.com/aidendle94/sparkrun-recipes
cd sparkrun-recipes/recipes/inkling-2-small/sglang-tool-fix
./quickstart.sh 192.0.2.11 192.0.2.12 192.0.2.13 192.0.2.14   # your four fabric addresses, this node first
```

Day-to-day controls:

```bash
./quickstart.sh --status                # serving, KV cache, and system-reinforce status
./quickstart.sh --stop                  # stop the fleet and its watchdog
launch/relaunch.sh                      # start or restart with this recipe
SYSTEM_REINFORCE= launch/relaunch.sh    # disable transient reinforcement (stock behavior)
KV_NVFP4= launch/relaunch.sh            # switch back to the 8-bit (FP8) KV cache
```

Optional environment tuning (passed through `NCCL_EXTRA`):
- `SPARK_SYSTEM_REINFORCE_MAX_CHARS` (default `1200`): maximum characters of condensed system instructions injected per turn.
- `SPARK_SYSTEM_REINFORCE_MIN_CHARS` (default `3600`): turn-0 prompt character threshold below which single-turn prompts stay bit-for-bit untouched.
