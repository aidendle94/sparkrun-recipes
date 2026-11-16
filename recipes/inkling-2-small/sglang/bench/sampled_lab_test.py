"""CPU-only logic test of patches/sampled_lab.py (no GPU, safe beside production in a memory-capped container):

    docker run --rm --runtime runc -m 900m --memory-swap 900m -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
      -e SPARK_GATE=0.4 -e SPARK_GATE_RID=1 -e SPARK_SAMPLED_LAB=1 -e PYTHONPATH=/patches -v $PWD/patches:/patches:ro \
      -v $PWD/bench:/bench:ro --entrypoint /opt/sglang/bin/python3 lmsysorg/sglang:dev-inkling-2-small \
      /bench/sampled_lab_test.py
Checks the request-id grammar, K of every feature against a direct computation of the rules bench/accept_sim.py
scores, that a control id runs the serving rule, that the hooks land on SGLang's own resolve_draft_params (both
module names), and the staged draft temperatures: alpha x the request's on sampled rows, untouched on argmax rows,
on rows past the batch and on requests without a lab id.
The draft graph and the accept that read the staged buffers are only exercised on a GPU (in a window: the dump of
patches/accept_stats.py shows the draft probabilities the accept used).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import math
import types

import torch

import spark_gate as g
from sglang.srt.speculative.dspark_components import dspark_draft, dspark_draft_sampler  # noqa: E402  (hooks first)
import sampled_lab as lab  # noqa: E402

bad = 0


def check(name, ok, detail=""):
    global bad
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}", flush=True)


torch.manual_seed(0)
assert g.TAU == 0.4 and g.RID, "run with SPARK_GATE=0.4 SPARK_GATE_RID=1 SPARK_SAMPLED_LAB=1"
check("request-id grammar",
      lab.parse("lab-top_0.25_0.7-w1-3") == ("top", math.log(0.25), 0.7) and lab.parse("lab-token_0_1-x")[1] == -math.inf
      and all(lab.parse(r) is None for r in ("lab-best_0.4_1.0-x", "lab-top_0.4-x", "xlab-top_0.4_1.0-x",
                                             "lab-top_0.4_0.01-x", "lab-top_0.4_9-x", "lab-top_.._1-x", "gate-k8-x")))
check("hooks: the gate's row rule and both names of resolve_draft_params",
      g._rows.__wrapped__.__module__ == "spark_gate" and lab._s["resolve"] is not None
      and dspark_draft.resolve_draft_params is lab._s["resolve"]
      and dspark_draft_sampler.resolve_draft_params is lab._s["resolve"] and lab._s["alpha_off"] == "")

# K per feature
bs, gamma, vocab = 8, 7, 3000
ep = types.SimpleNamespace(stride=8, gamma=7, max_bs=8, verify_lens_buf=torch.zeros(8, dtype=torch.int64))
logits = (torch.randn(bs, gamma, vocab) * 2.5).clamp(-30, 30).to(torch.bfloat16)
logits[:, :, 2900:] = float("-inf")  # past the vocab limit
tokens = logits.float().argmax(-1)
tokens[1] = torch.randint(0, 2900, (gamma,))  # mostly unlikely tokens
tokens[3, 3:] = torch.randint(0, 2900, (4,))
logits[2, :, 7] = 29.0  # a sure request
tokens[2] = 7
g._s["sampler"] = types.SimpleNamespace(step_logits=logits, out=tokens.reshape(-1).clone())
temps = torch.tensor([1.0, 0.7, 1.0, 1.3, 0.7, 1.0, 0.5, 1.0])
lab._s["temps"] = temps


def reference(b, feature, tau):
    """accept_sim's rules: K = 1 + the leading rows whose score is >= log tau."""
    z = logits[b].double()
    lq = torch.log_softmax(z, -1)
    tok = lq.gather(-1, tokens[b][:, None]).squeeze(-1)
    top = lq.max(-1).values
    f = {"token": tok, "tokenT": torch.log_softmax(z / temps[b].double(), -1).gather(-1, tokens[b][:, None]).squeeze(-1),
         "top": top, "topnext": top, "mix": (tok + top) / 2}[feature].cumsum(0).tolist()
    if feature == "topnext":
        f = f[1:] + f[-1:]
    k = 1
    for s in f:
        if s < math.log(tau) - 1e-9:
            break
        k += 1
    return k, min(abs(s - math.log(tau)) for s in f)


plain = [types.SimpleNamespace(rid=f"plain-{b}") for b in range(bs)]
g._s["reqs"] = plain
serving = g._rows(ep, bs).tolist()
check("requests without a lab id keep the serving rule", serving == [reference(b, "token", 0.4)[0] for b in range(bs)],
      str(serving))
g._s["reqs"] = [types.SimpleNamespace(rid=f"lab-token_0.4_1.0-{b}") for b in range(bs)]
check("the control id lab-token_0.4_1.0- runs the serving rule", g._rows(ep, bs).tolist() == serving)
wrong, tested, spread = [], 0, set()
for feature in lab.FEATURES:
    for tau in (0.02, 0.1, 0.25, 0.4, 0.6, 0.9):
        g._s["reqs"] = [types.SimpleNamespace(rid=f"lab-{feature}_{tau}_1.0-{b}") for b in range(bs)]
        got = g._rows(ep, bs).tolist()
        for b in range(bs):
            want, margin = reference(b, feature, tau)
            if margin < 1e-3:  # a score at the threshold: float32 against float64 may fall on either side
                continue
            tested += 1
            spread.add(got[b])
            if got[b] != want:
                wrong.append((feature, tau, b, got[b], want))
check("K of every feature and tau against the direct computation", not wrong and tested > 200 and len(spread) >= 6,
      f"({tested} cases, K values {sorted(spread)}) {wrong[:3]}")
mixed = ["plain", "lab-top_0.4_1.0-a", "gate-k3-x", "lab-mix_0.1_0.5-b", "gate-t0.05-x", "lab-tokenT_0.4_1.0-c",
         "lab-nothing_0.4_1.0-d", "lab-topnext_0.6_1.0-e"]
g._s["reqs"] = [types.SimpleNamespace(rid=r) for r in mixed]
got = g._rows(ep, bs).tolist()
want = [serving[0], reference(1, "top", 0.4)[0], 3, reference(3, "mix", 0.1)[0], reference(4, "token", 0.05)[0],
        reference(5, "tokenT", 0.4)[0], serving[6], reference(7, "topnext", 0.6)[0]]
check("a mixed batch: lab ids, the gate's own overrides and plain requests side by side", got == want, f"{got} {want}")
g._s["reqs"] = [types.SimpleNamespace(rid=r) for r in mixed[:3]]
check("a batch of three", g._rows(ep, 3).tolist() == want[:3])
lab._s["temps"] = None
g._s["reqs"] = [types.SimpleNamespace(rid="lab-tokenT_0.4_1.0-c")]
check("tokenT before any staging falls back to temperature 1", g._rows(ep, 1).tolist() == [serving[0]])

# staged draft temperatures, through SGLang's own resolve_draft_params
ALL = 1 << 30
info = types.SimpleNamespace(temperatures=torch.tensor([[1.0], [0.7], [1.0], [1.0], [0.0], [1.3]]),
                             top_ks=torch.tensor([[ALL], [ALL], [1], [40], [ALL], [ALL]]))
rids = ["lab-token_0.4_0.5-a", "plain", "lab-token_0.4_0.5-greedy", "lab-top_0.2_0.85-b", "lab-token_0.4_0.3-zero",
        "lab-token_0.4_1.0-c"]
alpha = torch.tensor([0.5, 1.0, 1.0, 0.85, 0.3, 1.0])
stock = dspark_draft.resolve_draft_params.__wrapped__


def buffers():
    return dspark_draft.DraftParams(temperatures=torch.full((8,), -1.0), greedy_mask=torch.zeros(8, dtype=torch.bool),
                                    step_temperatures=torch.full((8,), -1.0))


ref = stock(sampling_info=info, bs=6, out=buffers())
g._s["reqs"] = [types.SimpleNamespace(rid=r) for r in rids]
out = buffers()
got = dspark_draft_sampler.resolve_draft_params(sampling_info=info, bs=6, out=out)
check("staged temperatures: alpha x the request's on sampled rows, argmax rows untouched",
      torch.allclose(got.temperatures, ref.temperatures * alpha) and torch.equal(got.greedy_mask, ref.greedy_mask)
      and torch.allclose(got.step_temperatures, ref.step_temperatures * alpha)
      and got.step_temperatures[2] == 0 and got.temperatures[2] == 1.0,
      f"{[round(v, 4) for v in got.temperatures.tolist()]} step {[round(v, 4) for v in got.step_temperatures.tolist()]}")
check("written in place into the static buffers, rows past the batch untouched",
      got.temperatures.data_ptr() == out.temperatures.data_ptr() and out.temperatures[6:].tolist() == [-1.0, -1.0]
      and out.step_temperatures[6:].tolist() == [-1.0, -1.0])
check("the gate reads the requests' own temperatures", torch.equal(lab._s["temps"], ref.temperatures))
fresh = dspark_draft.resolve_draft_params(sampling_info=info, bs=6, device=torch.device("cpu"))
check("fresh tensors (no static buffers) get the same", torch.allclose(fresh.temperatures, ref.temperatures * alpha))
g._s["reqs"] = [types.SimpleNamespace(rid=f"plain-{i}") for i in range(6)]
same = dspark_draft.resolve_draft_params(sampling_info=info, bs=6, out=buffers())
check("requests without a lab id: SGLang's values, bit for bit",
      all(torch.equal(a, b) for a, b in zip(same, ref)))
g._s["reqs"] = None
none = dspark_draft.resolve_draft_params(sampling_info=None, bs=6, out=buffers())
want = stock(sampling_info=None, bs=6, out=buffers())
check("no sampling info, no request list: SGLang's values", all(torch.equal(a, b) for a, b in zip(none, want)))
g._s["reqs"] = [types.SimpleNamespace(rid=r) for r in rids[:2]]  # fewer requests known than the batch holds
short = dspark_draft.resolve_draft_params(sampling_info=info, bs=6, out=buffers())
check("a request list shorter than the batch: the rest keep their own",
      torch.allclose(short.temperatures, ref.temperatures * torch.tensor([0.5, 1, 1, 1, 1, 1])))

# the rule of rejection sampling the lab relies on, enumerated: a proposal at another temperature, the accept reading
# the proposal's own probabilities, emits the target's distribution
z = torch.randn(40, dtype=torch.float64) * 2
p = torch.softmax(z + torch.randn(40, dtype=torch.float64), -1)
worst = 0.0
for a in (0.3, 0.5, 0.85, 1.0, 1.5):
    q = torch.softmax(z / (a * 0.9), -1)
    accept = torch.minimum(torch.ones_like(p), p / q)
    residual = (p - q).clamp(min=0)
    emitted = q * accept + (1 - (q * accept).sum()) * residual / residual.sum()
    worst = max(worst, float((emitted - p).abs().max()))
check("rejection against the proposal's own probabilities emits the target's distribution", worst < 1e-12, f"{worst:.1e}")
print("ALL OK" if not bad else f"{bad} FAILED")
raise SystemExit(1 if bad else 0)
