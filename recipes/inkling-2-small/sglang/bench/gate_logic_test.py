"""CPU-only logic test of patches/spark_gate.py (no GPU, safe beside production in a memory-capped container):

    docker run --rm --runtime runc -m 900m --memory-swap 900m -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
      -e SPARK_GATE=0.4 -e SPARK_GATE_RID=1 -e PYTHONPATH=/patches -v $PWD/patches:/patches:ro -v $PWD/bench:/bench:ro \
      --entrypoint /opt/sglang/bin/python3 lmsysorg/sglang:dev-inkling-2-small /bench/gate_logic_test.py
Checks K against the rule bench/trace_sim.py scored, the request-id overrides, the row map of the verify forward,
that the hooks land on SGLang's classes, and the capped accept (SGLang's torch accept) on rows masked by the gate.
The Triton accept kernels and the captured graphs are only exercised on a GPU (bench/gate_run.py in a window).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import types

import torch

import spark_gate as g

torch.manual_seed(0)
assert g.TAU == 0.4 and g.RID, "run with SPARK_GATE=0.4 SPARK_GATE_RID=1"
bs, gamma, vocab = 8, 7, 3000
ep = types.SimpleNamespace(stride=8, gamma=7, max_bs=8, verify_lens_buf=torch.zeros(8, dtype=torch.int64))
logits = (torch.randn(bs, gamma, vocab) * 4).clamp(-30, 30).to(torch.bfloat16)
logits[:, :, 2900:] = float("-inf")  # past the vocab limit
tokens = logits.float().argmax(-1)
tokens[1] = torch.randint(0, 2900, (gamma,))  # a sampled request: mostly unlikely tokens
logits[2, :, 7] = 29.0  # a sure request
tokens[2] = 7
g._s["sampler"] = types.SimpleNamespace(step_logits=logits, out=tokens.reshape(-1).clone())
rid = ["a", "b", "c", "gate-k3-x", "gate-t0.05-x", "gate-k8-x", "gate-t0-x", "gate-k1-x"]
g._s["reqs"] = [types.SimpleNamespace(rid=r) for r in rid]


def reference(b, tau):
    p = torch.softmax(logits[b].float(), -1).gather(-1, tokens[b][:, None]).squeeze(-1).cumprod(0)
    doubt = (p < tau).nonzero().flatten()
    return (doubt[0].item() if len(doubt) else 7) + 1


for n in (1, 3, 8):
    k = g._rows(ep, n).tolist()
    want = [reference(0, .4), reference(1, .4), reference(2, .4), 3, reference(4, .05), 8, 8, 1][:n]
    assert k == want, (n, k, want)
print("K rule and request-id overrides: ok", g._rows(ep, 8).tolist())
assert g._rows(ep, 8)[2] == 8 and g._rows(ep, 8)[1] <= 2

# row map of a verify forward
seen = {}


class LLM:
    def forward(self, input_ids, positions, forward_batch):
        seen["src"] = g._s["src"]
        return g.mask_ids(seen["ids"])


g.install_model(types.SimpleNamespace(Inkling2CausalLLM=LLM))
g._s["epilogue"] = ep
ep.verify_lens_buf[:] = torch.tensor([3, 8, 1, 0, 0, 0, 0, 0])
mode = types.SimpleNamespace(is_target_verify=lambda: True)
seen["ids"] = torch.arange(24 * 8).view(24, 8)
out = LLM().forward(torch.zeros(24, dtype=torch.long), None, types.SimpleNamespace(forward_mode=mode, batch_size=3))
assert seen["src"].tolist() == [0, 1, 2, 0, 0, 0, 0, 0] + list(range(8, 16)) + [16] * 8, seen["src"].tolist()
assert torch.equal(out, seen["ids"][seen["src"]]) and g._s["src"] is None
assert g.mask_ids(seen["ids"]) is seen["ids"]  # outside a verify forward
ep.verify_lens_buf[:3] = torch.tensor([2, 0, 1])  # 0 rows: a request no step has set keeps its rows
LLM().forward(torch.zeros(24, dtype=torch.long), None, types.SimpleNamespace(forward_mode=mode, batch_size=3))
assert seen["src"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0] + list(range(8, 16)) + [16] * 8, seen["src"].tolist()
other = types.SimpleNamespace(is_target_verify=lambda: False)
LLM().forward(torch.zeros(24, dtype=torch.long), None, types.SimpleNamespace(forward_mode=other, batch_size=3))
assert seen["src"] is None
LLM().forward(torch.zeros(20, dtype=torch.long), None, types.SimpleNamespace(forward_mode=mode, batch_size=3))
assert seen["src"] is None
print("row map: ok")

# the hooks on SGLang's own classes, and the capped accept on CPU
g._s["epilogue"] = None
from sglang.srt.speculative.dspark_components import dspark_draft, dspark_draft_sampler, dspark_verify  # noqa: E402

for obj, name in ((dspark_verify.DsparkVerifyEpilogue, "__init__"), (dspark_verify.DsparkVerifyEpilogue, "begin_static_step"),
                  (dspark_verify.DsparkVerifyEpilogue, "_accept_static_greedy"), (dspark_verify, "accept_draft_tokens"),
                  (dspark_draft_sampler.DsparkDraftSampler, "__init__"), (dspark_draft.DraftBlockProposer, "propose")):
    assert hasattr(getattr(obj, name), "__wrapped__"), (obj, name)
import sglang.srt.models.inkling2 as inkling2  # noqa: E402

assert hasattr(inkling2.Inkling2CausalLLM.forward, "__wrapped__")
fake = types.SimpleNamespace(_fused_accept_tail=False, simulated_correct_len=None, deferred_commit=None, _post_fold=False)
assert g._refusal(fake, dspark_verify) == "", g._refusal(fake, dspark_verify)
fake._post_fold = True  # the verify graph relays the capped accept's own buffers
assert g._refusal(fake, dspark_verify) == "", g._refusal(fake, dspark_verify)
fake._fused_accept_tail = True
assert g._refusal(fake, dspark_verify)
g._s["sampler"].sharded_accept = True  # logits on vocab shards: the ranks would disagree on K, so every row is verified
assert g._rows(ep, 3) is None and g._s.get("sharded")
g._s["sampler"].sharded_accept = False
assert g._rows(ep, 3) is not None

bs = 4
cand = torch.randint(5, 900, (bs, 8))
target = torch.full((bs * 8, 1000), -5.0)
predict = torch.randint(5, 900, (bs, 8))
predict[:, :7] = cand[:, 1:]  # the target agrees with every draft ...
predict[1, 2] = 999  # ... except request 1 at its third row
predict[2, 5:] = 4  # rows the gate masks hold garbage
target.view(bs, 8, -1).scatter_(-1, predict[..., None], 5.0)
block = types.SimpleNamespace(greedy_mask=None)
kw = dict(candidates=cand, target_logits=target, draft_block=block, sampling_info=None, draft_input=None, gamma=7,
          verify_num_draft_tokens=8)
correct, bonus, trim = dspark_verify.accept_draft_tokens(**kw)
assert correct.tolist() == [7, 2, 5, 7], correct  # gate not armed: stock result
ep.verify_lens_buf[:4] = torch.tensor([0, 8, 5, 3])
g._s["epilogue"] = ep
correct, bonus, trim = dspark_verify.accept_draft_tokens(**kw)
assert correct.tolist() == [7, 2, 4, 2] and trim.tolist() == [0, 0, 1, 5], (correct, trim)
assert bonus.tolist() == [predict[0, 7], 999, predict[2, 4], predict[3, 2]], bonus
assert bonus[2] == cand[2, 5] and bonus[3] == cand[3, 3]  # the capped bonus is the next draft, which the target confirmed
print("capped eager accept (torch path): ok")
print("ALL OK")
