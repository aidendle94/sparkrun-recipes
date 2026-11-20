"""CPU test of patches/conf_capture.py: the recorder on made-up steps, its three files read back, the scalars against
plain arithmetic, the hidden copy through the wrapped compute_base_logits, and that other clients' requests and a
failing step leave nothing behind. No GPU, no fleet:
    docker run --rm --memory 2g --network none -v $PWD:/w:ro --entrypoint /opt/sglang/bin/python3 <image> -I /w/bench/conf_capture_test.py
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import math
import os
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
os.environ["SPARK_CONF_CAPTURE"] = tmp
os.environ["SPARK_CONF_CAPTURE_FLUSH"] = "3"
sys.path.insert(0, os.path.join(HERE, "..", "patches"))
import conf_capture as cc  # noqa: E402

G, H, V, MAXBS = 7, 48, 301, 8
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(("ok   " if cond else "FAIL ") + name + (f": {detail}" if detail else ""))


# the three install points on stand-ins of the engine's classes
class Sampler:
    def __init__(self, *, model, gamma, max_bs, device):
        self.model, self.gamma, self.max_bs = model, gamma, max_bs
        self.out = torch.empty((max_bs * gamma,), dtype=torch.int64, device=device)


class Model:
    config = types.SimpleNamespace(hidden_size=H)

    def compute_base_logits(self, hidden):
        return hidden.float().sum(-1), None


class Observers:
    calls = 0

    def observe_verify_step(self, **kwargs):
        Observers.calls += 1


mod = types.SimpleNamespace(DsparkDraftSampler=Sampler, Inkling2ForConditionalGenerationMTP=Model,
                            DsparkStepObservers=Observers)
cc.install_sampler(mod), cc.install_model(mod), cc.install_observer(mod)
model = Model()
Sampler(model=model, gamma=G, max_bs=MAXBS, device="cpu")
check("buffer made at the sampler's construction", cc._s["hbuf"] is not None and tuple(cc._s["hbuf"].shape) == (MAXBS * G, H))

gen = torch.Generator().manual_seed(3)
truth = []


def step(ct, rids, temps_topk):
    bs = len(rids)
    hidden = torch.randn((bs * G, H), generator=gen).to(torch.bfloat16)
    out, _ = model.compute_base_logits(hidden)
    z = torch.randn((bs, G, V), generator=gen) * 3
    t = torch.randn((bs * (G + 1), V), generator=gen) * 3
    z[:, :, -4:] = -math.inf                               # the padded end of the vocabulary is masked
    t[:, -4:] = -math.inf
    draft = torch.randint(0, V - 4, (bs, G), generator=gen)
    draft[:, 0] = z[:, 0].argmax(-1)                       # some drafts are the draft's best token
    t[torch.arange(bs) * (G + 1), draft[:, 0]] += 20       # and the target's
    reqs = [types.SimpleNamespace(rid=r, sampling_params=types.SimpleNamespace(temperature=T, top_k=k, top_p=1.0))
            for r, (T, k) in zip(rids, temps_topk)]
    prefix = torch.arange(bs) * 100 + ct
    correct = torch.randint(0, G + 1, (bs,), generator=gen)
    block = types.SimpleNamespace(corrected_logits=z, temperatures=torch.tensor([T for T, _ in temps_topk]),
                                  greedy_mask=torch.tensor([k == 1 for _, k in temps_topk]), vocab_shard=None)
    Observers().observe_verify_step(forward_ct=ct, reqs=reqs, bs=bs, proposal_folded=True, target_logits=t,
                                    prefix_lens=prefix, draft_tokens=draft, draft_block=block,
                                    correct_len=correct, commit_lens=correct + 1)
    for b, r in enumerate(rids):
        if r.startswith(cc.PREFIX):
            truth.append({"rid": r, "ct": ct, "hid": hidden.view(bs, G, H)[b].to(torch.float16).numpy(), "z": z[b],
                          "t": t[b * (G + 1): b * (G + 1) + G], "d": draft[b], "T": temps_topk[b][0],
                          "greedy": temps_topk[b][1] == 1, "prefix": int(prefix[b]), "n": int(correct[b])})


P = cc.PREFIX
step(1, [P + "a", "someone-else", P + "b"], [(0.7, 0), (0.7, 0), (1.0, 1)])
step(2, ["someone-else"], [(0.7, 0)])
step(3, [P + "a"], [(0.7, 0)])
check("nothing written before FLUSH recorded steps", cc._s["files"] is None)
step(4, [P + "a", P + "b", P + "c", P + "d"], [(0.7, 0), (1.0, 1), (1.0, 0), (0.3, 0)])
check("written at the third recorded step", cc._s["files"] is not None and cc._s["rows"] == 7)
step(5, [P + "c"], [(1.0, 0)])
cc._flush()
check("the stock observer still ran every step", Observers.calls == 5)
pid = os.getpid()
meta = [json.loads(x) for x in open(os.path.join(tmp, f"meta-{pid}.jsonl"))]
hid = np.fromfile(os.path.join(tmp, f"hid-{pid}.f16"), dtype=np.float16).reshape(-1, G, H)
sc = np.fromfile(os.path.join(tmp, f"sc-{pid}.f32"), dtype=np.float32).reshape(-1, G, cc.NSCAL)
check("one row per recorded request-step, none for other clients", len(meta) == len(truth) == hid.shape[0] == sc.shape[0],
      f"{len(meta)} meta, {len(truth)} expected, {hid.shape[0]} hidden, {sc.shape[0]} scalars")
check("row index runs 0..n-1", [m["i"] for m in meta] == list(range(len(meta))))
check("draft counter is the number of draft forwards so far", [m["ctr"] for m in meta] == [1, 1, 3, 4, 4, 4, 4, 5],
      str([m["ctr"] for m in meta]))
worst, rows_ok = 0.0, True
for m, w, h, s in zip(meta, truth, hid, sc):
    same = m["rid"] == w["rid"] and m["ct"] == w["ct"] and m["prefix"] == w["prefix"] and m["n"] == w["n"] \
        and m["commit"] == w["n"] + 1 and m["x"] == w["d"].tolist() and m["greedy"] == w["greedy"] and m["k"] == G + 1
    if not same or not np.array_equal(h, w["hid"]):
        rows_ok = False
        print("     row", json.dumps(m)[:200])
    lz, lt = torch.log_softmax(w["z"].double(), -1), torch.log_softmax(w["t"].double(), -1)
    d = w["d"].unsqueeze(-1)
    want = torch.zeros((G, cc.NSCAL), dtype=torch.float64)
    want[:, 0] = lz.gather(-1, d).squeeze(-1)
    want[:, 1:3] = lz.topk(2, -1).values
    want[:, 3] = -(lz.exp() * torch.where(lz.exp() > 0, lz, torch.zeros_like(lz))).sum(-1)
    want[:, 4] = lt.gather(-1, d).squeeze(-1)
    want[:, 5] = lt.max(-1).values
    want[:, 6] = (lt.argmax(-1) == w["d"]).double()
    if w["greedy"]:
        want[:, 7] = (lz.argmax(-1) == w["d"]).double()
        want[:, 8] = want[:, 6]
    else:
        want[:, 7] = torch.softmax(w["z"].double() / w["T"], -1).gather(-1, d).squeeze(-1)
        want[:, 8] = torch.softmax(w["t"].double() / w["T"], -1).gather(-1, d).squeeze(-1)
    worst = max(worst, float((torch.from_numpy(s).double() - want).abs().max()))
check("meta, hidden bytes and drafted tokens of every row", rows_ok)
check("scalars against float64 arithmetic, all finite", worst < 2e-4 and bool(np.isfinite(sc).all()),
      f"largest difference {worst:.2e}")
check("a drafted token that is both models' best is marked", all(s[0, 6] == 1.0 for s in sc) and
      all(abs(s[0, 0] - s[0, 1]) < 1e-6 for s in sc))

# a pause file, and a step that raises: nothing recorded, nothing raised, the recorder goes off
open(os.path.join(tmp, "pause"), "w").close()
before = len(truth)
step(6, [P + "a"], [(0.7, 0)])
truth[:] = truth[:before]
os.remove(os.path.join(tmp, "pause"))
cc._flush()
check("nothing recorded while the pause file exists",
      sum(1 for _ in open(os.path.join(tmp, f"meta-{pid}.jsonl"))) == len(meta))
Observers().observe_verify_step(forward_ct=7, reqs=[types.SimpleNamespace(rid=P + "x")], bs=1)   # missing arguments
check("a failing step turns the recorder off without raising", cc._s["off"] is True)
out, _ = model.compute_base_logits(torch.ones((G, H), dtype=torch.bfloat16))
check("the model's projection still answers with the recorder off", tuple(out.shape) == (G,))
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
