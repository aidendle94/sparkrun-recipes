"""Training data for a draft confidence head (SPARK_CONF_CAPTURE=<dir inside the container>; capture windows only).

The engine's own confidence head (DSparkConfidenceHead) reads the draft's final hidden state of each drafted
position, the tensor compute_base_logits projects to the base logits. The Inkling-2 draft ships without one, and the
gate (patches/spark_gate.py) reads the draft's token probability instead. This records, for requests whose id starts
with PREFIX (send them with SPARK_GATE_RID=1 and a "gate-k8-" id so that all 8 rows are verified), per decode step:
  hidden   [7, hidden] float16: what the confidence head would see (a copy node in the draft graph writes the tensor
           into a buffer made at the sampler's construction; a counter beside it counts the draft forwards)
  scalars  [7, 9] float32 per drafted position j (the target row that judges draft j is row j - 1 of the 8):
           0 draft log p of the drafted token at temperature 1 (what the gate reads)   1, 2 the draft's two largest log p
           3 the draft's entropy   4 target log p of the drafted token at temperature 1   5 the target's largest log p
           6 1 if the target's argmax is the drafted token   7, 8 draft and target probability of the drafted token
           at the request's temperature (their ratio is the sampled accept's; for a greedy request 7 = the draft's
           argmax is the drafted token and 8 = column 6)
  meta     a JSON line: rid, forward counter, batch size and row, tokens before the step (the context length),
           drafts accepted, tokens committed, the gate's rows, temperature, greedy flag, top_p, top_k, the draft
           counter, the 7 drafted tokens, whether target logits were there
Only a rank whose directory exists records (every rank computes the same step: make it on one node), and it does
nothing while <dir>/pause exists. Steps stay on the GPU and are written every FLUSH steps or when a step arrives
after 2 s of silence, to hid-<pid>.f16, sc-<pid>.f32 and meta-<pid>.jsonl; each write is synced and dropped from the
page cache (GB10's driver does not evict file cache for its own allocations). Requests of other clients are never
recorded. Any failure turns the recorder off and logs once; it never reaches a step.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import json
import logging
import os
import sys
import time

import torch

DIR = os.environ.get("SPARK_CONF_CAPTURE", "")
PREFIX = os.environ.get("SPARK_CONF_CAPTURE_PREFIX", "gate-k8-cc-")
FLUSH = int(os.environ.get("SPARK_CONF_CAPTURE_FLUSH") or 32)
NSCAL = 9
_log = logging.getLogger("conf_capture")
_s = {"hbuf": None, "ctr": None, "off": False, "steps": [], "last": 0.0, "files": None, "rows": 0, "noted": set(),
      "written": 0}


def active():
    return bool(DIR) and not _s["off"] and os.path.isdir(DIR)


def _note(key, msg, *args):
    if key not in _s["noted"]:
        _s["noted"].add(key)
        _log.warning("[conf_capture] " + msg, *args)


def _fail(where):
    _s["off"] = True
    _log.exception("[conf_capture] off after an error in %s", where)


def scalars(z, t, d, temp, greedy):
    """[g, NSCAL] float32 of one request: z [g, V] draft step logits, t [g, V] target logits of the judging rows (or
    None), d [g] drafted tokens, temp the request's temperature, greedy its flag."""
    d = d.long().unsqueeze(-1)
    out = torch.zeros((d.shape[0], NSCAL), dtype=torch.float32, device=d.device)
    lz = torch.log_softmax(z.float(), -1)
    top = lz.topk(2, -1)
    out[:, 0] = lz.gather(-1, d).squeeze(-1)
    out[:, 1:3] = top.values
    prob = lz.exp()
    out[:, 3] = -(prob * torch.where(prob > 0, lz, torch.zeros_like(lz))).sum(-1)  # masked entries are -inf: 0, not NaN
    drafted_best = (top.indices[:, :1] == d).squeeze(-1).float()
    sampled = (not greedy) and temp > 0
    out[:, 7] = torch.softmax(z.float() / temp, -1).gather(-1, d).squeeze(-1) if sampled else drafted_best
    if t is None:
        out[:, 4:7] = float("nan")
        out[:, 8] = float("nan")
        return out
    lt = torch.log_softmax(t.float(), -1)
    best = lt.max(-1)
    out[:, 4] = lt.gather(-1, d).squeeze(-1)
    out[:, 5] = best.values
    out[:, 6] = (best.indices.unsqueeze(-1) == d).squeeze(-1).float()
    out[:, 8] = torch.softmax(t.float() / temp, -1).gather(-1, d).squeeze(-1) if sampled else out[:, 6]
    return out


def _open():
    pid = os.getpid()
    _s["files"] = (open(os.path.join(DIR, f"hid-{pid}.f16"), "ab"), open(os.path.join(DIR, f"sc-{pid}.f32"), "ab"),
                   open(os.path.join(DIR, f"meta-{pid}.jsonl"), "a"))
    _log.warning("[conf_capture] writing to %s (pid %d), requests %s*, every %d steps", DIR, pid, PREFIX, FLUSH)


def _flush():
    steps, _s["steps"] = _s["steps"], []
    if not steps:
        return
    if _s["files"] is None:
        _open()
    hid_f, sc_f, meta_f = _s["files"]
    hid = torch.cat([s["hid"] for s in steps]).cpu().numpy()
    sc = torch.cat([s["sc"] for s in steps]).cpu().numpy()
    ints = torch.cat([s["ints"] for s in steps]).cpu().tolist()
    ctrs = torch.stack([s["ctr"] for s in steps]).cpu().tolist()
    hid_f.write(hid.tobytes())
    sc_f.write(sc.tobytes())
    row = 0
    for s, ctr in zip(steps, ctrs):
        for meta in s["meta"]:
            prefix, n, commit, k, *x = ints[row]
            meta.update(i=_s["rows"], prefix=prefix, n=n, commit=commit, k=k, x=x, ctr=ctr)
            meta_f.write(json.dumps(meta) + "\n")
            row += 1
            _s["rows"] += 1
    for f in _s["files"]:
        f.flush()
        os.fsync(f.fileno())
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    _s["written"] += 1
    if _s["written"] in (1, 10) or _s["written"] % 200 == 0:
        _log.warning("[conf_capture] %d request-steps written (%d writes)", _s["rows"], _s["written"])


def observe(kw):
    if not active() or os.path.exists(os.path.join(DIR, "pause")):
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        now = time.perf_counter()
        if now - _s["last"] > 2.0:
            _flush()
        _s["last"] = now
        hbuf = _s["hbuf"]
        if hbuf is None:
            return _note("nobuf", "no hidden buffer on this rank: nothing is recorded")
        bs = int(kw["bs"])
        reqs = list(kw["reqs"])[:bs]
        take = [b for b, r in enumerate(reqs) if str(r.rid).startswith(PREFIX)]
        if not take:
            return
        draft = kw["draft_tokens"].reshape(bs, -1)
        g = draft.shape[1]
        if bs * g > hbuf.shape[0]:
            return _note("rows", "a step of %d x %d drafts is larger than the buffer (%d rows)", bs, g, hbuf.shape[0])
        block = kw["draft_block"]
        logits = getattr(block, "corrected_logits", None)
        if logits is None or logits.dim() != 3 or logits.shape[0] < bs or logits.shape[1] != g:
            return _note("logits", "no step logits (%s): nothing is recorded",
                         None if logits is None else tuple(logits.shape))
        if getattr(block, "vocab_shard", None) is not None:
            return _note("shard", "the step logits are a vocab shard: nothing is recorded")
        target = kw["target_logits"]
        if target is not None and (target.dim() != 2 or target.shape[0] < bs * (g + 1)):
            _note("target", "target logits %s do not hold %d x %d rows: recorded without them", tuple(target.shape),
                  bs, g + 1)
            target = None
        vocab = logits.shape[2] if target is None else min(logits.shape[2], target.shape[1])
        temps = block.temperatures.reshape(-1)[:bs].float().tolist() if _s["rows"] == 0 else None
        gate = sys.modules.get("spark_gate")
        ep = None if gate is None else gate.epilogue()
        index = torch.tensor(take, dtype=torch.int64, device=draft.device)
        cols = [kw[name].reshape(-1)[:bs].to(torch.int64) for name in ("prefix_lens", "correct_len", "commit_lens")]
        cols.append(ep.verify_lens_buf[:bs].to(torch.int64) if ep is not None else torch.full_like(cols[0], g + 1))
        ints = torch.cat([torch.stack(cols, -1), draft.to(torch.int64)], -1).index_select(0, index)
        hid = hbuf[: bs * g].view(bs, g, -1).index_select(0, index).to(torch.float16)
        sc, meta = [], []
        for b in take:
            req = reqs[b]
            params = getattr(req, "sampling_params", None)
            temp = float(getattr(params, "temperature", 1.0) or 0.0)
            greedy = int(getattr(params, "top_k", 0) or 0) == 1
            judge = None if target is None else target[b * (g + 1): b * (g + 1) + g, :vocab]
            sc.append(scalars(logits[b, :, :vocab], judge, draft[b], temp, greedy))
            meta.append({"rid": str(req.rid), "ct": int(kw["forward_ct"]), "bs": bs, "b": b, "T": round(temp, 4),
                         "greedy": greedy, "top_p": float(getattr(params, "top_p", 1.0) or 0.0),
                         "top_k": int(getattr(params, "top_k", 0) or 0), "tl": target is not None,
                         "folded": bool(kw.get("proposal_folded")), "t": round(time.time(), 3)})
        if temps is not None:
            _log.warning("[conf_capture] first step: bs %d, drafts %d, vocab %d, hidden %s, target logits %s, block "
                         "temperatures %s", bs, g, vocab, tuple(hid.shape), None if target is None else
                         tuple(target.shape), temps)
        _s["steps"].append({"hid": hid, "sc": torch.stack(sc), "ints": ints, "ctr": _s["ctr"].clone(), "meta": meta})
        if len(_s["steps"]) >= FLUSH:
            _flush()
    except Exception:  # a recorder must not take a step down
        _fail("observe")


def install_observer(mod):
    stock = mod.DsparkStepObservers.observe_verify_step

    def observe_verify_step(self, **kwargs):
        stock(self, **kwargs)
        observe(kwargs)

    observe_verify_step.__wrapped__ = stock
    mod.DsparkStepObservers.observe_verify_step = observe_verify_step


def install_sampler(mod):
    cls = mod.DsparkDraftSampler
    stock = cls.__init__

    def __init__(self, *args, **kwargs):
        stock(self, *args, **kwargs)
        if not DIR or not os.path.isdir(DIR):
            _log.warning("[conf_capture] no directory %s on this rank: it records nothing", DIR)
            return
        try:  # eager, at the worker's construction: a buffer must not be made under graph capture
            hidden = int(self.model.config.hidden_size)
            rows = int(self.max_bs) * int(self.gamma)
            _s["hbuf"] = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=self.out.device)
            _s["ctr"] = torch.zeros((), dtype=torch.int64, device=self.out.device)
            _log.warning("[conf_capture] armed: hidden buffer %d x %d on %s", rows, hidden, self.out.device)
        except Exception:
            _fail("the sampler's construction")

    __init__.__wrapped__ = stock
    cls.__init__ = __init__


def install_model(mod):
    cls = mod.Inkling2ForConditionalGenerationMTP
    stock = cls.compute_base_logits

    def compute_base_logits(self, hidden):
        hbuf = _s["hbuf"]
        if hbuf is not None and not _s["off"]:
            try:  # under graph capture these two become nodes of the draft graph: the buffer then holds every replay's
                if hidden.dim() == 2 and hidden.shape[0] <= hbuf.shape[0] and hidden.shape[1] == hbuf.shape[1]:
                    hbuf[: hidden.shape[0]].copy_(hidden)
                    _s["ctr"].add_(1)
                else:
                    _note("shape", "draft hidden %s does not fit the buffer %s: not copied", tuple(hidden.shape),
                          tuple(hbuf.shape))
            except Exception:
                _fail("the hidden copy")
        return stock(self, hidden)

    compute_base_logits.__wrapped__ = stock
    cls.compute_base_logits = compute_base_logits
