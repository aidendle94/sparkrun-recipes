"""Per-step DSpark trace for diagnosis windows (SPARK_TRACE=<dir inside the container>; off by default, rank 0 only).

What a decode step's cost depends on is not in any log: which experts every verify row routes to (the bytes each rank
reads, and how unevenly), how sure the draft was of each token, and how many of them the target then accepted. With
SPARK_TRACE set, sitecustomize calls note_ids from Inkling2Experts.forward (a copy of the routed ids into a buffer
per layer; under CUDA-graph capture the copy becomes a node of the graph, so the buffers hold the last replay's ids)
and observe after every verify step (DsparkStepObservers.observe_verify_step, eager, once per step). Per step:
  t        perf_counter at the end of the step (step time = difference between consecutive steps of a busy fleet)
  bs, ct   requests in the step, the scheduler's forward counter; static = no compact layout (rows are bs x 8)
  rids     request ids (bench/trace_run.py names them after phase and prompt)
  prefix, correct, commit, bonus   [bs]: tokens before the step, drafts accepted, tokens committed, the bonus token
  draft    [bs, 7] drafted tokens; target [bs, 8] the target's argmax at the 8 verify rows
  top, top_ids   [bs, 8, 2] the target's two largest logits (float32 of the BF16 logits) and their tokens at the 8
           verify rows: top[..., 0] - top[..., 1] is the margin a rounding difference has to cross to flip a greedy token
  conf     [bs, 7, 5] float32 of the draft's step logits: top-1, top-2, top-3, logsumexp, the drafted token's logit
  ids      [bs, layers, 8, top_k] int16: expert ids of each layer for the request's 8 verify rows (anchor, 7 drafts)
  k        [bs] with SPARK_GATE: the rows the gate kept (ids are the routed ids after its masking)
Steps are kept on the GPU and written every FLUSH steps, or when a step arrives after 2 s of silence, to
<dir>/trace-<n>.pt (torch.save of a list of dicts); a short request after a pause therefore flushes a window's tail.
A failure here turns the trace off and logs once; it never reaches the step.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import logging
import os
import sys
import time

import torch

DIR = os.environ.get("SPARK_TRACE", "")
ROWS = 64    # 8 requests x 8 verify rows: larger forwards (prefill) are not traced
FLUSH = 256
_experts = []  # Inkling2Experts in construction order (= layer order)
_steps = []
_state = {"files": 0, "last": 0.0, "off": False}
_log = logging.getLogger("spark_trace")


def register(module):
    _experts.append(module)


def note_ids(module, ids):
    if ids.dim() != 2 or ids.shape[0] > ROWS:
        return
    buf = getattr(module, "_trace_ids", None)
    if buf is None:
        buf = module._trace_ids = torch.zeros((ROWS, ids.shape[1]), dtype=torch.int16, device=ids.device)
    buf[: ids.shape[0]].copy_(ids)


def _flush():
    if not _steps:
        return
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, f"trace-{_state['files']:05d}.pt")
    cpu = [{k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()} for s in _steps]
    torch.save(cpu, path + ".tmp")
    os.replace(path + ".tmp", path)
    if _state["files"] == 0:
        _log.warning("[trace] %d expert layers, first file %s", len(_experts), path)
    _state["files"] += 1
    _steps.clear()


def observe(kw):
    if _state["off"] or not _experts or _experts[0].ep_rank != 0:
        return
    try:
        now = time.perf_counter()
        if now - _state["last"] > 2.0:
            _flush()
        _state["last"] = now
        bs = int(kw["bs"])
        if bs * 8 > ROWS or any(getattr(m, "_trace_ids", None) is None for m in _experts):
            return
        draft = kw["draft_tokens"].reshape(bs, -1)
        step = {"t": now, "bs": bs, "ct": int(kw["forward_ct"]), "static": kw["layout"] is None,
                "rids": [str(r.rid) for r in kw["reqs"]][:bs], "draft": draft.to(torch.int32, copy=True)}
        for name, key in (("prefix", "prefix_lens"), ("correct", "correct_len"), ("commit", "commit_lens"),
                          ("bonus", "bonus")):
            step[name] = kw[key].reshape(-1)[:bs].to(torch.int32, copy=True)
        logits = getattr(kw["draft_block"], "corrected_logits", None)
        if logits is not None and logits.dim() == 3 and logits.shape[1] == draft.shape[1]:
            logits = logits[:bs].float()
            step["conf"] = torch.cat([logits.topk(3, -1).values, torch.logsumexp(logits, -1, keepdim=True),
                                      logits.gather(-1, draft.long()[..., None])], -1)
        target = kw["target_logits"]
        if target is not None and target.shape[0] >= bs * 8:
            step["target"] = target[: bs * 8].argmax(-1).view(bs, 8).to(torch.int32)
            top = target[: bs * 8].float().topk(2, -1)
            step["top"] = top.values.view(bs, 8, 2)
            step["top_ids"] = top.indices.view(bs, 8, 2).to(torch.int32)
        gate = sys.modules.get("spark_gate")
        if gate is not None and gate.epilogue() is not None:
            step["k"] = gate.epilogue().verify_lens_buf[:bs].to(torch.int32, copy=True)
        ids = torch.stack([m._trace_ids[: bs * 8] for m in _experts])  # [layers, bs * 8, top_k], rows request-major
        step["ids"] = ids.view(len(_experts), bs, 8, -1).transpose(0, 1).contiguous()
        _steps.append(step)
        if len(_steps) >= FLUSH:
            _flush()
    except Exception:  # a diagnostic must not take a step down
        _state["off"] = True
        _log.exception("[trace] recorder failed and is now off")
