#!/usr/bin/env python3
"""CPU check of the lookup drafter (overlay/lookup_draft.py) against the pure-Python reference (overlay/lookup_index.py).

    docker run --rm -e TRITON_INTERPRET=1 -e PYTHONPATH=/w/overlay -v <repo>:/w:ro [-v <data>:/data:ro] \
        --entrypoint python3 aidendle94/sparkrun-sglang-dsv41-gb10:production-1.6 \
        /w/tests/test_lookup_draft_cpu.py [/data/lookup_data.json]

The Triton matcher runs in the interpreter. Checks:
  1. match/lookup_block equal LookupIndex(4).propose(5) at every position of every sequence, whole context and windowed;
  2. a decode loop driven through the hooks' own entry points (write_prefill in two chunks, apply, write_verify), with
     a target that accepts the longest correct prefix, takes exactly the same steps as the same loop run on LookupIndex,
     rows are reused by later requests, and the table holds the committed tokens after every step;
  3. the sampling patch turns replaced positions of corrected_logits into one-hot distributions on the drafted token
     and leaves every other position bit-identical.
Sequences come from a lookup_collect.py output when given (prompt + greedy output ids), else synthetic text with
copied spans.

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import json
import os
import random
import sys
import types

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["SPARK_LOOKUP_LOG_SECONDS"] = "0"

import msgspec  # noqa: E402
import torch  # noqa: E402

import lookup_draft as ld  # noqa: E402
from lookup_index import LookupIndex  # noqa: E402

GAMMA = 5
GARBAGE = 1_000_003          # a draft token no sequence contains


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: object
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    draft_hidden: object = None


def samples(path):
    if path:
        return [(d["prompt_ids"], d["output_ids"]) for d in json.load(open(path))]
    rng = random.Random(7)
    out = []
    for _ in range(4):
        prompt = [rng.randrange(50, 5000) for _ in range(3000)]
        gen = []
        while len(gen) < 800:          # alternate fresh text and copies of earlier spans
            if rng.random() < 0.5:
                gen += [rng.randrange(50, 5000) for _ in range(rng.randrange(5, 40))]
            else:
                src = prompt + gen
                a = rng.randrange(0, len(src) - 60)
                gen += src[a:a + rng.randrange(8, 60)]
        out.append((prompt, gen[:800]))
    return out


def check_matcher(seqs, window):
    width = max(len(s) for s in seqs) + 8
    table = torch.zeros(len(seqs), width, dtype=torch.int32)
    for r, s in enumerate(seqs):
        table[r, :len(s)] = torch.tensor(s, dtype=torch.int32)
    checked = 0
    for L in range(0, width, 97):
        rows = [r for r, s in enumerate(seqs) if L < len(seqs[r])]
        if not rows:
            continue
        rows_t = torch.tensor(rows, dtype=torch.int64)
        lens = torch.full((len(rows),), L, dtype=torch.int64)
        anchor = torch.tensor([seqs[r][L] for r in rows], dtype=torch.int64)
        toks, avail = ld.lookup_block(table, rows_t, lens, anchor, GAMMA, window)
        for i, r in enumerate(rows):
            seq = seqs[r][:L + 1]
            if window:
                # reference restricted to occurrences starting at or after L - window
                key, best = seq[-4:], -1
                for p in range(max(0, L - window), L - 3):
                    if seq[p:p + 4] == key:
                        best = p
                want = seq[best + 4:best + 4 + GAMMA] if best >= 0 and L >= 4 else []
            else:
                want = LookupIndex(4, seq).propose(GAMMA)
            got = toks[i, :int(avail[i])].tolist()
            assert got == want, (r, L, window, got, want)
            checked += 1
    print(f"matcher window={window or 'all'}: {checked} positions identical to the reference")


def fake_req(rid, row, prompt):
    return types.SimpleNamespace(rid=rid, kv=types.SimpleNamespace(req_pool_idx=row), origin_input_ids=prompt,
                                 output_ids=[], prefix_indices=[], extend_range=None)


def fake_batch(pool, reqs, rows, lens):
    return types.SimpleNamespace(req_to_token_pool=pool, reqs=reqs, has_grammar=False,
                                 req_pool_indices=torch.tensor(rows, dtype=torch.int64),
                                 seq_lens=torch.tensor(lens, dtype=torch.int64))


def run_engine(pool, rid, row, prompt, out):
    req = fake_req(rid, row, prompt)
    cut = len(prompt) // 2                      # chunked prefill in two pieces
    for start, end in ((0, cut), (cut, len(prompt))):
        req.prefix_indices = [0] * start
        req.extend_range = types.SimpleNamespace(length=end - start)
        ld.write_prefill(fake_batch(pool, [req], [row], [end]))
    L, anchor, pos, accepted = len(prompt), out[0], 1, []
    while pos < len(out):
        batch = fake_batch(pool, [req], [row], [L])
        prop = DraftProposal(draft_block_ids=torch.tensor([[anchor] + [0] * (GAMMA - 1)]),
                             draft_block=DraftBlockResult(torch.full((1, GAMMA), GARBAGE), None,
                                                          torch.ones(1, dtype=torch.bool), torch.ones(1)))
        drafts = ld.apply(prop, batch).draft_block.draft_tokens[0].tolist()
        acc = 0
        while acc < GAMMA and pos + acc < len(out) and drafts[acc] == out[pos + acc]:
            acc += 1
        commit = acc + 1
        ld.write_verify(torch.tensor([[anchor] + drafts]), batch.req_pool_indices, torch.tensor([commit]))
        committed = prompt + out[:pos + acc]            # anchor + accepted are now in the table
        L += commit
        assert L == len(committed)
        assert ld._S.table[row, :L].tolist() == committed, f"table diverged at L={L}"
        if pos + acc >= len(out):
            break
        anchor = out[pos + acc]
        pos += acc + 1
        accepted.append(acc)
    return accepted


def run_reference(prompt, out):
    idx = LookupIndex(4, prompt + [out[0]])
    pos, accepted = 1, []
    while pos < len(out):
        d = idx.propose(GAMMA)
        d = d if len(d) >= ld._MIN_TOKENS else []
        acc = 0
        while acc < len(d) and pos + acc < len(out) and d[acc] == out[pos + acc]:
            acc += 1
        if pos + acc >= len(out):
            break
        idx.extend(out[pos:pos + acc + 1])
        pos += acc + 1
        accepted.append(acc)
    return accepted


def check_loop(data):
    width = max(len(p) + len(o) for p, o in data) + 16
    pool = types.SimpleNamespace(req_to_token=torch.zeros(3, width, dtype=torch.int32))
    ld._S.__init__()
    for i, (prompt, out) in enumerate(data):
        row = (2, 0, 2, 1)[i % 4]                    # row 2 is reused by a later request
        got = run_engine(pool, f"req-{i}", row, prompt, out)
        want = run_reference(prompt, out)
        assert got == want, (i, len(got), len(want))
        steps = len(got)
        print(f"loop {i}: prompt {len(prompt):6d} output {len(out):5d}: {steps} steps, "
              f"{sum(got) / max(1, steps):.2f} lookup drafts accepted/step, identical to the reference")
    st = dict(zip(ld._STAT_NAMES, ld._S.stats.tolist()))
    assert st["rows"] > 0 and st["accepted_dspark"] == 0, st   # garbage DSpark drafts are never accepted
    print("stats:", st)


def run_engine_extend(pool, rid, row, prompt, out, gamma, base):
    """Extend mode: DSpark is an oracle for its first `base` tokens (garbage after), lookup fills the rest."""
    req = fake_req(rid, row, prompt)
    req.extend_range = types.SimpleNamespace(length=len(prompt))
    ld.write_prefill(fake_batch(pool, [req], [row], [len(prompt)]))
    L, anchor, pos, accepted = len(prompt), out[0], 1, []
    while pos < len(out):
        batch = fake_batch(pool, [req], [row], [L])
        oracle = (out[pos:pos + base] + [GARBAGE] * base)[:base]
        prop = DraftProposal(draft_block_ids=torch.tensor([[anchor] + [0] * (gamma - 1)]),
                             draft_block=DraftBlockResult(torch.tensor([oracle + [GARBAGE] * (gamma - base)]), None,
                                                          torch.ones(1, dtype=torch.bool), torch.ones(1)))
        drafts = ld.apply(prop, batch).draft_block.draft_tokens[0].tolist()
        assert drafts[:base] == oracle
        acc = 0
        while acc < gamma and pos + acc < len(out) and drafts[acc] == out[pos + acc]:
            acc += 1
        ld.write_verify(torch.tensor([[anchor] + drafts]), batch.req_pool_indices, torch.tensor([acc + 1]))
        L += acc + 1
        assert ld._S.table[row, :L].tolist() == prompt + out[:pos + acc]
        if pos + acc >= len(out):
            break
        anchor = out[pos + acc]
        pos += acc + 1
        accepted.append(acc)
    return accepted


def run_reference_extend(prompt, out, gamma, base):
    pos, accepted = 1, []
    while pos < len(out):
        committed = prompt + out[:pos - 1]
        L = len(committed)
        oracle = (out[pos:pos + base] + [GARBAGE] * base)[:base]
        tail = [out[pos - 1]] + oracle
        K = ld._EXT_KEY
        virt = committed + tail
        key = virt[-K:]
        best = -1
        for p in range(L - K, -1, -1):
            if committed[p:p + K] == key:
                best = p
                break
        ext = []
        if best >= 0:
            ext = virt[best + K:best + K + gamma - base]
        drafts = oracle + (ext if len(ext) >= ld._MIN_TOKENS else [])
        acc = 0
        while acc < len(drafts) and pos + acc < len(out) and drafts[acc] == out[pos + acc]:
            acc += 1
        if pos + acc >= len(out):
            break
        pos += acc + 1
        accepted.append(acc)
    return accepted


def check_extend(data):
    gamma, base = 15, 5
    width = max(len(p) + len(o) for p, o in data) + 32
    pool = types.SimpleNamespace(req_to_token=torch.zeros(3, width, dtype=torch.int32))
    ld._S.__init__()
    ld._MODE, ld._BASE = "extend", base
    try:
        for i, (prompt, out) in enumerate(data):
            row = (1, 2, 1)[i % 3]
            got = run_engine_extend(pool, f"ext-{i}", row, prompt, out, gamma, base)
            want = run_reference_extend(prompt, out, gamma, base)
            assert got == want, (i, len(got), len(want), next(j for j, (a, b) in enumerate(zip(got, want)) if a != b))
            beyond = sum(max(0, a - base) for a in got)
            print(f"extend {i}: {len(got)} steps, {sum(got) / max(1, len(got)):.2f} accepted/step "
                  f"({beyond} beyond the oracle's {base}), identical to the reference")
        st = dict(zip(ld._STAT_NAMES, ld._S.stats.tolist()))
        print("extend stats:", st)
    finally:
        ld._MODE = "replace"


def check_wide(data):
    """Wide mode: DSpark's 5 (oracle) + lookup, per-row verify lengths, graph tier, epilogue buffer, sampling patch."""
    gamma, base = 15, 5
    buckets = sorted({b * 6 + e for b in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16) for e in (0, 10, 20)})
    width = max(len(p) + len(o) for p, o in data) + 32
    pool = types.SimpleNamespace(req_to_token=torch.zeros(3, width, dtype=torch.int32))
    epi = types.SimpleNamespace(draft_tokens_buf=torch.zeros(16 * gamma, dtype=torch.int64))
    runner = types.SimpleNamespace(decode_cuda_graph_runner=types.SimpleNamespace(capture_num_tokens=buckets))
    ld._S.__init__()
    ld._MODE, ld._BASE = "wide", base
    ld._S.wide = {"gamma": gamma, "base": base, "epilogue": epi, "runner": runner, "budget": 20, "max_bs": 16}
    try:
        for i, (prompt, out) in enumerate(data[:6]):
            row = (1, 2)[i % 2]
            req = fake_req(f"wide-{i}", row, prompt)
            req.extend_range = types.SimpleNamespace(length=len(prompt))
            ld.write_prefill(fake_batch(pool, [req], [row], [len(prompt)]))
            L, anchor, pos, got = len(prompt), out[0], 1, []
            while pos < len(out):
                batch = fake_batch(pool, [req], [row], [L])
                oracle = (out[pos:pos + base] + [GARBAGE] * base)[:base]
                prop = DraftProposal(draft_block_ids=torch.tensor([[anchor] + [0] * (base - 1)]),
                                     draft_block=DraftBlockResult(torch.tensor([oracle]), None,
                                                                  torch.ones(1, dtype=torch.bool), torch.ones(1)))
                new = ld.apply_wide(prop, batch).draft_block.draft_tokens
                layout = ld._S.layout
                vlen = int(layout.verify_lens[0])
                assert new.shape == (1, gamma) and new[0, :base].tolist() == oracle
                assert epi.draft_tokens_buf[:gamma].tolist() == new[0].tolist()
                assert layout.graph_num_tokens == min(b for b in buckets if b >= vlen), (vlen, layout.graph_num_tokens)
                drafts = new[0, :vlen - 1].tolist()             # only the verified part of the block
                acc = 0
                while acc < len(drafts) and pos + acc < len(out) and drafts[acc] == out[pos + acc]:
                    acc += 1
                ld.write_verify(torch.tensor([[anchor] + new[0].tolist()]), batch.req_pool_indices,
                                torch.tensor([acc + 1]))
                L += acc + 1
                assert ld._S.table[row, :L].tolist() == prompt + out[:pos + acc]
                if pos + acc >= len(out):
                    break
                anchor = out[pos + acc]
                pos += acc + 1
                got.append(acc)
            want = run_reference_extend(prompt, out, gamma, base)
            assert got == want, (i, len(got), len(want))
            print(f"wide {i}: {len(got)} steps, {sum(got) / max(1, len(got)):.2f} accepted/step, identical to extend")
        # two rows at once: the extension budget caps the batch total, and sampled rows get one-hot tails
        prompt = list(range(100, 140)) * 3
        reqs = []
        for row in (1, 2):
            r = fake_req(f"pair-{row}", row, prompt)
            r.extend_range = types.SimpleNamespace(length=len(prompt))
            ld.write_prefill(fake_batch(pool, [r], [row], [len(prompt)]))
            reqs.append(r)
        ld._S.wide["budget"] = 12
        batch = fake_batch(pool, reqs, [1, 2], [len(prompt), len(prompt)])
        drafts = torch.tensor([[101, 102, 103, 104, 105]] * 2)
        logits = torch.randn(2, base, 300)
        prop = DraftProposal(draft_block_ids=torch.tensor([[100, 0, 0, 0, 0]] * 2),
                             draft_block=DraftBlockResult(drafts, logits.clone(), torch.zeros(2, dtype=torch.bool),
                                                          torch.ones(2)))
        res = ld.apply_wide(prop, batch).draft_block
        lens = ld._S.layout.verify_lens.tolist()
        assert lens == [16, 8], lens                            # 10 + 2 extension tokens = the budget of 12
        assert res.draft_tokens[0, base:].tolist() == list(range(106, 116)), res.draft_tokens[0]
        assert torch.equal(res.corrected_logits[:, :base], logits)
        probs = torch.softmax(res.corrected_logits[0, base:], dim=-1)
        assert all(probs[j, t] == 1.0 for j, t in enumerate(res.draft_tokens[0, base:].tolist()))
        print("wide batch: budget caps the extension (lens [16, 8]), sampled tails are one-hot, DSpark logits kept")
        # The packed verify ids SGLang's compact Triton gather builds from this proposal must match the strided
        # [anchor, drafts] rows: it reads row r's anchor at draft_block_ids[r * draft_tokens.shape[1]].
        from sglang.kernels.ops.speculative.dspark.dspark_verify_window import compact_verify_ids_triton
        out = ld.apply_wide(prop, batch)
        layout = ld._S.layout
        assert out.draft_block_ids.shape == (2, gamma), out.draft_block_ids.shape
        got = compact_verify_ids_triton(draft_block_ids=out.draft_block_ids, draft_tokens=out.draft_block.draft_tokens,
                                        layout=layout, device="cpu")
        rows = torch.cat([out.draft_block_ids[:, :1], out.draft_block.draft_tokens], dim=1)
        want = torch.cat([rows[r, :n] for r, n in enumerate(layout.verify_lens.tolist())])
        assert torch.equal(got[:want.numel()], want), (got, want)
        narrow = torch.tensor([[100, 0, 0, 0, 0], [200, 0, 0, 0, 0]])   # the pre-fix [bs, 5] ids: row 1's anchor is lost
        bad = compact_verify_ids_triton(draft_block_ids=narrow, draft_tokens=out.draft_block.draft_tokens,
                                        layout=layout, device="cpu")
        assert int(bad[int(layout.qo_indptr_device[1])]) != 200, "the narrow-ids reproduction no longer reproduces"
        print("wide batch: SGLang's compact verify-id gather yields the strided rows for every request "
              "(the pre-fix 5-wide ids give row 1 a stray anchor)")
    finally:
        ld._MODE, ld._S.wide = "replace", None


BUCKETS = sorted({b * 6 + e for b in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16) for e in (0, 10, 20)})


def gated_loop(pool, data, *, hold=32, split=True):
    """Wide decode loop (oracle DSpark block) with the sync gate on. Per step: (cold?, lens, graph tokens, demand)."""
    gamma, base = 15, 5
    epi = types.SimpleNamespace(draft_tokens_buf=torch.zeros(16 * gamma, dtype=torch.int64))
    runner = types.SimpleNamespace(decode_cuda_graph_runner=types.SimpleNamespace(capture_num_tokens=BUCKETS))
    ld._S.__init__()
    ld._MODE, ld._BASE = "wide", base
    ld._S.wide = {"gamma": gamma, "base": base, "epilogue": epi, "runner": runner, "budget": 20, "max_bs": 16,
                  "gate": True, "hold": hold, "override": False, "split_grid": split}
    syncs = {"n": 0}
    stock_host = ld._host_lens

    def counting(v):
        syncs["n"] += 1
        return stock_host(v)

    ld._host_lens = counting
    trace = []
    try:
        for i, (prompt, out) in enumerate(data):
            row = 1 + i % 2
            req = fake_req(f"gate-{i}", row, prompt)
            req.extend_range = types.SimpleNamespace(length=len(prompt))
            ld.write_prefill(fake_batch(pool, [req], [row], [len(prompt)]))
            L, anchor, pos = len(prompt), out[0], 1
            while pos < len(out):
                batch = fake_batch(pool, [req], [row], [L])
                oracle = (out[pos:pos + base] + [GARBAGE] * base)[:base]
                drafts = torch.tensor([oracle])
                _, avail = ld.lookup_extension(ld._S.table, batch.req_pool_indices, batch.seq_lens,
                                               torch.tensor([anchor]), drafts, base, 0, gamma - base)
                demand = bool((avail >= ld._MIN_TOKENS).any())
                before = syncs["n"]
                prop = DraftProposal(draft_block_ids=torch.tensor([[anchor] + [0] * (base - 1)]),
                                     draft_block=DraftBlockResult(drafts, None, torch.ones(1, dtype=torch.bool),
                                                                  torch.ones(1)))
                new = ld.apply_wide(prop, batch).draft_block.draft_tokens[0].tolist()
                lay = ld._S.layout
                cold = lay is ld._S.uniform.get((1, "cpu"))
                assert cold == (syncs["n"] == before), "a cold step synchronised or a hot step did not"
                lens = lay.verify_lens.tolist()
                trace.append((cold, lens, lay.graph_num_tokens, demand))
                vlen = lens[0]
                acc = 0
                while acc < vlen - 1 and pos + acc < len(out) and new[acc] == out[pos + acc]:
                    acc += 1
                ld.write_verify(torch.tensor([[anchor] + new]), batch.req_pool_indices, torch.tensor([acc + 1]))
                L += acc + 1
                assert ld._S.table[row, :L].tolist() == prompt + out[:pos + acc]
                if pos + acc >= len(out):
                    break
                anchor = out[pos + acc]
                pos += acc + 1
    finally:
        ld._host_lens = stock_host
        ld._MODE, ld._S.wide = "replace", None
    return trace


def check_gate(data):
    """Sync gate: cold steps never synchronise and reuse one cached layout; demand at step d opens the gate by d+2;
    hot steps use exactly the ungated lookup; runs are deterministic; every graph choice fits and is routed right."""
    width = max(len(p) + len(o) for p, o in data) + 64
    pool = types.SimpleNamespace(req_to_token=torch.zeros(3, width, dtype=torch.int32))
    rng = random.Random(3)
    fresh = [([rng.randrange(1_000, 90_000) for _ in range(400)], [rng.randrange(1_000, 90_000) for _ in range(300)])]
    t0 = gated_loop(pool, fresh)
    assert all(c for c, *_ in t0), "a no-repeat sequence took a synced step"
    t1 = gated_loop(pool, data[:6])
    t2 = gated_loop(pool, data[:6])
    assert t1 == t2, "gated runs are not deterministic"
    for i, (cold, lens, g, demand) in enumerate(t1):
        assert g in BUCKETS and sum(lens) <= g and max(1, min(g // 6, 16)) >= len(lens)
        assert (g % 6 == 0) == all(v == 6 for v in lens), (lens, g)
        if cold:
            assert lens == [6], lens
        if demand and i + 2 < len(t1):
            assert not t1[i + 2][0], f"demand at step {i} did not open the gate at step {i + 2}"
    cold_n = sum(c for c, *_ in t1)
    missed = sum(1 for c, _, _, dm in t1 if c and dm)
    print(f"sync gate: {len(t0)} no-repeat steps all sync-free; copy data {len(t1)} steps, {cold_n} cold, "
          f"{missed} extension steps missed by the 2-step lag, deterministic, routing and fit checked")


def check_narrow_slots():
    """Narrow capture slots: SGLang builds every capture layout, every live batch fits its graph, and the padded
    layout (torch and Triton) keeps rows within the 16-token window."""
    import lookup_wide as lw
    from sglang.srt.speculative.ragged_verify import (RaggedVerifyLayout, build_capture_verify_lens,
                                                     round_up_grid)
    for g in BUCKETS:
        slots = lw.narrow_capture_slots(g, 6, 16)
        lens = build_capture_verify_lens(num_tokens=g, num_slots=slots, num_draft_tokens=16)
        assert sum(lens) == g and max(lens) <= 16
        if g % 6 == 0:
            assert lens == [6] * slots, (g, lens)
    rng = random.Random(5)
    checked = 0
    for bs in range(1, 17):
        for _ in range(12):
            ext = [rng.choice([0, 0, 6, 7, 8, 9, 10]) for _ in range(bs)]
            left, capped = 20, []
            for e in ext:
                capped.append(min(e, left)); left -= capped[-1]
            vl = [6 + e for e in capped]
            uniform = all(v == 6 for v in vl)
            grid = [g for g in BUCKETS if (g % 6 == 0) == uniform]
            g = round_up_grid(sum(vl), grid)
            slots = lw.narrow_capture_slots(g, 6, 16)
            assert bs <= slots, (vl, g, slots)
            live = RaggedVerifyLayout.from_verify_lens(verify_lens_cpu=vl, device="cpu", grid=[g])
            padded = live.padded_to_bucket(padded_bs=slots, cap=16).verify_lens.tolist()
            assert padded[:bs] == vl or bs == slots, (vl, padded)
            assert max(padded) <= 16
            if uniform:
                assert padded == [6] * slots, (vl, padded)
            checked += 1
    print(f"narrow slots: {len(BUCKETS)} capture layouts build, {checked} live batches fit their graph and pad "
          f"uniformly where the graph is uniform")


def check_fused_predicate():
    import lookup_wide as lw
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
    layer = types.SimpleNamespace(compress_ratio=2, compressor=types.SimpleNamespace(use_fused_compress=True))

    def fb(lens, slots):
        lay = RaggedVerifyLayout.from_verify_lens(verify_lens_cpu=lens, device="cpu", grid=[sum(lens)])
        return types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_target_verify=lambda: True),
                                     batch_size=slots, spec_info=types.SimpleNamespace(ragged_verify_layout=lay))

    ok = lambda f, rows, **kw: lw.fused_c2_applies(forward_batch=f, rows=rows, layer=kw.get("layer", layer),
                                                   is_dspark_draft=kw.get("draft", False), narrow=6)
    assert ok(fb([6, 6], 2), 12)
    assert not ok(fb([6, 16], 2), 22)
    assert not ok(fb([1] * 12, 12), 12)                   # stock min(T,16) capture layout: not uniform
    assert not ok(fb([6, 6], 2), 12, draft=True)
    assert not ok(fb([6, 6], 2), 12, layer=types.SimpleNamespace(compress_ratio=1, compressor=layer.compressor))
    print("fused ratio-2 predicate: uniform 6b batches only (not ragged, not the stock capture layout, not the draft)")


def check_image_placeholders():
    """A continuation that runs into an image placeholder (prompt ids >= 1,000,000) stops before it, so no placeholder
    is ever drafted or used as a scatter index into the draft distribution (the 2026-09-29 production-1.7 crash: a
    sampled request whose lookup continued into a screenshot's placeholders hit a device-side assert)."""
    gamma, base, V = 15, 5, 300
    text = list(range(10, 40))                         # 30 tokens of text that the answer will repeat
    prompt = [5, 6, 7] + text + [1_000_000 + 829] * 12 + list(range(100, 130))
    width = len(prompt) + 64
    pool = types.SimpleNamespace(req_to_token=torch.zeros(2, width, dtype=torch.int32))
    epi = types.SimpleNamespace(draft_tokens_buf=torch.zeros(16 * gamma, dtype=torch.int64))
    runner = types.SimpleNamespace(decode_cuda_graph_runner=types.SimpleNamespace(capture_num_tokens=BUCKETS))
    ld._S.__init__()
    ld._MODE, ld._BASE = "wide", base
    ld._S.vocab = V
    ld._S.wide = {"gamma": gamma, "base": base, "epilogue": epi, "runner": runner, "budget": 20, "max_bs": 16}
    try:
        req = fake_req("img", 1, prompt)
        req.extend_range = types.SimpleNamespace(length=len(prompt))
        ld.write_prefill(fake_batch(pool, [req], [1], [len(prompt)]))
        # committed so far: the prompt + the first 22 tokens of `text` repeated; anchor = text[22]; DSpark drafts text[23:28]
        answer = text[:22]
        ld._S.table[1, len(prompt):len(prompt) + len(answer)] = torch.tensor(answer, dtype=torch.int32)
        L = len(prompt) + len(answer)
        batch = fake_batch(pool, [req], [1], [L])
        drafts = torch.tensor([text[23:28]])
        logits = torch.randn(1, base, V)
        prop = DraftProposal(draft_block_ids=torch.tensor([[text[22]] + [0] * (base - 1)]),
                             draft_block=DraftBlockResult(drafts, logits, torch.zeros(1, dtype=torch.bool), torch.ones(1)))
        out = ld.apply_wide(prop, batch).draft_block
        lens = ld._S.layout.verify_lens.tolist()
        ext = out.draft_tokens[0, base:lens[0] - 1].tolist()
        assert ext == text[28:30], (ext, lens)          # the two text tokens before the image, then it stops
        assert int(out.draft_tokens.max()) < V
        print(f"image placeholders: the continuation stops at the image ({len(ext)} text tokens drafted), "
              f"sampled one-hot stays inside the vocabulary")
    finally:
        ld._MODE, ld._S.wide, ld._S.vocab = "replace", None, None


def check_prefetch_hash():
    """engram_prefetch's CPU hash of a prompt chunk equals SGLang's EngramHasher (extend mode), image placeholders
    included (the prefetch thread used to index the token map with the raw placeholder ids and die)."""
    import engram_prefetch as ep
    from sglang.srt.layers.engram import MODE_EXTEND, EngramHasher
    from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
    g = torch.Generator().manual_seed(11)
    V, N, L, H, IMG = 600, 3, 2, 4, 555
    fake = types.SimpleNamespace(
        max_ngram_size=N, image_token_id=IMG, pad_id=7,
        token_map=torch.randint(0, 300, (V,), generator=g),
        multipliers=torch.randint(1, 1 << 20, (L, N), generator=g) * 2 + 1,
        primes=torch.tensor([[[1009, 1013, 1019, 1021], [1031, 1033, 1039, 1049]]] * L),
        offsets=torch.arange(L * (N - 1) * H).view(L, (N - 1) * H) * 5000)
    ep.HASHER.clear()
    ep.HASHER.update(pad_id=fake.pad_id, ngram=N, layer_ids=[0, 1], token_map=fake.token_map,
                     multipliers=fake.multipliers, primes=fake.primes, offsets=fake.offsets, obj=fake)
    ids = [int(x) for x in torch.randint(0, V, (40,), generator=g)]
    ids[10:16] = [MM_PAD_SHIFT_VALUE + 829107793 % 1000] * 6     # an image span
    ids[30] = MM_PAD_SHIFT_VALUE + 42
    got = ep._chunk_hashes(ids, 0, len(ids))
    t = torch.tensor(ids)
    want, _ = EngramHasher._torch_hash_ids(fake, t, torch.arange(len(ids)), MODE_EXTEND,
                                           torch.zeros(1, N - 1, dtype=torch.int64), len(ids), 1,
                                           torch.zeros(len(ids), dtype=torch.int64), torch.tensor([0]))
    assert torch.equal(got, want), (got - want).abs().max()
    part = ep._chunk_hashes(ids, 12, 25)
    assert torch.equal(part, want[12:25])
    print("engram prefetch: chunk hashes equal SGLang's hasher, image placeholders included")


def check_sampling_patch():
    torch.manual_seed(0)
    prompt = list(range(100, 140)) * 3
    pool = types.SimpleNamespace(req_to_token=torch.zeros(2, 512, dtype=torch.int32))
    ld._S.__init__()
    req = fake_req("s", 1, prompt)
    req.extend_range = types.SimpleNamespace(length=len(prompt))
    batch = fake_batch(pool, [req], [1], [len(prompt)])
    ld.write_prefill(batch)
    anchor = prompt[0]                              # 137 138 139 | 100 -> continuation 101 102 103 104 105
    batch = fake_batch(pool, [req, req], [1, 1], [len(prompt), 7])
    logits = torch.randn(2, GAMMA, 300)
    before = logits.clone()
    drafts = torch.randint(0, 300, (2, GAMMA))
    prop = DraftProposal(draft_block_ids=torch.tensor([[anchor] + [0] * 4, [55] + [0] * 4]),
                         draft_block=DraftBlockResult(drafts, logits, torch.zeros(2, dtype=torch.bool), torch.ones(2)))
    new = ld.apply(prop, batch).draft_block.draft_tokens
    assert new[0].tolist() == [101, 102, 103, 104, 105], new[0]
    assert torch.equal(new[1], drafts[1])           # no match in row 1: DSpark's block and logits untouched
    assert torch.equal(logits[1], before[1])
    probs = torch.softmax(logits[0], dim=-1)
    for i, t in enumerate(new[0].tolist()):
        assert probs[i, t] == 1.0 and probs[i].sum() == 1.0, (i, probs[i, t])
    print("sampling patch: replaced positions are one-hot on the lookup token, other rows untouched")


def main():
    data = samples(sys.argv[1] if len(sys.argv) > 1 else None)
    seqs = [p + o for p, o in data]
    check_matcher(seqs, 0)
    check_matcher(seqs, 2000)
    check_loop(data)
    check_extend(data)
    check_wide(data)
    check_gate(data)
    check_narrow_slots()
    check_fused_predicate()
    check_image_placeholders()
    check_prefetch_hash()
    check_sampling_patch()
    print("OK")


if __name__ == "__main__":
    main()
