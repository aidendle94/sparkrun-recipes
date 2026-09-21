#!/usr/bin/env python3
"""CPU check that the overlay's import hooks bind to the SGLang modules (run inside the image, no GPU).

    docker run --rm -v <engram-dir>:/engram-local:ro -e SPARK_ENGRAM_DIR=/engram-local \
        --entrypoint python3 aidendle94/sparkrun-sglang-dsv41-gb10:production-1.1 \
        /opt/dsv41-spark/tests/test_hooks_cpu.py

The Engram directory only needs its manifest (or SPARK_ENGRAM_ROWS in the environment) and the table
headers; nothing is gathered here. Optional: SPARK_SERVED_ALIASES=a,b also checks the /v1/models
routes, SPARK_MXFP8_BACKEND and SPARK_PREFILL_FLUSH_TOKENS change what is expected of those hooks.
Every module in the hook table is imported, so a hook that raises fails the test; a module that
cannot be imported on a CPU-only box for another reason is reported and skipped.

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _row_count(span) -> int | None:
    """Rows of a table_span() result: a (path, rows, ...) tuple or an object/dict with a rows field."""
    if isinstance(span, (tuple, list)) and len(span) >= 2 and isinstance(span[1], int):
        return span[1]
    for name in ("rows", "num_rows", "n"):
        value = span.get(name) if isinstance(span, dict) else getattr(span, name, None)
        if isinstance(value, int):
            return value
    return None


def _in_wrapper_chain(fn, module_name: str) -> bool:
    """Whether `fn` or one of the callables it wraps (via __wrapped__) was defined in module_name."""
    seen = 0
    while fn is not None and seen < 16:
        if getattr(fn, "__module__", None) == module_name:
            return True
        fn = getattr(fn, "__wrapped__", None)
        seen += 1
    return False


def main() -> int:
    assert os.environ.get("SPARK_ENGRAM_DIR"), "set SPARK_ENGRAM_DIR"
    import sitecustomize as sc

    ok = True
    if getattr(sc, "FINDER", None) is None:
        print("sitecustomize did not arm the overlay (is the overlay directory first on PYTHONPATH?)")
        return 1

    import sglang.srt.layers.engram as eng
    import sglang.srt.layers.quantization.fp8_utils as fu
    import importlib
    impl_name = "engram_staged" if os.environ.get("SPARK_ENGRAM_MODE", "hostnode") == "staged" else "engram_store"
    engram_store = importlib.import_module(impl_name)  # the active Engram implementation (SPARK_ENGRAM_MODE)
    import mxfp8_kernel
    import prefill_flush

    init_mod = eng.EngramEmbedding.__init__.__module__
    rows_mod = eng.EngramEmbedding._owned_rows.__module__
    print("EngramEmbedding hooks:", init_mod, rows_mod)
    ok &= init_mod == impl_name and rows_mod == impl_name

    want = mxfp8_kernel.wanted_backend()
    patched = fu.flashinfer_mxfp8_blockscaled_linear.__module__ == "mxfp8_kernel"
    print("MXFP8 linear routed:", patched, "(backend", (want or "stock") + ")")
    ok &= patched if want is not None else not patched

    d = Path(os.environ["SPARK_ENGRAM_DIR"])
    ranges = engram_store.row_ranges(d)
    print("row ranges:", ranges)
    for layer, (lo, hi) in sorted(ranges.items()):
        span = engram_store.table_span(d, layer)
        rows = _row_count(span)
        print(f"layer {layer}: span={span} owned=[{lo},{hi})")
        if rows is None:
            print(f"layer {layer}: table_span() row count not recognised; range check FAILED")
            ok = False
        else:
            ok &= 0 <= lo <= hi <= rows

    import sglang.srt.layers.attention.dsv4.metadata as md
    post = md.PagedIndexerMetadata.__post_init__.__module__
    print("indexer schedule:", post, "(_IS_SM120", str(md._IS_SM120) + ")")
    ok &= (post == "indexer_schedule") == bool(md._IS_SM120)

    import sglang.srt.model_executor.model_runner as mr
    flushed = _in_wrapper_chain(mr.ModelRunner.forward, "prefill_flush")
    print("prefill flush wrapped:", flushed, "(threshold", str(prefill_flush.threshold_tokens()) + ")")
    ok &= flushed == (prefill_flush.threshold_tokens() > 0)

    if os.environ.get("SPARK_SERVED_ALIASES"):
        import sglang.srt.entrypoints.http_server as hs

        ep = {r.path: r.endpoint.__module__ for r in hs.app.routes if hasattr(r, "path")}
        print("model routes:", ep.get("/v1/models"), ep.get("/v1/models/{model:path}"))
        ok &= ep.get("/v1/models") == "served_aliases" and ep.get("/v1/models/{model:path}") == "served_aliases"

    import importlib

    for name in sc.HOOKS:
        try:
            importlib.import_module(name)
        except sc.OverlayHookError as exc:
            print(f"hook FAILED on {name}: {exc!r} <- {exc.__cause__!r}")
            ok = False
        except Exception as exc:  # noqa: BLE001 — a CPU-only box cannot import everything
            print(f"skipped {name}: import failed on this box ({type(exc).__name__}: {exc})")
    pending = sc.FINDER.pending
    print("hooks still pending:", pending or "none")

    print("HOOKS CPU TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
