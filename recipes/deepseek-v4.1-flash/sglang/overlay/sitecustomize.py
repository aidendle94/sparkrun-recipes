"""Install the Spark overlay's import hooks when the interpreter starts.

The image puts this directory on PYTHONPATH, so Python executes this module at start-up in every
process SGLang creates (launcher, tokenizer manager, scheduler, TP workers, spawned children) before
a single SGLang line runs. With SPARK_ENGRAM_DIR set it places one finder at the front of
`sys.meta_path` that watches the module names in HOOKS. The finder loads nothing itself: it asks the
finders behind it for the real spec and only wraps that spec's loader, so the module is executed by
its own loader exactly as stock, and the listed overlay functions then receive the live module
object and patch it in place. Because the patch happens inside the import itself, it fires exactly
once for the first execution, whichever spelling triggers it (`import a.b.c`, `from a.b import c`,
`importlib.import_module`), and before any importer can bind a stock name with `from ... import`.
An unrelated import costs one dictionary lookup.

Without SPARK_ENGRAM_DIR the overlay is inert and the same image serves stock SGLang. In both cases
the interpreter's own `sitecustomize` (Ubuntu's apport hook), which this file shadows on PYTHONPATH,
is still executed afterwards.

A hook that raises is never swallowed. The error is re-raised as OverlayHookError chained from the
original, so the full traceback reaches the log and the process dies instead of serving with a
half-installed overlay (a stock MXFP8 kernel is merely slow, but a missing Engram gather or an
unforced indexer plan answers with wrong tokens or crashes at graph capture). ImportError is
deliberately not preserved as the outer type, so `try: import ... except ImportError` blocks
upstream cannot mistake a broken hook for an optional dependency. Should the failed import be
retried, the hook is still armed and fails again rather than handing out the stock module.

Environment:
  SPARK_ENGRAM_DIR   master switch; unset or empty installs nothing

MIT License, Copyright (c) 2026 Aiden Le.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

# target module -> [(overlay module, function)] applied in order, each as fn(module).
# SPARK_ENGRAM_MODE selects the Engram row implementation: "hostnode" (default; engram_store: the rows are gathered by
# a host-function node inside the CUDA graph) or "staged" (engram_staged: the rows are staged before each forward
# from a pre-forward hook on the model runner; pure Python, no C library; slower on unique-text prefill).
_ENGRAM_STAGED = os.environ.get("SPARK_ENGRAM_MODE", "hostnode") == "staged"
_ENGRAM_IMPL = "engram_staged" if _ENGRAM_STAGED else "engram_store"
HOOKS: dict[str, list[tuple[str, str]]] = {
    "triton.compiler.compiler": [("kernel_load_guard", "install")],
    # lookup_wide first: it teaches the stock hasher ragged verify, the Engram store then wraps that forward
    "sglang.srt.layers.engram": [("lookup_wide", "install_engram"), (_ENGRAM_IMPL, "install"), ("step_timers", "install_engram")],
    "sglang.srt.layers.attention.dsv4.dsv41_sparse": [("lookup_wide", "install_dsv41_sparse")],
    "sglang.srt.layers.quantization.fp8_utils": [("mxfp8_kernel", "install")],
    "sglang.srt.layers.attention.dsv4.metadata": [("indexer_schedule", "install")],
    "sglang.kernels.ops.attention.flash_mla_sm120": [("sm120_prefill_pages", "install")],
    "sglang.srt.model_executor.model_runner": [
        ("prefill_flush", "install"),
        ("page_cache_release", "install"),
        ("late_tail", "install_runner"),
        ("roce_collectives", "install_health"),
    ] + ([("engram_staged", "install_runner")] if _ENGRAM_STAGED else []),
    "sglang.srt.distributed.parallel_state": [("roce_collectives", "install")],
    "sglang.srt.managers.scheduler": [("engram_prefetch", "install"), ("late_tail", "install_scheduler")],
    "sglang.srt.entrypoints.http_server": [("served_aliases", "install")],
    "sglang.srt.entrypoints.anthropic.serving": [("inline_system", "install")],
    "sglang.srt.managers.tokenizer_manager": [("request_guard", "install")],
    "sglang.srt.models.deepseek_v4": [("step_timers", "install_v4"), ("sm120_prefill_pages", "install_real_heads"),
                                      ("late_tail", "install_model"), ("wo_a_w8a16", "install")],
    "sglang.srt.layers.attention.deepseek_v4_backend": [("lookup_wide", "install_backend"), ("step_timers", "install_attn_backend"),
                                                         ("late_tail", "install_backend")],
    "sglang.srt.models.deepseek_v2": [("step_timers", "install_v2")],
    "sglang.srt.model_executor.model_runner_components.ngram_embedding_manager": [("lookup_draft", "install_manager")],
    "sglang.srt.speculative.dspark_components.dspark_draft": [("lookup_draft", "install_proposer")],
    "sglang.srt.speculative.spec_info": [("lookup_wide", "install_spec_info")],
    "sglang.srt.speculative.dspark_components.dspark_planner": [("lookup_wide", "install_planner")],
    "sglang.srt.speculative.dspark_components.dspark_worker_v2": [("lookup_wide", "install_worker")],
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner": [("lookup_wide", "install_graph_runner")],
}


class OverlayHookError(RuntimeError):
    """An overlay hook failed while patching a freshly executed SGLang module."""


def run_hooks(module, hooks: list[tuple[str, str]]) -> None:
    """Apply `hooks` to `module`; any failure is fatal for the importing process."""
    for overlay_name, fn_name in hooks:
        try:
            fn = getattr(importlib.import_module(overlay_name), fn_name)
            fn(module)
        except Exception as exc:
            logger.error("overlay hook %s.%s failed on %s: %r", overlay_name, fn_name, module.__name__, exc)
            raise OverlayHookError(
                f"overlay hook {overlay_name}.{fn_name} failed on {module.__name__}; "
                "the process cannot serve with a partly installed overlay"
            ) from exc
        logger.info("overlay hook %s.%s applied to %s", overlay_name, fn_name, module.__name__)


class _HookedLoader:
    """Stands in for the real loader; identical except that exec_module runs the hooks once."""

    def __init__(self, loader, fullname: str, hooks: list[tuple[str, str]], finder: "_HookFinder"):
        self._loader = loader
        self._fullname = fullname
        self._hooks = hooks
        self._finder = finder
        self._done = False

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module) -> None:
        self._loader.exec_module(module)
        if self._done:  # importlib.reload(): the spec says once, after the first execution
            return
        run_hooks(module, self._hooks)
        self._done = True
        self._finder.mark_done(self._fullname)

    def __getattr__(self, name):  # get_source, get_code, get_filename, is_package, ...
        return getattr(self._loader, name)

    def __repr__(self) -> str:
        return f"<overlay-hooked {self._loader!r}>"


class _HookFinder:
    """Front-of-meta_path finder (duck-typed; importlib.abc would cost ~15 ms of `typing` at every
    interpreter start): resolves the real spec through the other finders and wraps its loader."""

    spark_overlay = True

    def __init__(self, hooks: dict[str, list[tuple[str, str]]]):
        self._pending = {name: list(fns) for name, fns in hooks.items()}
        self._busy: set[str] = set()

    def mark_done(self, fullname: str) -> None:
        self._pending.pop(fullname, None)

    @property
    def pending(self) -> list[str]:
        return sorted(self._pending)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._pending or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None:
                    break
        finally:
            self._busy.discard(fullname)
        if spec is None:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            logger.warning("overlay: %s has no exec_module loader (%r); its hooks cannot run", fullname, spec.loader)
            return spec
        spec.loader = _HookedLoader(spec.loader, fullname, self._pending[fullname], self)
        return spec

    def invalidate_caches(self) -> None:
        return None


def install() -> _HookFinder | None:
    """Register the finder (idempotent) and patch any target that is somehow already imported."""
    for finder in sys.meta_path:
        if getattr(finder, "spark_overlay", False):
            return finder
    finder = _HookFinder(HOOKS)
    sys.meta_path.insert(0, finder)
    for name in list(HOOKS):
        module = sys.modules.get(name)
        if module is not None:
            run_hooks(module, HOOKS[name])
            finder.mark_done(name)
    logger.info("overlay: %d import hooks armed", len(HOOKS))
    sys.stderr.write(f"overlay: {len(HOOKS)} import hooks armed\n")
    return finder


def _run_shadowed_sitecustomize() -> None:
    """Execute the interpreter's own sitecustomize (the one this file hides on PYTHONPATH)."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        rest = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != here]
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
        if spec is None or spec.loader is None or spec.origin == __file__:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # the stock hook is a convenience; never fail start-up over it
        logger.debug("overlay: stock sitecustomize not run: %r", exc)


FINDER: _HookFinder | None = install() if os.environ.get("SPARK_ENGRAM_DIR") else None
_run_shadowed_sitecustomize()
