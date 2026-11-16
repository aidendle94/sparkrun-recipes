"""SM120 FA4 forward kernel: the far key blocks of a call with a relative bias (SPARK_FA4_BIAS_FAR=f1|f2, off by
default).

A prefill call of a global layer walks every key block of the prefix, and only the last 8 (1,024 keys) carry a
relative bias. Compiled with a bias, SGLang's kernel takes 41% longer for each of the other blocks than the same
kernel compiled without one (0.536 against 0.379 ms per 1,000 prefix keys per 4,096-row chunk on GB10, LSE
returned), although all it adds there is the multiply by the softmax scale, which has to come before the bias is
added. Two changes of the code's shape, neither of its arithmetic:

  f1  the far branch of apply_sheared_bias scales the accumulator with one whole-tensor store, the way the kernel's
      own rescale does, where stock walks its 64 values a thread through an unrolled loop of scalar accesses.
  f2  f1, and the loop over the blocks before the causal diagonal is split: the blocks that carry a bias (stock's
      body, stock's runtime test) and after them the far blocks, whose body is compiled without the bias state (a
      compile-time flag where stock tests `n_block >= n_block_max - num_bias_loads` on every block).

Both keep every operation and the order of the blocks: a far block is still acc * scale, 64 IEEE multiplies a
thread, then stock's mask and online softmax. bench/bias_far_window.py compares output and LSE bit for bit with
stock's on a GPU.

Measured (GB10, 2026-11-16): f1 gains nothing (0.542); f2 takes 0.482, 10% less per far key, with output and LSE
equal to stock's in every case of the tool; deleting the multiply altogether gives the same 0.484, so the rest of
the gap to the no-bias build is not the multiply. In a 4-node boot f2 saved 2% of a 120,000-token cold prefill
and 3.6% of a 300,000-token one.

The class is derived from the image's source by exact replacements (a changed image fails the boot, not the first
request) and bound where the SM120 host looks its forward kernel up; the decode subclass keeps the stock base. The
interface keeps compiled forward kernels in a persistent cache keyed by its own sources: a derived kernel under the
same key would be loaded by a later stock boot, so the forward cache moves to a directory named after the derived
source. The kernel is SGLang's (Apache-2.0), changed as described above.

MIT License, Copyright (c) 2026 Aiden Le.
"""
import hashlib
import inspect
import linecache
import os
import sys
import types

VARIANTS = ("f1", "f2")
MADE = [0]  # kernel objects made from the derived class

_FAR = ("        else:\n"
        "            for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):\n"
        "                acc_S[i] = acc_S[i] * base_softmax_scale\n")
_FAR_F1 = ("        else:\n"
           "            acc_S.store(acc_S.load() * base_softmax_scale)\n")
_LOOP = ("            for n_tile in cutlass.range(\n"
         "                n_block_upper - n_block_min_before_local_mask, unroll=1\n"
         "            ):\n")
_LOOP_END = "                )\n"
_BLOCK = "                    n_block_upper - n_tile - 1,\n"
_NEAR = ("                    apply_bias=(\n"
         "                        n_block_upper - n_tile - 1 >= n_block_max - num_bias_loads\n"
         "                    ),\n")
_DEF = "    def compute_one_n_block(\n"
_SIG = "        check_inf: cutlass.Constexpr = True,\n"
_CALL = ("        if const_expr(self.has_bias):\n"
         "            self.apply_sheared_bias(\n"
         "                acc_S,\n"
         "                mma_params.thr_mma_qk,\n"
         "                sBias,\n"
         "                pipeline_bias,\n"
         "                consumer_state,\n"
         "                base_softmax_scale,\n"
         "                apply_bias,\n"
         "            )\n")


def replace_once(src, old, new):
    if src.count(old) != 1:
        raise RuntimeError(f"fa4_bias_far: fa4_sm120/flash_fwd.py changed, `{old.strip()[:60]}` found "
                           f"{src.count(old)} times")
    return src.replace(old, new)


def _indent(text, by):
    return "".join(by + line if line.strip() else line for line in text.splitlines(True))


def _in_block_method(tail, old, new):
    """A replacement inside compute_one_n_block: `tail` is the source after its def line."""
    cut = tail.index("    @cute.jit\n")
    return replace_once(tail[:cut], old, new) + tail[cut:]


def _far_loop(src):
    start = src.index(_LOOP)
    body_at = start + len(_LOOP)
    end = src.index(_LOOP_END, body_at) + len(_LOOP_END)
    body = src[body_at:end]
    far = replace_once(replace_once(body, _BLOCK, "                    n_block_upper - spark_near - n_tile - 1,\n"),
                       _NEAR, "                    apply_bias=False,\n                    far_only=True,\n")
    # spark_near: how many of the loop's blocks, walked from the diagonal down, stock's test sends to the bias branch
    new = ("            if const_expr(self.has_bias):\n"
           "                spark_near = cutlass.max(\n"
           "                    cutlass.min(\n"
           "                        n_block_upper - (n_block_max - num_bias_loads),\n"
           "                        n_block_upper - n_block_min_before_local_mask,\n"
           "                    ),\n"
           "                    0,\n"
           "                )\n"
           "                for n_tile in cutlass.range(spark_near, unroll=1):\n"
           + _indent(body, "    ") +
           "                for n_tile in cutlass.range(\n"
           "                    n_block_upper - n_block_min_before_local_mask - spark_near, unroll=1\n"
           "                ):\n"
           + _indent(far, "    ") +
           "            else:\n"
           + _indent(_LOOP + body, "    "))
    src = src[:start] + new + src[end:]
    head, tail = src.split(_DEF)
    tail = _in_block_method(tail, _SIG, _SIG + "        far_only: cutlass.Constexpr = False,\n")
    tail = _in_block_method(tail, _CALL, (
        "        if const_expr(self.has_bias and far_only):\n"
        "            acc_S.store(acc_S.load() * base_softmax_scale)\n"
        "        elif const_expr(self.has_bias):\n" + _CALL.split("\n", 1)[1]))
    return head + _DEF + tail


def derived_source(variant, src):
    if variant not in VARIANTS:
        raise ValueError(f"SPARK_FA4_BIAS_FAR={variant!r}: one of {', '.join(VARIANTS)}")
    replace_once(src, _LOOP, _LOOP)
    replace_once(src, _DEF, _DEF)
    src = replace_once(src, _FAR, _FAR_F1)
    return _far_loop(src) if variant == "f2" else src


def derive_class(src, name):
    """The forward kernel class of a changed copy of flash_fwd's source, counting the objects made from it."""
    filename = f"<{name}: fa4_sm120/flash_fwd.py>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)  # the DSL reads the source back
    mod = types.ModuleType(name)
    mod.__file__ = filename
    sys.modules[name] = mod
    exec(compile(src, filename, "exec"), mod.__dict__)
    derived = mod.FlashAttentionForwardSm120
    stock_init = derived.__init__

    def counted(self, *args, **kwargs):
        MADE[0] += 1
        return stock_init(self, *args, **kwargs)

    derived.__init__ = counted
    return derived


def install(variant, interface=None):
    from sglang.kernels.jit.cute_aot_cache import get_jit_cache
    from sglang.kernels.ops.attention.fa4_sm120 import flash_fwd, runtime

    if interface is None:
        from sglang.kernels.ops.attention.flash_attn.cute import interface
    src = derived_source(variant, inspect.getsource(flash_fwd))
    digest = hashlib.sha256(src.encode()).hexdigest()[:12]
    runtime.FlashAttentionForwardSm120 = derive_class(src, f"spark_fa4_bias_far_{variant}")
    interface._flash_attn_fwd.compile_cache = get_jit_cache(
        f"fwd_spark_bias_far_{variant}_{digest}",
        source_paths=(os.path.dirname(os.path.abspath(interface.__file__)),), enable_tvm_ffi=True)
    print(f"[fa4_bias_far] {variant}: forward kernel derived ({digest}), own kernel cache", file=sys.stderr, flush=True)
