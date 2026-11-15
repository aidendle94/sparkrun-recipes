"""A stand-in for Triton that runs @triton.jit kernels on the CPU with torch, one program at a time (slow: for tiny
shapes). It checks what a kernel computes and that no load or store leaves its tensor; it knows nothing of layouts,
warps or speed. Put this directory's parent on PYTHONPATH ahead of the real package:
    PYTHONPATH=bench/emu python3 bench/dense_p12_test.py --emulate --patches patches
Only what the kernels of this workspace use is implemented (triton/language.py).
MIT License, Copyright (c) 2026 Aiden Le.
"""
import itertools

import torch

from . import language


class _Kernel:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        grid = tuple(grid) + (1,) * (3 - len(grid))

        def launch(*args, **options):
            args = [language.pointer(a) if isinstance(a, torch.Tensor) else a for a in args]
            for ids in itertools.product(*(range(g) for g in grid)):
                language._PID[:] = ids
                self.fn(*args)

        return launch


EMULATED = True  # patches/dense_p12.py then takes CPU tensors at its engine seams


def jit(fn=None, **options):
    return _Kernel(fn) if fn is not None else _Kernel


def cdiv(a, b):
    return -(-a // b)
