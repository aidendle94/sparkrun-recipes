"""triton.language for the CPU stand-in: values are torch tensors (integers as int64 wrapped to their width, floats as
float32, BF16 rounded to BF16) with their Triton dtype.
MIT License, Copyright (c) 2026 Aiden Le."""
import torch

_PID = [0, 0, 0]
constexpr = int
static_range = range


class dtype:
    def __init__(self, name, bits, kind):
        self.name, self.bits, self.kind = name, bits, kind  # kind: u, i, f, b

    def __repr__(self):
        return self.name


int1 = dtype("int1", 1, "b")
uint8, uint16, uint32 = dtype("uint8", 8, "u"), dtype("uint16", 16, "u"), dtype("uint32", 32, "u")
int8, int16, int32, int64 = dtype("int8", 8, "i"), dtype("int16", 16, "i"), dtype("int32", 32, "i"), dtype("int64", 64, "i")
bfloat16, float32 = dtype("bfloat16", 16, "f"), dtype("float32", 32, "f")
_OF = {torch.bool: int1, torch.uint8: uint8, torch.int8: int8, torch.int16: int16, torch.int32: int32,
       torch.int64: int64, torch.bfloat16: bfloat16, torch.float32: float32}
_TORCH = {v: k for k, v in _OF.items()}


def _fit(t, ty):
    """A tensor as the stored form of dtype ty."""
    if ty.kind == "b":
        return t.bool()
    if ty.kind == "f":
        t = t.to(torch.float32)
        return t.to(torch.bfloat16).to(torch.float32) if ty is bfloat16 else t
    t = t.to(torch.int64)
    if ty.bits == 64:
        return t
    if ty.kind == "u":
        return t & ((1 << ty.bits) - 1)
    half = 1 << (ty.bits - 1)
    return ((t + half) & ((1 << ty.bits) - 1)) - half


def _join(a, b):
    """The dtype of an operation on dtypes a and b."""
    if a.kind == "f" or b.kind == "f":
        return float32 if float32 in (a, b) else bfloat16
    if a.kind == "b" and b.kind == "b":
        return int1
    if a.bits != b.bits:
        return a if a.bits > b.bits else b
    return a if a.kind == "u" else b


class V:
    def __init__(self, t, ty):
        self.t, self.dtype = _fit(torch.as_tensor(t), ty), ty

    @property
    def shape(self):
        return tuple(self.t.shape)

    def _pair(self, other):
        if isinstance(other, V):
            ty = _join(self.dtype, other.dtype)
            return self.t, other.t, ty
        if isinstance(other, float) and self.dtype.kind != "f":
            return self.t.float(), torch.tensor(other), float32
        return self.t, torch.as_tensor(other, dtype=self.t.dtype if self.dtype.kind != "b" else None), \
            (self.dtype if self.dtype.kind != "b" else int32)

    def _op(self, other, f, swap=False, compare=False):
        a, b, ty = self._pair(other)
        if ty.kind == "f":
            a, b = a.float(), b.float()
        elif ty.kind != "b":
            a, b = a.to(torch.int64), b.to(torch.int64)
        out = f(b, a) if swap else f(a, b)
        return V(out, int1 if compare else ty)

    def __add__(self, o): return self._op(o, torch.add)
    def __radd__(self, o): return self._op(o, torch.add, True)
    def __sub__(self, o): return self._op(o, torch.sub)
    def __rsub__(self, o): return self._op(o, torch.sub, True)
    def __mul__(self, o): return self._op(o, torch.mul)
    def __rmul__(self, o): return self._op(o, torch.mul, True)
    def __and__(self, o): return self._op(o, torch.bitwise_and)
    def __rand__(self, o): return self._op(o, torch.bitwise_and, True)
    def __or__(self, o): return self._op(o, torch.bitwise_or)
    def __ror__(self, o): return self._op(o, torch.bitwise_or, True)
    def __lshift__(self, o): return self._op(o, torch.bitwise_left_shift)
    def __rshift__(self, o): return self._op(o, torch.bitwise_right_shift)
    def __lt__(self, o): return self._op(o, torch.lt, compare=True)
    def __le__(self, o): return self._op(o, torch.le, compare=True)
    def __gt__(self, o): return self._op(o, torch.gt, compare=True)
    def __ge__(self, o): return self._op(o, torch.ge, compare=True)
    def __eq__(self, o): return self._op(o, torch.eq, compare=True)
    def __ne__(self, o): return self._op(o, torch.ne, compare=True)
    __hash__ = None

    def __getitem__(self, index):
        return V(self.t[index], self.dtype)

    def to(self, ty, bitcast=False):
        if not bitcast:
            return V(self.t, ty)
        assert self.dtype.kind in "ui" and ty.kind == "f" and self.dtype.bits == ty.bits, (self.dtype, ty)
        signed = {16: torch.int16, 32: torch.int32}[ty.bits]
        wrapped = _fit(self.t, int16 if ty.bits == 16 else int32).to(signed)
        return V(wrapped.view(_TORCH[ty]).to(torch.float32), ty)


class _PointerType:
    def __init__(self, element_ty):
        self.element_ty = element_ty


class P:
    """A tensor's storage from its first element on, and an offset in elements."""

    def __init__(self, flat, off=0):
        self.flat, self.off, self.dtype = flat, off, _PointerType(_OF[flat.dtype])

    def __add__(self, other):
        return P(self.flat, self.off + other)

    __radd__ = __add__


def pointer(t):
    n = t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()
    return P(torch.as_strided(t, (n,), (1,)))


def _offsets(p, mask):
    off = p.off.t if isinstance(p.off, V) else torch.as_tensor(p.off)
    assert not off.is_floating_point()
    if mask is None:
        mask = torch.ones((), dtype=torch.bool)
    else:
        mask = mask.t if isinstance(mask, V) else torch.as_tensor(mask)
    off, mask = torch.broadcast_tensors(off.to(torch.int64), mask.bool())
    used = off[mask]
    assert used.numel() == 0 or (int(used.min()) >= 0 and int(used.max()) < p.flat.numel()), \
        f"out of bounds: offsets {int(used.min())}..{int(used.max())} of {p.flat.numel()} elements"
    return off, mask


def program_id(axis):
    return _PID[axis]


def arange(start, end):
    return V(torch.arange(start, end), int32)


def zeros(shape, dtype):
    return V(torch.zeros(tuple(shape)), dtype)


def load(p, mask=None, other=None):
    off, mask = _offsets(p, mask)
    ty = p.dtype.element_ty
    got = p.flat[torch.where(mask, off, 0)]
    got = got.to(torch.float32) if ty.kind == "f" else got.to(torch.int64) if ty.kind != "b" else got
    if other is not None:
        got = torch.where(mask, got, torch.as_tensor(other, dtype=got.dtype))
    return V(got, ty)


def store(p, value, mask=None):
    off, mask = _offsets(p, mask)
    ty = p.dtype.element_ty
    t = V(value.t, ty).t if isinstance(value, V) else _fit(torch.as_tensor(value), ty)
    off, mask, t = torch.broadcast_tensors(off, mask, t)
    p.flat[off[mask]] = t[mask].to(p.flat.dtype)


def where(cond, a, b):
    if not isinstance(a, V):
        a = V(torch.as_tensor(a), b.dtype if isinstance(b, V) else int32)
    if not isinstance(b, V):
        b = V(torch.as_tensor(b), a.dtype)
    ty = _join(a.dtype, b.dtype)
    x, y = (a.t.float(), b.t.float()) if ty.kind == "f" else (a.t, b.t)
    return V(torch.where(cond.t, x, y), ty)


def trans(v):
    return V(v.t.T, v.dtype)


def dot(a, b, acc=None):
    assert a.dtype.kind == "f" and b.dtype.kind == "f" and a.shape[1] == b.shape[0], (a.dtype, b.dtype, a.shape, b.shape)
    out = a.t.float() @ b.t.float()
    return V(out if acc is None else acc.t + out, float32)
