"""Read torch.save files of plain containers and CPU tensors without torch (zip + pickle + raw little-endian
storages): the head has no torch outside the containers and no memory to start one beside production.
MIT License, Copyright (c) 2026 Aiden Le.
"""
import array, pickle, zipfile
CODES = {"FloatStorage": "f", "DoubleStorage": "d", "LongStorage": "q", "IntStorage": "i", "ShortStorage": "h",
         "BoolStorage": "b", "CharStorage": "b", "ByteStorage": "B"}


class Tensor:
    def __init__(self, flat, shape):
        self.flat, self.shape = flat, tuple(shape)

    def tolist(self):
        def build(off, dims):
            if not dims:
                return self.flat[off]
            step = 1
            for d in dims[1:]:
                step *= d
            if len(dims) == 1:
                return list(self.flat[off:off + dims[0]])
            return [build(off + i * step, dims[1:]) for i in range(dims[0])]
        return build(0, self.shape)


def load(path):
    z = zipfile.ZipFile(path)
    root = [n for n in z.namelist() if n.endswith("data.pkl")][0][:-len("data.pkl")]
    cache = {}

    def rebuild(storage, offset, size, stride, *rest):
        n, want = 1, []
        for d in reversed(size):
            want.insert(0, n)
            n *= d
        assert all(a == b for a, b, d in zip(stride, want, size) if d > 1), (size, stride)
        return Tensor(storage[offset:offset + n], size)

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name in CODES:
                return CODES[name]
            if module == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            raise pickle.UnpicklingError(f"{module}.{name}")

        def persistent_load(self, pid):
            _, code, key, _loc, _n = pid
            if key not in cache:
                a = array.array(code)
                a.frombytes(z.read(f"{root}data/{key}"))
                cache[key] = a
            return cache[key]

    return U(z.open(root + "data.pkl")).load()
