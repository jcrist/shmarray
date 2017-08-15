import numpy as np

from .shmbuffer import shmbuffer


class shmarray(np.ndarray):
    __array_priority__ = -100.0

    def __new__(subtype, name, shape, dtype=np.uint8, mode='r+', offset=0,
                order='C'):
        assert offset == 0

        descr = np.dtype(dtype)
        dbytes = descr.itemsize

        if not isinstance(shape, tuple):
            shape = (shape,)
        size = 1
        for k in shape:
            size *= k
        nbytes = size * dbytes

        if mode != 'x+':
            buffer = shmbuffer(name, mode=mode)
            if buffer.nbytes != nbytes:
                raise ValueError("specified shape of %r doesn't match buffer "
                                 "of size %d" % (shape, buffer.nbytes))
        else:
            buffer = shmbuffer(name, nbytes=nbytes, mode=mode)

        self = np.ndarray.__new__(subtype, shape, dtype=descr,
                                  offset=offset, order=order)
        self._shmbuffer = buffer

        return self

    def __reduce__(self):
        name = self._shmbuffer.name
        order = 'C' if self.flags.carray else 'F'
        return (shmarray, (name, self.shape, self.dtype, 'r+', 0, order))

    def __array_finalize__(self, obj):
        if hasattr(obj, '_shmbuffer') and np.may_share_memory(self, obj):
            self._shmbuffer = obj._shmbuffer
        else:
            self._shmbuffer = None

    def __array_wrap__(self, arr, context=None):
        arr = super(shmarray, self).__array_wrap__(arr, context)

        # Return a shmarray if a shmarray was given as the output of the ufunc.
        # Leave the arr class unchanged if self is not a shmarray to keep
        # original shmarray subclasses behavior
        if self is arr or type(self) is not shmarray:
            return arr
        # Return scalar instead of 0d shmarray, e.g. for np.sum with
        # axis=None
        if arr.shape == ():
            return arr[()]
        # Return ndarray otherwise
        return arr.view(np.ndarray)

    def __getitem__(self, index):
        res = super(shmarray, self).__getitem__(index)
        if type(res) is shmarray and res._shmbuffer is None:
            return res.view(type=np.ndarray)
        return res
