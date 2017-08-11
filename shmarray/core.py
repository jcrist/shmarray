import ctypes.util
import os
import mmap
from numbers import Integral

import numpy as np


libc_path = ctypes.util.find_library('libc')
libc = ctypes.CDLL(libc_path)

shm_open = libc.shm_open
shm_open.argtypes = ctypes.c_char_p, ctypes.c_int, ctypes.c_int
shm_open.restype = ctypes.c_int

shm_unlink = libc.shm_unlink
shm_unlink.argtypes = (ctypes.c_char_p,)
shm_unlink.restype = ctypes.c_int

ftruncate = libc.ftruncate
ftruncate.argtypes = ctypes.c_int, ctypes.c_int
ftruncate.restype = ctypes.c_int


def cerror(msg):
    errno = ctypes.get_errno()
    raise ValueError("%s\n"
                     "errno: %d\n"
                     "msg: %s" % (msg, errno, os.strerror(errno)))


def unlink(name):
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    o = shm_unlink(name.encode())
    if o < 0:
        cerror("Failed to remove shm path: %r" % name)


class shmbuffer(mmap.mmap):
    def __new__(cls, name, nbytes, mode='r+', offset=0):
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        if mode == 'r+':
            flag = os.O_RDWR
            prot = mmap.PROT_READ | mmap.PROT_WRITE
            set_nbytes = False
        elif mode == 'r':
            flag = os.O_RDONLY
            prot = mmap.PROT_READ
            set_nbytes = False
        elif mode == 'x+':
            flag = os.O_RDWR | os.O_CREAT | os.O_EXCL
            prot = mmap.PROT_READ | mmap.PROT_WRITE
            mode = 'r+'
            set_nbytes = True
        else:
            raise ValueError("Unknown mode: %r" % mode)

        if not isinstance(nbytes, Integral) and nbytes > 0:
            raise ValueError("nbytes must be an integer > 0")
        nbytes = int(nbytes)

        fd = shm_open(name.encode(), flag, 0o666)
        if fd < 0:
            raise cerror("Failed to create shm file descriptor.")

        if set_nbytes:
            err = ftruncate(fd, nbytes)
            if err < 0:
                raise cerror("Failed to allocate nbytes %d for shm" % nbytes)

        try:
            obj = mmap.mmap.__new__(cls, fd, nbytes, mmap.MAP_SHARED, prot,
                                    offset=offset)
        except Exception:
            unlink(name)
            raise

        obj.fd = fd
        obj.name = name
        obj.nbytes = nbytes
        obj.mode = mode

        return obj

    def __reduce__(self):
        return (shmbuffer, (self.name, self.nbytes, self.mode))


class shmarray(np.ndarray):
    __array_priority__ = -100.0

    def __new__(subtype, name, shape, dtype=np.uint8, mode='r+', offset=0,
                order='C'):

        descr = np.dtype(dtype)
        dbytes = descr.itemsize

        if not isinstance(shape, tuple):
            shape = (shape,)
        size = 1
        for k in shape:
            size *= k

        nbytes = offset + size * dbytes
        start = offset - offset % mmap.ALLOCATIONGRANULARITY
        nbytes -= start
        array_offset = offset - start

        buffer = shmbuffer(name, nbytes, mode, offset=start)

        self = np.ndarray.__new__(subtype, shape, dtype=descr,
                                  buffer=buffer, offset=array_offset,
                                  order=order)
        self._shmbuffer = buffer
        self.offset = offset

        return self

    def __reduce__(self):
        name = self._shmbuffer.name
        mode = self._shmbuffer.mode
        order = 'C' if self.flags.carray else 'F'
        return (shmarray, (name, self.shape, self.dtype, mode,
                           self.offset, order))

    def __array_finalize__(self, obj):
        if hasattr(obj, '_shmbuffer') and np.may_share_memory(self, obj):
            self._shmbuffer = obj._shmbuffer
            self.offset = obj.offset
        else:
            self._shmbuffer = None
            self.offset = None

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
