#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <stdatomic.h>

#include <Python.h>
#include "structmember.h"


typedef enum
{
    MODE_READ,      /* 'r'  */
    MODE_READPLUS,  /* 'r+' */
    MODE_CREATE     /* 'x+' */
} mode_enum;


typedef struct {
    PyObject_HEAD
    char *data;
    char *name;
    size_t nbytes;
    int exports;
    mode_enum mode;
    PyObject *weakreflist;
} shmbuffer_object;


typedef struct {
    atomic_uint refcount;
    size_t nbytes;
} shmbuffer_header;


static void
shmbuffer_init(char *buffer, size_t nbytes) {
    shmbuffer_header *header = (shmbuffer_header *)buffer;
    header->refcount = ATOMIC_VAR_INIT(1);
    header->nbytes = nbytes;
}


static unsigned int
shmbuffer_incref(char *buffer) {
    shmbuffer_header *header = (shmbuffer_header *)buffer;
    return atomic_fetch_add(&(header->refcount), 1) + 1;
}


static unsigned int
shmbuffer_decref(char *buffer) {
    shmbuffer_header *header = (shmbuffer_header *)buffer;
    return atomic_fetch_sub(&(header->refcount), 1) - 1;
}


static unsigned int
shmbuffer_getref(char *buffer) {
    shmbuffer_header *header = (shmbuffer_header *)buffer;
    return atomic_load(&(header->refcount));
}


static size_t
shmbuffer_get_nbytes(char *buffer) {
    shmbuffer_header *header = (shmbuffer_header *)buffer;
    return header->nbytes;
}


static void
shmbuffer_object_dealloc(shmbuffer_object *self)
{
    if (self->data != NULL) {
        if (!shmbuffer_decref(self->data)) {
            shm_unlink(self->name);
        }
        munmap(self->data, self->nbytes + sizeof(shmbuffer_header));
    }

    if (self->weakreflist != NULL)
        PyObject_ClearWeakRefs((PyObject *) self);

    if (self->name)
        PyMem_Free(self->name);

    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *
shmbuffer_close_method(shmbuffer_object *self, PyObject *unused)
{
    if (self->exports > 0) {
        PyErr_SetString(PyExc_BufferError,
                        "cannot close exported pointers exist");
        return NULL;
    }

    if (self->data != NULL) {
        if (!shmbuffer_decref(self->data)) {
            shm_unlink(self->name);
        }
        munmap(self->data, self->nbytes + sizeof(shmbuffer_header));
        self->data = NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}


#define CHECK_VALID(err)                                                \
do {                                                                    \
    if (self->data == NULL) {                                           \
    PyErr_SetString(PyExc_ValueError, "shmbuffer closed or invalid");   \
    return err;                                                         \
    }                                                                   \
} while (0)


static PyObject *
shmbuffer_closed_get(shmbuffer_object *self)
{
    return PyBool_FromLong(self->data == NULL ? 1 : 0);
}


static PyObject *
shmbuffer_refcount_get(shmbuffer_object *self)
{
    CHECK_VALID(NULL);
    return PyLong_FromUnsignedLong(shmbuffer_getref(self->data));
}


static PyObject *
shmbuffer_nbytes_get(shmbuffer_object *self)
{
    CHECK_VALID(NULL);
    return PyLong_FromSize_t(self->nbytes);
}


static struct PyMethodDef shmbuffer_object_methods[] = {
    {"close",     (PyCFunction) shmbuffer_close_method,   METH_NOARGS},
    {NULL, NULL}  /* sentinel */
};


static PyGetSetDef shmbuffer_object_getset[] = {
    {"closed", (getter) shmbuffer_closed_get, NULL, NULL},
    {"refcount", (getter) shmbuffer_refcount_get, NULL, NULL},
    {"nbytes", (getter) shmbuffer_nbytes_get, NULL, NULL},
    {NULL}  /* sentinel */
};


static int
shmbuffer_buffer_getbuf(shmbuffer_object *self, Py_buffer *view, int flags)
{
    CHECK_VALID(-1);
    if (PyBuffer_FillInfo(view, (PyObject*)self,
                          self->data + sizeof(shmbuffer_header),
                          self->nbytes,
                          (self->mode == MODE_READ), flags) < 0)
        return -1;
    self->exports++;
    return 0;
}


static void
shmbuffer_buffer_releasebuf(shmbuffer_object *self, Py_buffer *view)
{
    self->exports--;
}


static PyBufferProcs shmbuffer_as_buffer = {
    (getbufferproc)shmbuffer_buffer_getbuf,
    (releasebufferproc)shmbuffer_buffer_releasebuf,
};


static Py_ssize_t
_get_nbytes(PyObject *o)
{
    if (o == NULL)
        return 0;

    if (!PyIndex_Check(o)) {
        PyErr_SetString(PyExc_TypeError,
                        "shared memory nbytes must be an integral value");
        return -1;
    }

    Py_ssize_t i = PyNumber_AsSsize_t(o, PyExc_OverflowError);
    if (i==-1 && PyErr_Occurred())
        return -1;

    if (i <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "shared memory nbytes must be positive");
        return -1;
    }
    return i;
}


static PyObject *
new_shmbuffer_object(PyTypeObject *type, PyObject *args, PyObject *kwdict)
{
    shmbuffer_object *self;
    char *nametemp = NULL, *name = NULL, *mode_str = "x+", *map_addr = NULL;
    Py_ssize_t nbytes, map_size;
    PyObject *nbytes_obj = NULL;
    int fd, flags, prot, mode;

    static char *keywords[] = {"name", "nbytes", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "s|Os", keywords,
                                     &nametemp, &nbytes_obj, &mode_str))
        return NULL;

    /* Determine nbytes*/
    nbytes = _get_nbytes(nbytes_obj);
    if (nbytes < 0)
        return NULL;

    /* Copy the tagname over */
    name = PyMem_Malloc(strlen(nametemp) + 1);
    if (name == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    strcpy(name, nametemp);

    /* Determine mode, prot, and flag from mode string */
    if (strcmp(mode_str, "r") == 0) {
        mode = MODE_READ;
        prot = PROT_READ;
        flags = O_RDONLY;
    }
    else if (strcmp(mode_str, "r+") == 0) {
        mode = MODE_READPLUS;
        prot = PROT_READ | PROT_WRITE;
        flags = O_RDWR;
    }
    else if (strcmp(mode_str, "x+") == 0) {
        mode = MODE_CREATE;
        prot = PROT_READ | PROT_WRITE;
        flags = O_RDWR | O_CREAT | O_EXCL;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "shmbuffer invalid mode parameter.");
        return NULL;
    }

    /* Validate mode and nbytes agree */
    if (mode == MODE_CREATE && nbytes == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "shmbuffer with mode='x+' requires "
                        "specified nbytes.");
        return NULL;
    } else if (mode != MODE_CREATE && nbytes != 0) {
        return PyErr_Format(PyExc_ValueError,
                            "shmbuffer cannot use 'nbytes' parameter with "
                            "mode='%s'", mode);
    }

    /* Open the file descriptor */
    fd = shm_open(name, flags, 0666);
    if (fd < 0)
        return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);

    if (mode == MODE_CREATE) {
        map_size = nbytes + sizeof(shmbuffer_header);
        if (ftruncate(fd, map_size) < 0) {
            close(fd);
            shm_unlink(name);
            return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);
        }
    } else {
        /* Initial mmap to find nbytes */
        map_addr = mmap(NULL, sizeof(shmbuffer_header),
                        prot, MAP_SHARED, fd, 0);
        if (map_addr == MAP_FAILED) {
            close(fd);
            shm_unlink(name);
            return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);
        }
        nbytes = shmbuffer_get_nbytes(map_addr);
        map_size = nbytes + sizeof(shmbuffer_header);
        munmap(map_addr, sizeof(shmbuffer_header));
    }

    /* Map it */
    map_addr = mmap(NULL, map_size, prot, MAP_SHARED, fd, 0);
    if (map_addr == MAP_FAILED) {
        close(fd);
        shm_unlink(name);
        return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);
    }

    /* Initialize if needed */
    if (mode == MODE_CREATE)
        shmbuffer_init(map_addr, nbytes);
    else
        shmbuffer_incref(map_addr);
    close(fd);

    self = (shmbuffer_object *)type->tp_alloc(type, 0);
    if (self == NULL) {
        shmbuffer_decref(map_addr);
        munmap(map_addr, map_size);
        shm_unlink(name);
        return NULL;
    }

    self->name = name;
    self->nbytes = (size_t) nbytes;
    self->weakreflist = NULL;
    self->exports = 0;
    self->data = map_addr;
    self->mode = (mode_enum)mode;

    return (PyObject *)self;
}

PyDoc_STRVAR(shmbuffer_doc, "shmbuffer(name, nbytes, mode='x+'");


static PyTypeObject shmbuffer_object_type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "shmbuffer.shmbuffer",                      /* tp_name */
    sizeof(shmbuffer_object),                   /* tp_size */
    0,                                          /* tp_itemsize */
    (destructor) shmbuffer_object_dealloc,      /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_reserved */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    PyObject_GenericGetAttr,                    /* tp_getattro */
    0,                                          /* tp_setattro */
    &shmbuffer_as_buffer,                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    shmbuffer_doc,                              /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    offsetof(shmbuffer_object, weakreflist),    /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    shmbuffer_object_methods,                   /* tp_methods */
    0,                                          /* tp_members */
    shmbuffer_object_getset,                    /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    PyType_GenericAlloc,                        /* tp_alloc */
    new_shmbuffer_object,                       /* tp_new */
    PyObject_Del,                               /* tp_free */
};


static struct PyModuleDef shmbuffermodule = {
    PyModuleDef_HEAD_INIT,
    "shmbuffer",
    NULL,
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit_shmbuffer(void)
{
    PyObject *dict, *module;

    if (PyType_Ready(&shmbuffer_object_type) < 0)
        return NULL;

    module = PyModule_Create(&shmbuffermodule);
    if (module == NULL)
        return NULL;
    dict = PyModule_GetDict(module);
    if (!dict)
        return NULL;

    PyDict_SetItemString(dict, "shmbuffer", (PyObject*) &shmbuffer_object_type);

    return module;
}
