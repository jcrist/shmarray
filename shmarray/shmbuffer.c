#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>

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
    size_t size;
    int exports;
    mode_enum mode;
    PyObject *weakreflist;
} shmbuffer_object;


static void
shmbuffer_object_dealloc(shmbuffer_object *self)
{
    if (self->data != NULL) {
        munmap(self->data, self->size);
        shm_unlink(self->name);
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
        munmap(self->data, self->size);
        shm_unlink(self->name);
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

static struct PyMethodDef shmbuffer_object_methods[] = {
    {"close",     (PyCFunction) shmbuffer_close_method,   METH_NOARGS},
    {NULL, NULL}  /* sentinel */
};

static PyGetSetDef shmbuffer_object_getset[] = {
    {"closed", (getter) shmbuffer_closed_get, NULL, NULL},
    {NULL}  /* sentinel */
};


static int
shmbuffer_buffer_getbuf(shmbuffer_object *self, Py_buffer *view, int flags)
{
    CHECK_VALID(-1);
    if (PyBuffer_FillInfo(view, (PyObject*)self, self->data, self->size,
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


static PyObject *
new_shmbuffer_object(PyTypeObject *type, PyObject *args, PyObject *kwdict)
{
    shmbuffer_object *self;
    char *nametemp = NULL, *name = NULL, *mode_str = "x+", *map_addr = NULL;
    Py_ssize_t map_size;
    int fd, flags, prot, mode;
    int resize;

    static char *keywords[] = {"name", "nbytes", "mode", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "sn|s", keywords,
                                     &nametemp, &map_size, &mode_str))
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
        resize = 0;
    }
    else if (strcmp(mode_str, "r+") == 0) {
        mode = MODE_READPLUS;
        prot = PROT_READ | PROT_WRITE;
        flags = O_RDWR;
        resize = 0;
    }
    else if (strcmp(mode_str, "x+") == 0) {
        mode = MODE_CREATE;
        prot = PROT_READ | PROT_WRITE;
        flags = O_RDWR | O_CREAT | O_EXCL;
        resize = 1;
    }
    else {
        return PyErr_Format(PyExc_ValueError,
                            "shmbuffer invalid mode parameter.");
    }

    /* Open the file descriptor */
    fd = shm_open(name, flags, 0666);
    if (fd < 0)
        return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);

    /* Grow the file if needed */
    if (resize) {
        if (ftruncate(fd, map_size) < 0) {
            close(fd);
            shm_unlink(name);
            return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);
        }
    }

    /* Map it */
    map_addr = mmap(NULL, map_size, prot, MAP_SHARED, fd, 0);
    close(fd);
    if (map_addr == MAP_FAILED) {
        shm_unlink(name);
        return PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);
    }

    self = (shmbuffer_object *)type->tp_alloc(type, 0);
    if (self == NULL) {
        munmap(map_addr, map_size);
        shm_unlink(name);
        return NULL;
    }

    self->name = name;
    self->size = (size_t) map_size;
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