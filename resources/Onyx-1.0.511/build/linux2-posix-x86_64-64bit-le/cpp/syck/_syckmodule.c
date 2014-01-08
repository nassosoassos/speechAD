
#include <Python.h>
#include "syck.h"

/****************************************************************************
 * Python 2.2 compatibility.
 ****************************************************************************/

#ifndef PyDoc_STR
#define PyDoc_VAR(name)         static char name[]
#define PyDoc_STR(str)          (str)
#define PyDoc_STRVAR(name, str) PyDoc_VAR(name) = PyDoc_STR(str)
#endif

#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC  void
#endif

/****************************************************************************
 * Global objects: _syck.error, 'scalar', 'seq', 'map',
 * '1quote', '2quote', 'fold', 'literal', 'plain', '+', '-'.
 ****************************************************************************/

static PyObject *PySyck_Error;

static PyObject *PySyck_ScalarKind;
static PyObject *PySyck_SeqKind;
static PyObject *PySyck_MapKind;

static PyObject *PySyck_1QuoteStyle;
static PyObject *PySyck_2QuoteStyle;
static PyObject *PySyck_FoldStyle;
static PyObject *PySyck_LiteralStyle;
static PyObject *PySyck_PlainStyle;

static PyObject *PySyck_StripChomp;
static PyObject *PySyck_KeepChomp;

/****************************************************************************
 * The type _syck.Node.
 ****************************************************************************/

PyDoc_STRVAR(PySyckNode_doc,
    "_syck.Node() -> TypeError\n\n"
    "_syck.Node is an abstract type. It is the base type for _syck.Scalar,\n"
    "_syck.Seq, and _syck.Map. You cannot create an instance of _syck.Node\n"
    "directly. You may use _syck.Node for type checking or subclassing.\n");

typedef struct {
    PyObject_HEAD
    /* Common fields for all Node types: */
    PyObject *value;    /* always an object */
    PyObject *tag;      /* a string object or NULL */
    PyObject *anchor;   /* a string object or NULL */
} PySyckNodeObject;


static int
PySyckNode_clear(PySyckNodeObject *self)
{
    PyObject *tmp;

    tmp = self->value;
    self->value = NULL;
    Py_XDECREF(tmp);

    tmp = self->tag;
    self->tag = NULL;
    Py_XDECREF(tmp);

    tmp = self->anchor;
    self->value = NULL;
    Py_XDECREF(tmp);

    return 0;
}

static int
PySyckNode_traverse(PySyckNodeObject *self, visitproc visit, void *arg)
{
    int ret;

    if (self->value)
        if ((ret = visit(self->value, arg)) != 0)
            return ret;

    if (self->tag)
        if ((ret = visit(self->tag, arg)) != 0)
            return ret;

    if (self->anchor)
        if ((ret = visit(self->anchor, arg)) != 0)
            return ret;

    return 0;
}

static void
PySyckNode_dealloc(PySyckNodeObject *self)
{
    PySyckNode_clear(self);
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject *
PySyckNode_getkind(PySyckNodeObject *self, PyObject **closure)
{
    Py_INCREF(*closure);
    return *closure;
}

static PyObject *
PySyckNode_getvalue(PySyckNodeObject *self, void *closure)
{
    Py_INCREF(self->value);
    return self->value;
}

static PyObject *
PySyckNode_gettag(PySyckNodeObject *self, void *closure)
{
    PyObject *value = self->tag ? self->tag : Py_None;
    Py_INCREF(value);
    return value;
}

static int
PySyckNode_settag(PySyckNodeObject *self, PyObject *value, void *closure)
{
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'tag'");
        return -1;
    }

    if (value == Py_None) {
        Py_XDECREF(self->tag);
        self->tag = NULL;
        return 0;
    }

    if (!PyString_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'tag' must be a string");
        return -1;
    }

    Py_XDECREF(self->tag);
    Py_INCREF(value);
    self->tag = value;

    return 0;
}

static PyObject *
PySyckNode_getanchor(PySyckNodeObject *self, void *closure)
{
    PyObject *value = self->anchor ? self->anchor : Py_None;
    Py_INCREF(value);
    return value;
}

static int
PySyckNode_setanchor(PySyckNodeObject *self, PyObject *value, void *closure)
{
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'anchor'");
        return -1;
    }

    if (value == Py_None) {
        Py_XDECREF(self->anchor);
        self->anchor = NULL;
        return 0;
    }

    if (!PyString_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'anchor' must be a string");
        return -1;
    }

    Py_XDECREF(self->anchor);
    Py_INCREF(value);
    self->anchor = value;

    return 0;
}

static PyTypeObject PySyckNode_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
    "_syck.Node",                               /* tp_name */
    sizeof(PySyckNodeObject),                   /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)PySyckNode_dealloc,             /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,  /* tp_flags */
    PySyckNode_doc,                             /* tp_doc */
    (traverseproc)PySyckNode_traverse,          /* tp_traverse */
    (inquiry)PySyckNode_clear,                  /* tp_clear */
};

/****************************************************************************
 * The type _syck.Scalar.
 ****************************************************************************/

PyDoc_STRVAR(PySyckScalar_doc,
    "Scalar(value='', tag=None, style=None, indent=0, width=0, chomp=None)\n"
    "      -> a Scalar node\n\n"
    "_syck.Scalar represents a scalar node in Syck parser and emitter\n"
    "trees. A scalar node points to a single string value.\n");

typedef struct {
    PyObject_HEAD
    /* Common fields for all Node types: */
    PyObject *value;    /* always a string object */
    PyObject *tag;      /* a string object or NULL */
    PyObject *anchor;   /* a string object or NULL */
    /* Scalar-specific fields: */
    enum scalar_style style;
    int indent;
    int width;
    char chomp;
} PySyckScalarObject;

static PyObject *
PySyckScalar_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PySyckScalarObject *self;

    self = (PySyckScalarObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    self->value = PyString_FromString("");
    if (!self->value) {
        Py_DECREF(self);
        return NULL;
    }

    self->tag = NULL;
    self->anchor = NULL;
    self->style = scalar_none;
    self->indent = 0;
    self->width = 0;
    self->chomp = 0;

    return (PyObject *)self;
}

static int
PySyckScalar_setvalue(PySyckScalarObject *self, PyObject *value, void *closure)
{
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'value'");
        return -1;
    }
    if (!PyString_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'value' must be a string");
        return -1;
    }

    Py_DECREF(self->value);
    Py_INCREF(value);
    self->value = value;

    return 0;
}

static PyObject *
PySyckScalar_getstyle(PySyckScalarObject *self, void *closure)
{
    PyObject *value;

    switch (self->style) {
        case scalar_1quote: value = PySyck_1QuoteStyle; break;
        case scalar_2quote: value = PySyck_2QuoteStyle; break;
        case scalar_fold: value = PySyck_FoldStyle; break;
        case scalar_literal: value = PySyck_LiteralStyle; break;
        case scalar_plain: value = PySyck_PlainStyle; break;
        default: value = Py_None;
    }

    Py_INCREF(value);
    return value;
}

static int
PySyckScalar_setstyle(PySyckScalarObject *self, PyObject *value, void *closure)
{
    char *str;

    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'style'");
        return -1;
    }

    if (value == Py_None) {
        self->style = scalar_none;
        return 0;
    }

    if (!PyString_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'style' must be a string or None");
        return -1;
    }

    str = PyString_AsString(value);
    if (!str) return -1;

    if (strcmp(str, "1quote") == 0)
        self->style = scalar_1quote;
    else if (strcmp(str, "2quote") == 0)
        self->style = scalar_2quote;
    else if (strcmp(str, "fold") == 0)
        self->style = scalar_fold;
    else if (strcmp(str, "literal") == 0)
        self->style = scalar_literal;
    else if (strcmp(str, "plain") == 0)
        self->style = scalar_plain;
    else {
        PyErr_SetString(PyExc_ValueError, "unknown 'style'");
        return -1;
    }

    return 0;
}

static PyObject *
PySyckScalar_getindent(PySyckScalarObject *self, void *closure)
{
    return PyInt_FromLong(self->indent);
}

static int
PySyckScalar_setindent(PySyckScalarObject *self, PyObject *value, void *closure)
{
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'indent'");
        return -1;
    }

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'indent' must be an integer");
        return -1;
    }

    self->indent = PyInt_AS_LONG(value);

    return 0;
}

static PyObject *
PySyckScalar_getwidth(PySyckScalarObject *self, void *closure)
{
    return PyInt_FromLong(self->width);
}

static int
PySyckScalar_setwidth(PySyckScalarObject *self, PyObject *value, void *closure)
{
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'width'");
        return -1;
    }

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'width' must be an integer");
        return -1;
    }

    self->width = PyInt_AS_LONG(value);

    return 0;
}

static PyObject *
PySyckScalar_getchomp(PySyckScalarObject *self, void *closure)
{
    PyObject *value;

    switch (self->chomp) {
        case NL_CHOMP: value = PySyck_StripChomp; break;
        case NL_KEEP: value = PySyck_KeepChomp; break;
        default: value = Py_None;
    }

    Py_INCREF(value);
    return value;
}

static int
PySyckScalar_setchomp(PySyckScalarObject *self, PyObject *value, void *closure)
{
    char *str;

    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'chomp'");
        return -1;
    }

    if (value == Py_None) {
        self->chomp = 0;
        return 0;
    }

    if (!PyString_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'chomp' must be '+', '-', or None");
        return -1;
    }

    str = PyString_AsString(value);
    if (!str) return -1;

    if (strcmp(str, "-") == 0)
        self->chomp = NL_CHOMP;
    else if (strcmp(str, "+") == 0)
        self->chomp = NL_KEEP;
    else {
        PyErr_SetString(PyExc_TypeError, "'chomp' must be '+', '-', or None");
        return -1;
    }

    return 0;
}

static int
PySyckScalar_init(PySyckScalarObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *value = NULL;
    PyObject *tag = NULL;
    PyObject *anchor = NULL;
    PyObject *style = NULL;
    PyObject *indent = NULL;
    PyObject *width = NULL;
    PyObject *chomp = NULL;

    static char *kwdlist[] = {"value", "tag", "anchor",
        "style", "indent", "width", "chomp", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOOOO", kwdlist,
                &value, &tag, &anchor, &style, &indent, &width, &chomp))
        return -1;

    if (value && PySyckScalar_setvalue(self, value, NULL) < 0)
        return -1;

    if (tag && PySyckNode_settag((PySyckNodeObject *)self, tag, NULL) < 0)
        return -1;

    if (anchor && PySyckNode_setanchor((PySyckNodeObject *)self, anchor, NULL) < 0)
        return -1;

    if (style && PySyckScalar_setstyle(self, style, NULL) < 0)
        return -1;

    if (indent && PySyckScalar_setindent(self, indent, NULL) < 0)
        return -1;

    if (width && PySyckScalar_setwidth(self, width, NULL) < 0)
        return -1;

    if (chomp && PySyckScalar_setchomp(self, chomp, NULL) < 0)
        return -1;

    return 0;
}

static PyGetSetDef PySyckScalar_getsetters[] = {
    {"kind", (getter)PySyckNode_getkind, NULL,
        PyDoc_STR("the node kind, always 'scalar', read-only"),
        &PySyck_ScalarKind},
    {"value", (getter)PySyckNode_getvalue, (setter)PySyckScalar_setvalue,
        PyDoc_STR("the node value, a string"), NULL},
    {"tag", (getter)PySyckNode_gettag, (setter)PySyckNode_settag,
        PyDoc_STR("the node tag, a string or None"), NULL},
    {"anchor", (getter)PySyckNode_getanchor, (setter)PySyckNode_setanchor,
        PyDoc_STR("the node anchor, a string or None"), NULL},
    {"style", (getter)PySyckScalar_getstyle, (setter)PySyckScalar_setstyle,
        PyDoc_STR("the node style, values: None (means literal or plain),\n"
            "'1quote', '2quote', 'fold', 'literal', 'plain'"), NULL},
    {"indent", (getter)PySyckScalar_getindent, (setter)PySyckScalar_setindent,
        PyDoc_STR("the node indentation, an integer"), NULL},
    {"width", (getter)PySyckScalar_getwidth, (setter)PySyckScalar_setwidth,
        PyDoc_STR("the node width, an integer"), NULL},
    {"chomp", (getter)PySyckScalar_getchomp, (setter)PySyckScalar_setchomp,
        PyDoc_STR("the chomping method,\n"
            "values: None (clip), '-' (strip), or '+' (keep)"), NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PySyckScalar_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
    "_syck.Scalar",                             /* tp_name */
    sizeof(PySyckScalarObject),                 /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,     /* tp_flags */
    PySyckScalar_doc,                           /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    PySyckScalar_getsetters,                    /* tp_getset */
    &PySyckNode_Type,                           /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PySyckScalar_init,                /* tp_init */
    0,                                          /* tp_alloc */
    PySyckScalar_new,                           /* tp_new */
};

/****************************************************************************
 * The type _syck.Seq.
 ****************************************************************************/

PyDoc_STRVAR(PySyckSeq_doc,
    "Seq(value=[], tag=None, inline=False) -> a Seq node\n\n"
    "_syck.Seq represents a sequence node in Syck parser and emitter\n"
    "trees. A sequence node points to an ordered set of subnodes.\n");

typedef struct {
    PyObject_HEAD
    /* Common fields for all Node types: */
    PyObject *value;    /* always an object */
    PyObject *tag;      /* a string object or NULL */
    PyObject *anchor;   /* a string object or NULL */
    /* Seq-specific fields: */
    enum seq_style style;
} PySyckSeqObject;

static PyObject *
PySyckSeq_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PySyckSeqObject *self;

    self = (PySyckSeqObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    self->value = PyList_New(0);
    if (!self->value) {
        Py_DECREF(self);
        return NULL;
    }

    self->tag = NULL;
    self->anchor = NULL;
    self->style = seq_none;

    return (PyObject *)self;
}

static int
PySyckSeq_setvalue(PySyckSeqObject *self, PyObject *value, void *closure)
{
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'value'");
        return -1;
    }
    if (!PyList_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'value' must be a list");
        return -1;
    }

    Py_DECREF(self->value);
    Py_INCREF(value);
    self->value = value;

    return 0;
}

static PyObject *
PySyckSeq_getinline(PySyckSeqObject *self, void *closure)
{
    PyObject *value = (self->style == seq_inline) ? Py_True : Py_False;

    Py_INCREF(value);
    return value;
}

static int
PySyckSeq_setinline(PySyckSeqObject *self, PyObject *value, void *closure)
{
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'inline'");
        return -1;
    }

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'inline' must be a Boolean object");
        return -1;
    }

    self->style = PyInt_AS_LONG(value) ? seq_inline : seq_none;

    return 0;
}

static int
PySyckSeq_init(PySyckSeqObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *value = NULL;
    PyObject *tag = NULL;
    PyObject *anchor = NULL;
    PyObject *inline_ = NULL;

    static char *kwdlist[] = {"value", "tag", "anchor", "inline", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", kwdlist,
                &value, &tag, &anchor, &inline_))
        return -1;

    if (value && PySyckSeq_setvalue(self, value, NULL) < 0)
        return -1;

    if (tag && PySyckNode_settag((PySyckNodeObject *)self, tag, NULL) < 0)
        return -1;

    if (anchor && PySyckNode_setanchor((PySyckNodeObject *)self, anchor, NULL) < 0)
        return -1;

    if (inline_ && PySyckSeq_setinline(self, inline_, NULL) < 0)
        return -1;

    return 0;
}

static PyGetSetDef PySyckSeq_getsetters[] = {
    {"kind", (getter)PySyckNode_getkind, NULL,
        PyDoc_STR("the node kind, always 'seq', read-only"), &PySyck_SeqKind},
    {"value", (getter)PySyckNode_getvalue, (setter)PySyckSeq_setvalue,
        PyDoc_STR("the node value, a sequence"), NULL},
    {"tag", (getter)PySyckNode_gettag, (setter)PySyckNode_settag,
        PyDoc_STR("the node tag, a string or None"), NULL},
    {"anchor", (getter)PySyckNode_getanchor, (setter)PySyckNode_setanchor,
        PyDoc_STR("the node anchor, a string or None"), NULL},
    {"inline", (getter)PySyckSeq_getinline, (setter)PySyckSeq_setinline,
        PyDoc_STR("the block/flow flag"), NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PySyckSeq_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
    "_syck.Seq",                                /* tp_name */
    sizeof(PySyckSeqObject),                    /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,  /* tp_flags */
    PySyckSeq_doc,                              /* tp_doc */
    (traverseproc)PySyckNode_traverse,          /* tp_traverse */
    (inquiry)PySyckNode_clear,                  /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    PySyckSeq_getsetters,                       /* tp_getset */
    &PySyckNode_Type,                           /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PySyckSeq_init,                   /* tp_init */
    0,                                          /* tp_alloc */
    PySyckSeq_new,                              /* tp_new */
};

/****************************************************************************
 * The type _syck.Map.
 ****************************************************************************/

PyDoc_STRVAR(PySyckMap_doc,
    "Map(value={}, tag=None, inline=False) -> a Map node\n\n"
    "_syck.Map represents a mapping node in Syck parser and emitter\n"
    "trees. A mapping node points to an unordered collections of pairs.\n");

typedef struct {
    PyObject_HEAD
    /* Common fields for all Node types: */
    PyObject *value;    /* always an object */
    PyObject *tag;      /* a string object or NULL */
    PyObject *anchor;   /* a string object or NULL */
    /* Map-specific fields: */
    enum map_style style;
} PySyckMapObject;

static PyObject *
PySyckMap_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PySyckMapObject *self;

    self = (PySyckMapObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    self->value = PyDict_New();
    if (!self->value) {
        Py_DECREF(self);
        return NULL;
    }

    self->tag = NULL;
    self->anchor = NULL;
    self->style = seq_none;

    return (PyObject *)self;
}

static int
PySyckMap_setvalue(PySyckMapObject *self, PyObject *value, void *closure)
{
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'value'");
        return -1;
    }
    if (!PyDict_Check(value) && !PyList_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                "'value' must be a list of pairs or a dictionary");
        return -1;
    }

    Py_DECREF(self->value);
    Py_INCREF(value);
    self->value = value;

    return 0;
}

static PyObject *
PySyckMap_getinline(PySyckMapObject *self, void *closure)
{
    PyObject *value = (self->style == map_inline) ? Py_True : Py_False;

    Py_INCREF(value);
    return value;
}

static int
PySyckMap_setinline(PySyckMapObject *self, PyObject *value, void *closure)
{
    if (!value) {
        PyErr_SetString(PyExc_TypeError, "cannot delete 'inline'");
        return -1;
    }

    if (!PyInt_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "'inline' must be a Boolean object");
        return -1;
    }

    self->style = PyInt_AS_LONG(value) ? map_inline : map_none;

    return 0;
}

static int
PySyckMap_init(PySyckMapObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *value = NULL;
    PyObject *tag = NULL;
    PyObject *anchor = NULL;
    PyObject *inline_ = NULL;

    static char *kwdlist[] = {"value", "tag", "anchor", "inline", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", kwdlist,
                &value, &tag, &anchor, &inline_))
        return -1;

    if (value && PySyckMap_setvalue(self, value, NULL) < 0)
        return -1;

    if (tag && PySyckNode_settag((PySyckNodeObject *)self, tag, NULL) < 0)
        return -1;

    if (anchor && PySyckNode_setanchor((PySyckNodeObject *)self, anchor, NULL) < 0)
        return -1;

    if (inline_ && PySyckMap_setinline(self, inline_, NULL) < 0)
        return -1;

    return 0;
}

static PyGetSetDef PySyckMap_getsetters[] = {
    {"kind", (getter)PySyckNode_getkind, NULL,
        PyDoc_STR("the node kind, always 'map', read-only"), &PySyck_MapKind},
    {"value", (getter)PySyckNode_getvalue, (setter)PySyckMap_setvalue,
        PyDoc_STR("the node value, a list of pairs or a dictionary"), NULL},
    {"tag", (getter)PySyckNode_gettag, (setter)PySyckNode_settag,
        PyDoc_STR("the node tag, a string or None"), NULL},
    {"anchor", (getter)PySyckNode_getanchor, (setter)PySyckNode_setanchor,
        PyDoc_STR("the node anchor, a string or None"), NULL},
    {"inline", (getter)PySyckMap_getinline, (setter)PySyckMap_setinline,
        PyDoc_STR("the block/flow flag"), NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PySyckMap_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
    "_syck.Map",                                /* tp_name */
    sizeof(PySyckMapObject),                    /* tp_basicsize */
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,  /* tp_flags */
    PySyckMap_doc,                              /* tp_doc */
    (traverseproc)PySyckNode_traverse,          /* tp_traverse */
    (inquiry)PySyckNode_clear,                  /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    PySyckMap_getsetters,                       /* tp_getset */
    &PySyckNode_Type,                           /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PySyckMap_init,                   /* tp_init */
    0,                                          /* tp_alloc */
    PySyckMap_new,                              /* tp_new */
};

/****************************************************************************
 * The type _syck.Parser.
 ****************************************************************************/

PyDoc_STRVAR(PySyckParser_doc,
    "Parser(source, implicit_typing=True, taguri_expansion=True)\n"
    "      -> a Parser object\n\n"
    "_syck.Parser is a low-lever wrapper of the Syck parser. It parses\n"
    "a YAML stream and produces a tree of Nodes.\n");

typedef struct {
    PyObject_HEAD
    /* Attributes: */
    PyObject *source;       /* a string or file-like object */
    int implicit_typing;
    int taguri_expansion;
    /* Internal fields: */
    PyObject *symbols;      /* symbol table, a list, NULL outside parse() */
    SyckParser *parser;
    int parsing;
    int halt;
} PySyckParserObject;

static PyObject *
PySyckParser_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PySyckParserObject *self;

    self = (PySyckParserObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    self->source = NULL;
    self->implicit_typing = 0;
    self->taguri_expansion = 0;
    self->symbols = NULL;
    self->parser = NULL;
    self->parsing = 0;
    self->halt = 1;

    /*
    self->symbols = PyList_New(0);
    if (!self->symbols) {
        Py_DECREF(self);
        return NULL;
    }
    */

    return (PyObject *)self;
}

static int
PySyckParser_clear(PySyckParserObject *self)
{
    PyObject *tmp;

    if (self->parser) {
        syck_free_parser(self->parser);
        self->parser = NULL;
    }

    tmp = self->source;
    self->source = NULL;
    Py_XDECREF(tmp);

    tmp = self->symbols;
    self->symbols = NULL;
    Py_XDECREF(tmp);

    return 0;
}

static int
PySyckParser_traverse(PySyckParserObject *self, visitproc visit, void *arg)
{
    int ret;

    if (self->source)
        if ((ret = visit(self->source, arg)) != 0)
            return ret;

    if (self->symbols)
        if ((ret = visit(self->symbols, arg)) != 0)
            return ret;

    return 0;
}

static void
PySyckParser_dealloc(PySyckParserObject *self)
{
    PySyckParser_clear(self);
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject *
PySyckParser_getsource(PySyckParserObject *self, void *closure)
{
    PyObject *value = self->source ? self->source : Py_None;

    Py_INCREF(value);
    return value;
}

static PyObject *
PySyckParser_getimplicit_typing(PySyckParserObject *self, void *closure)
{
    PyObject *value = self->implicit_typing ? Py_True : Py_False;

    Py_INCREF(value);
    return value;
}

static PyObject *
PySyckParser_gettaguri_expansion(PySyckParserObject *self, void *closure)
{
    PyObject *value = self->taguri_expansion ? Py_True : Py_False;

    Py_INCREF(value);
    return value;
}

static PyObject *
PySyckParser_geteof(PySyckParserObject *self, void *closure)
{
    PyObject *value = self->halt ? Py_True : Py_False;

    Py_INCREF(value);
    return value;
}

static PyGetSetDef PySyckParser_getsetters[] = {
    {"source", (getter)PySyckParser_getsource, NULL,
        PyDoc_STR("IO source, a string or a file-like object"), NULL},
    {"implicit_typing", (getter)PySyckParser_getimplicit_typing, NULL,
        PyDoc_STR("implicit typing of builtin YAML types"), NULL},
    {"taguri_expansion", (getter)PySyckParser_gettaguri_expansion, NULL,
        PyDoc_STR("expansion of types in full taguri"), NULL},
    {"eof", (getter)PySyckParser_geteof, NULL,
        PyDoc_STR("EOF flag"), NULL},
    {NULL}  /* Sentinel */
};

static SYMID
PySyckParser_node_handler(SyckParser *parser, SyckNode *node)
{
    PyGILState_STATE gs;

    PySyckParserObject *self = (PySyckParserObject *)parser->bonus;

    SYMID index;
    PySyckNodeObject *object = NULL;

    PyObject *key, *value;
    int k;

    if (self->halt)
        return -1;

    gs = PyGILState_Ensure();

    switch (node->kind) {

        case syck_str_kind:
            object = (PySyckNodeObject *)
                PySyckScalar_new(&PySyckScalar_Type, NULL, NULL);
            if (!object) goto error;
            value = PyString_FromStringAndSize(node->data.str->ptr,
                    node->data.str->len);
            if (!value) goto error;
            Py_DECREF(object->value);
            object->value = value;
            break;

        case syck_seq_kind:
            object = (PySyckNodeObject *)
                PySyckSeq_new(&PySyckSeq_Type, NULL, NULL);
            if (!object) goto error;
            for (k = 0; k < node->data.list->idx; k++) {
                index = syck_seq_read(node, k)-1;
                value = PyList_GetItem(self->symbols, index);
                if (!value) goto error;
                if (PyList_Append(object->value, value) < 0)
                    goto error;
            }
            break;

        case syck_map_kind:
            object = (PySyckNodeObject *)
                PySyckMap_new(&PySyckMap_Type, NULL, NULL);
            if (!object) goto error;
            for (k = 0; k < node->data.pairs->idx; k++)
            {
                index = syck_map_read(node, map_key, k)-1;
                key = PyList_GetItem(self->symbols, index);
                if (!key) goto error;
                index = syck_map_read(node, map_value, k)-1;
                value = PyList_GetItem(self->symbols, index);
                if (!value) goto error;
                if (PyDict_SetItem(object->value, key, value) < 0)
                    goto error;
            }
            break;
    }

    if (node->type_id) {
        object->tag = PyString_FromString(node->type_id);
        if (!object->tag) goto error;
    }

    if (node->anchor) {
        object->anchor = PyString_FromString(node->anchor);
        if (!object->anchor) goto error;
    }

    if (PyList_Append(self->symbols, (PyObject *)object) < 0)
        goto error;

    Py_DECREF(object);

    index = PyList_GET_SIZE(self->symbols);
    PyGILState_Release(gs);
    return index;

error:
    Py_XDECREF(object);
    PyGILState_Release(gs);
    self->halt = 1;
    return -1;
}

static void
PySyckParser_error_handler(SyckParser *parser, char *str)
{
    PyGILState_STATE gs;

    PySyckParserObject *self = (PySyckParserObject *)parser->bonus;
    PyObject *value;

    if (self->halt) return;

    gs = PyGILState_Ensure();

    self->halt = 1;

    value = Py_BuildValue("(sii)", str,
            parser->linect, parser->cursor - parser->lineptr);
    if (value) {
        PyErr_SetObject(PySyck_Error, value);
    }

    PyGILState_Release(gs);
}

SyckNode *
PySyckParser_bad_anchor_handler(SyckParser *parser, char *anchor)
{
    PyGILState_STATE gs;

    PySyckParserObject *self = (PySyckParserObject *)parser->bonus;

    if (!self->halt) {
        gs = PyGILState_Ensure();

        self->halt = 1;
        PyErr_SetString(PyExc_TypeError, "recursive anchors are not implemented");

        PyGILState_Release(gs);
    }

    return syck_alloc_str();
}

static long
PySyckParser_read_handler(char *buf, SyckIoFile *file, long max_size, long skip)
{
    PyGILState_STATE gs;

    PySyckParserObject *self = (PySyckParserObject *)file->ptr;

    PyObject *value;

    char *str;
    int length;

    buf[skip] = '\0';

    if (self->halt) {
        return skip;
    }
    
    max_size -= skip;

    gs = PyGILState_Ensure();

    value = PyObject_CallMethod(self->source, "read", "(i)", max_size);
    if (!value) {
        self->halt = 1;

        PyGILState_Release(gs);

        return skip;
    }

    if (!PyString_CheckExact(value)) {
        Py_DECREF(value);
        PyErr_SetString(PyExc_TypeError, "file-like object should return a string");
        self->halt = 1;
        
        PyGILState_Release(gs);

        return skip;
    }

    str = PyString_AS_STRING(value);
    length = PyString_GET_SIZE(value);
    if (!length) {
        Py_DECREF(value);

        PyGILState_Release(gs);

        return skip;
    }

    if (length > max_size) {
        Py_DECREF(value);
        PyErr_SetString(PyExc_ValueError, "read returns an overly long string");
        self->halt = 1;

        PyGILState_Release(gs);

        return skip;
    }

    memcpy(buf+skip, str, length);
    length += skip;
    buf[length] = '\0';

    Py_DECREF(value);

    PyGILState_Release(gs);

    return length;
}

static int
PySyckParser_init(PySyckParserObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *source = NULL;
    int implicit_typing = 1;
    int taguri_expansion = 1;

    static char *kwdlist[] = {"source", "implicit_typing", "taguri_expansion",
        NULL};

    PySyckParser_clear(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ii", kwdlist,
                &source, &implicit_typing, &taguri_expansion))
        return -1;

    Py_INCREF(source);
    self->source = source;

    self->implicit_typing = implicit_typing;
    self->taguri_expansion = taguri_expansion;

    self->parser = syck_new_parser();
    self->parser->bonus = self;

    if (PyString_CheckExact(self->source)) {
        syck_parser_str(self->parser,
                PyString_AS_STRING(self->source),
                PyString_GET_SIZE(self->source), NULL);
    }
    /*
    else if (PyUnicode_CheckExact(self->source)) {
        syck_parser_str(self->parser,
                PyUnicode_AS_DATA(self->source),
                PyString_GET_DATA_SIZE(self->source), NULL);
    }
    */
    else {
        syck_parser_file(self->parser, (FILE *)self, PySyckParser_read_handler);
    }

    syck_parser_implicit_typing(self->parser, self->implicit_typing);
    syck_parser_taguri_expansion(self->parser, self->taguri_expansion);

    syck_parser_handler(self->parser, PySyckParser_node_handler);
    syck_parser_error_handler(self->parser, PySyckParser_error_handler);
    syck_parser_bad_anchor_handler(self->parser, PySyckParser_bad_anchor_handler);

    self->parsing = 0;
    self->halt = 0;

    return 0;
}

static PyObject *
PySyckParser_parse(PySyckParserObject *self)
{
    SYMID index;
    PyObject *value;

    if (self->parsing) {
        PyErr_SetString(PyExc_RuntimeError,
                "do not call Parser.parse while it is already running");
        return NULL;
    }

    if (self->halt) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    self->symbols = PyList_New(0);
    if (!self->symbols) {
        return NULL;
    }

    self->parsing = 1;
    Py_BEGIN_ALLOW_THREADS
    index = syck_parse(self->parser)-1;
    Py_END_ALLOW_THREADS
    self->parsing = 0;

    if (self->halt || self->parser->eof) {
        Py_DECREF(self->symbols);
        self->symbols = NULL;

        if (self->halt) return NULL;

        self->halt = 1;
        Py_INCREF(Py_None);
        return Py_None;
    }

    value = PyList_GetItem(self->symbols, index);
    Py_XINCREF(value);

    Py_DECREF(self->symbols);
    self->symbols = NULL;

    return value;
}

PyDoc_STRVAR(PySyckParser_parse_doc,
    "parse() -> the root Node object\n\n"
    "Parses the source and returns the root of the Node tree. Call it\n"
    "several times to retrieve all documents from the source. On EOF,\n"
    "returns None and sets the 'eof' attribute on.\n");

static PyMethodDef PySyckParser_methods[] = {
    {"parse",  (PyCFunction)PySyckParser_parse,
        METH_NOARGS, PySyckParser_parse_doc},
    {NULL}  /* Sentinel */
};

static PyTypeObject PySyckParser_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
    "_syck.Parser",                             /* tp_name */
    sizeof(PySyckParserObject),                 /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)PySyckParser_dealloc,           /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,  /* tp_flags */
    PySyckParser_doc,                           /* tp_doc */
    (traverseproc)PySyckParser_traverse,        /* tp_traverse */
    (inquiry)PySyckParser_clear,                /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    PySyckParser_methods,                       /* tp_methods */
    0,                                          /* tp_members */
    PySyckParser_getsetters,                    /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PySyckParser_init,                /* tp_init */
    0,                                          /* tp_alloc */
    PySyckParser_new,                           /* tp_new */
};

/****************************************************************************
 * The type _syck.Emitter.
 ****************************************************************************/

PyDoc_STRVAR(PySyckEmitter_doc,
    "Emitter(output, headless=False, use_header=False, use_version=False,\n"
    "        explicit_typing=True, style=None, best_width=80, indent=2)\n"
    "                -> an Emitter object\n\n"
    "_syck.Emitter is a low-lever wrapper of the Syck emitter. It emits\n"
    "a tree of Nodes into a YAML stream.\n");

typedef struct {
    PyObject_HEAD
    /* Attributes: */
    PyObject *output;       /* a file-like object */
    int headless;
    int use_header;
    int use_version;
    int explicit_typing;
    enum scalar_style style;
    int best_width;
    int indent;
    /* Internal fields: */
    PyObject *symbols;      /* symbol table, a list, NULL outside emit() */
    PyObject *nodes;        /* node -> symbol, a dict, NULL outside emit() */
    SyckEmitter *emitter;
    int emitting;
    int halt;
} PySyckEmitterObject;

static PyObject *
PySyckEmitter_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PySyckEmitterObject *self;

    self = (PySyckEmitterObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    self->output = NULL;
    self->headless = 0;
    self->use_header = 0;
    self->use_version = 0;
    self->explicit_typing = 0;
    self->style = scalar_none;
    self->best_width = 0;
    self->indent = 0;
    self->symbols = NULL;
    self->nodes = NULL;
    self->emitter = NULL;
    self->emitting = 0;
    self->halt = 1;

    return (PyObject *)self;
}

static int
PySyckEmitter_clear(PySyckEmitterObject *self)
{
    PyObject *tmp;

    if (self->emitter) {
        syck_free_emitter(self->emitter);
        self->emitter = NULL;
    }

    tmp = self->output;
    self->output = NULL;
    Py_XDECREF(tmp);

    tmp = self->symbols;
    self->symbols = NULL;
    Py_XDECREF(tmp);

    tmp = self->nodes;
    self->nodes = NULL;
    Py_XDECREF(tmp);

    return 0;
}

static int
PySyckEmitter_traverse(PySyckEmitterObject *self, visitproc visit, void *arg)
{
    int ret;

    if (self->output)
        if ((ret = visit(self->output, arg)) != 0)
            return ret;

    if (self->symbols)
        if ((ret = visit(self->symbols, arg)) != 0)
            return ret;

    if (self->nodes)
        if ((ret = visit(self->nodes, arg)) != 0)
            return ret;

    return 0;
}

static void
PySyckEmitter_dealloc(PySyckEmitterObject *self)
{
    PySyckEmitter_clear(self);
    self->ob_type->tp_free((PyObject *)self);
}

static PyObject *
PySyckEmitter_getoutput(PySyckEmitterObject *self, void *closure)
{
    PyObject *value = self->output ? self->output : Py_None;

    Py_INCREF(value);
    return value;
}

static PyObject *
PySyckEmitter_getheadless(PySyckEmitterObject *self, void *closure)
{
    PyObject *value = self->headless ? Py_True : Py_False;

    Py_INCREF(value);
    return value;
}

static PyObject *
PySyckEmitter_getuse_header(PySyckEmitterObject *self, void *closure)
{
    PyObject *value = self->use_header ? Py_True : Py_False;

    Py_INCREF(value);
    return value;
}

static PyObject *
PySyckEmitter_getuse_version(PySyckEmitterObject *self, void *closure)
{
    PyObject *value = self->use_version ? Py_True : Py_False;

    Py_INCREF(value);
    return value;
}

static PyObject *
PySyckEmitter_getexplicit_typing(PySyckEmitterObject *self, void *closure)
{
    PyObject *value = self->explicit_typing ? Py_True : Py_False;

    Py_INCREF(value);
    return value;
}

static PyObject *
PySyckEmitter_getstyle(PySyckEmitterObject *self, void *closure)
{
    PyObject *value;

    switch (self->style) {
        case scalar_1quote: value = PySyck_1QuoteStyle; break;
        case scalar_2quote: value = PySyck_2QuoteStyle; break;
        case scalar_fold: value = PySyck_FoldStyle; break;
        case scalar_literal: value = PySyck_LiteralStyle; break;
        case scalar_plain: value = PySyck_PlainStyle; break;
        default: value = Py_None;
    }

    Py_INCREF(value);
    return value;
}

static PyObject *
PySyckEmitter_getbest_width(PySyckEmitterObject *self, void *closure)
{
    return PyInt_FromLong(self->best_width);
}

static PyObject *
PySyckEmitter_getindent(PySyckEmitterObject *self, void *closure)
{
    return PyInt_FromLong(self->indent);
}

static PyGetSetDef PySyckEmitter_getsetters[] = {
    {"output", (getter)PySyckEmitter_getoutput, NULL,
        PyDoc_STR("output stream, a file-like object"), NULL},
    {"headless", (getter)PySyckEmitter_getheadless, NULL,
        PyDoc_STR("headerless document flag"), NULL},
    {"use_header", (getter)PySyckEmitter_getuse_header, NULL,
        PyDoc_STR("force header flag"), NULL},
    {"use_version", (getter)PySyckEmitter_getuse_version, NULL,
        PyDoc_STR("force version flag"), NULL},
    {"explicit_typing", (getter)PySyckEmitter_getexplicit_typing, NULL,
        PyDoc_STR("explicit typing for all collections"), NULL},
    {"style", (getter)PySyckEmitter_getstyle, NULL,
        PyDoc_STR("use literal or folded blocks on all text"), NULL},
    {"best_width", (getter)PySyckEmitter_getbest_width, NULL,
        PyDoc_STR("best width for folded scalars"), NULL},
    {"indent", (getter)PySyckEmitter_getindent, NULL,
        PyDoc_STR("default indentation"), NULL},
    {NULL}  /* Sentinel */
};

static void
PySyckEmitter_node_handler(SyckEmitter *emitter, st_data_t id)
{
    PyGILState_STATE gs;

    PySyckEmitterObject *self = (PySyckEmitterObject *)emitter->bonus;

    PySyckNodeObject *node;
    char *tag = NULL;
    PyObject *index;
    PyObject *key, *value, *item, *pair;
    int j, k, l;
    char *str;
    Py_ssize_t len;
    Py_ssize_t dict_pos;

    if (self->halt) return;

    gs = PyGILState_Ensure();

    node = (PySyckNodeObject *)PyList_GetItem(self->symbols, id);
    if (!node) {
        PyErr_SetString(PyExc_RuntimeError, "unknown data id");
        self->halt = 1;
        PyGILState_Release(gs);
        return;
    }

    if (node->tag) {
        tag = PyString_AsString(node->tag);
        if (!tag) {
            self->halt = 1;
            PyGILState_Release(gs);
            return;
        }
    }

    if (PyObject_TypeCheck((PyObject *)node, &PySyckSeq_Type)) {

        syck_emit_seq(emitter, tag, ((PySyckSeqObject *)node)->style);

        if (!PyList_Check(node->value)) {
            PyErr_SetString(PyExc_TypeError, "value of _syck.Seq must be a list");
            self->halt = 1;
            PyGILState_Release(gs);
            return;
        }
        l = PyList_GET_SIZE(node->value);
        for (k = 0; k < l; k ++) {
            item = PyList_GET_ITEM(node->value, k);
            if ((index = PyDict_GetItem(self->nodes, item))) {
                syck_emit_item(emitter, PyInt_AS_LONG(index));
                if (self->halt) {
                    PyGILState_Release(gs);
                    return;
                }
            }
            else {
                PyErr_SetString(PyExc_RuntimeError, "sequence item is not marked");
                self->halt = 1;
                PyGILState_Release(gs);
                return;
            }
        }
        syck_emit_end(emitter);
    }

    else if (PyObject_TypeCheck((PyObject *)node, &PySyckMap_Type)) {

        syck_emit_map(emitter, tag, ((PySyckMapObject *)node)->style);
        
        if (PyList_Check(node->value)) {
            l = PyList_GET_SIZE(node->value);
            for (k = 0; k < l; k ++) {
                pair = PyList_GET_ITEM(node->value, k);
                if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
                    PyErr_SetString(PyExc_TypeError,
                            "value of _syck.Map must be a list of pairs or a dictionary");
                    self->halt = 1;
                    PyGILState_Release(gs);
                    return;
                }
                for (j = 0; j < 2; j++) {
                    item = PyTuple_GET_ITEM(pair, j);
                    if ((index = PyDict_GetItem(self->nodes, item))) {
                        syck_emit_item(emitter, PyInt_AS_LONG(index));
                        if (self->halt) {
                            PyGILState_Release(gs);
                            return;
                        }
                    }
                    else {
                        PyErr_SetString(PyExc_RuntimeError, "mapping item is not marked");
                        self->halt = 1;
                        PyGILState_Release(gs);
                        return;
                    }
                }
            }
        }
        else if (PyDict_Check(node->value)) {
            dict_pos = 0;
            while (PyDict_Next(node->value, &dict_pos, &key, &value)) {
                for (j = 0; j < 2; j++) {
                    item = j ? value : key;
                    if ((index = PyDict_GetItem(self->nodes, item))) {
                        syck_emit_item(emitter, PyInt_AS_LONG(index));
                        if (self->halt) {
                            PyGILState_Release(gs);
                            return;
                        }
                    }
                    else {
                        PyErr_SetString(PyExc_RuntimeError, "mapping item is not marked");
                        self->halt = 1;
                        PyGILState_Release(gs);
                        return;
                    }
                }
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "value of _syck.Map must be a list of pairs or a dictionary");
            self->halt = 1;
            PyGILState_Release(gs);
            return;
        }

        syck_emit_end(emitter);
    }

    else if (PyObject_TypeCheck((PyObject *)node, &PySyckScalar_Type)) {
        if (PyString_AsStringAndSize(node->value, &str, &len) < 0) {
            self->halt = 1;
            PyGILState_Release(gs);
            return;
        }
        syck_emit_scalar(emitter, tag, ((PySyckScalarObject *)node)->style,
                ((PySyckScalarObject *)node)->indent,
                ((PySyckScalarObject *)node)->width,
                ((PySyckScalarObject *)node)->chomp, str, len);
    }

    else {
        PyErr_SetString(PyExc_TypeError, "Node instance is required");
        self->halt = 1;
        PyGILState_Release(gs);
        return;
    }   
    PyGILState_Release(gs);
}

static void
PySyckEmitter_write_handler(SyckEmitter *emitter, char *buf, long len)
{
    PyGILState_STATE gs;

    PySyckEmitterObject *self = (PySyckEmitterObject *)emitter->bonus;

    gs = PyGILState_Ensure();

    if (!PyObject_CallMethod(self->output, "write", "(s#)", buf, len))
        self->halt = 1;

    PyGILState_Release(gs);
}

static int
PySyckEmitter_init(PySyckEmitterObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *output = NULL;
    int headless = 0;
    int use_header = 0;
    int use_version = 0;
    int explicit_typing = 0;
    PyObject *style = NULL;
    int best_width = 80;
    int indent = 2;

    char *str;

    static char *kwdlist[] = {"output", "headless", "use_header",
        "use_version", "explicit_typing", "style",
        "best_width", "indent", NULL};

    PySyckEmitter_clear(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iiiiOii", kwdlist,
                &output, &headless, &use_header, &use_version,
                &explicit_typing, &style, &best_width, &indent))
        return -1;

    if (best_width <= 0) {
        PyErr_SetString(PyExc_ValueError, "'best_width' must be positive");
        return -1;
    }
    if (indent <= 0) {
        PyErr_SetString(PyExc_ValueError, "'indent' must be positive");
        return -1;
    }

    if (!style || style == Py_None) {
        self->style = scalar_none;
    }
    else {
        if (!PyString_Check(style)) {
            PyErr_SetString(PyExc_TypeError, "'style' must be a string or None");
            return -1;
        }

        str = PyString_AsString(style);
        if (!str) return -1;

        if (strcmp(str, "1quote") == 0)
            self->style = scalar_1quote;
        else if (strcmp(str, "2quote") == 0)
            self->style = scalar_2quote;
        else if (strcmp(str, "fold") == 0)
            self->style = scalar_fold;
        else if (strcmp(str, "literal") == 0)
            self->style = scalar_literal;
        else if (strcmp(str, "plain") == 0)
            self->style = scalar_plain;
        else {
            PyErr_SetString(PyExc_ValueError, "unknown 'style'");
            return -1;
        }
    }

    self->headless = headless;
    self->use_header = use_header;
    self->use_version = use_version;
    self->explicit_typing = explicit_typing;
    self->best_width = best_width;
    self->indent = indent;

    Py_INCREF(output);
    self->output = output;

    self->emitting = 0;
    self->halt = 0;

    return 0;
}

static int
PySyckEmitter_mark(PySyckEmitterObject *self, PyObject *root_node)
{
    int current, last;
    int j, k, l;
    PySyckNodeObject *node;
    PyObject *item, *key, *value, *pair;
    PyObject *index;
    Py_ssize_t dict_pos;

    last = 0;
    syck_emitter_mark_node(self->emitter, last);
    if (PyList_Append(self->symbols, root_node) < 0)
        return -1;
    index = PyInt_FromLong(last);
    if (!index) return -1;
    if (PyDict_SetItem(self->nodes, root_node, index) < 0) {
        Py_DECREF(index);
        return -1;
    }
    Py_DECREF(index);

    for (current = 0; current < PyList_GET_SIZE(self->symbols); current++) {

        node = (PySyckNodeObject *)PyList_GET_ITEM(self->symbols, current);

        if (PyObject_TypeCheck((PyObject *)node, &PySyckSeq_Type)) {
            if (!PyList_Check(node->value)) {
                PyErr_SetString(PyExc_TypeError, "value of _syck.Seq must be a list");
                return -1;
            }
            l = PyList_GET_SIZE(node->value);
            for (k = 0; k < l; k ++) {
                item = PyList_GET_ITEM(node->value, k);
                if ((index = PyDict_GetItem(self->nodes, item))) {
                    syck_emitter_mark_node(self->emitter, PyInt_AS_LONG(index));
                }
                else {
                    syck_emitter_mark_node(self->emitter, ++last);
                    if (PyList_Append(self->symbols, item) < 0)
                        return -1;
                    index = PyInt_FromLong(last);
                    if (!index) return -1;
                    if (PyDict_SetItem(self->nodes, item, index) < 0) {
                        Py_DECREF(index);
                        return -1;
                    }
                    Py_DECREF(index);
                }
            }
        }

        else if (PyObject_TypeCheck((PyObject *)node, &PySyckMap_Type)) {
            
            if (PyList_Check(node->value)) {
                l = PyList_GET_SIZE(node->value);
                for (k = 0; k < l; k ++) {
                    pair = PyList_GET_ITEM(node->value, k);
                    if (!PyTuple_Check(pair) || PyTuple_GET_SIZE(pair) != 2) {
                        PyErr_SetString(PyExc_TypeError,
                                "value of _syck.Map must be a list of pairs or a dictionary");
                        return -1;
                    }
                    for (j = 0; j < 2; j++) {
                        item = PyTuple_GET_ITEM(pair, j);
                        if ((index = PyDict_GetItem(self->nodes, item))) {
                            syck_emitter_mark_node(self->emitter, PyInt_AS_LONG(index));
                        }
                        else {
                            syck_emitter_mark_node(self->emitter, ++last);
                            if (PyList_Append(self->symbols, item) < 0)
                                return -1;
                            index = PyInt_FromLong(last);
                            if (!index) return -1;
                            if (PyDict_SetItem(self->nodes, item, index) < 0) {
                                Py_DECREF(index);
                                return -1;
                            }
                            Py_DECREF(index);
                        }
                    }
                }
                
            }
            else if (PyDict_Check(node->value)) {
                dict_pos = 0;
                while (PyDict_Next(node->value, &dict_pos, &key, &value)) {
                    for (j = 0; j < 2; j++) {
                        item = j ? value : key;
                        if ((index = PyDict_GetItem(self->nodes, item))) {
                            syck_emitter_mark_node(self->emitter, PyInt_AS_LONG(index));
                        }
                        else {
                            syck_emitter_mark_node(self->emitter, ++last);
                            if (PyList_Append(self->symbols, item) < 0)
                                return -1;
                            index = PyInt_FromLong(last);
                            if (!index) return -1;
                            if (PyDict_SetItem(self->nodes, item, index) < 0) {
                                Py_DECREF(index);
                                return -1;
                            }
                            Py_DECREF(index);
                        }
                    }
                }
            }
            else {
                PyErr_SetString(PyExc_TypeError,
                        "value of _syck.Map must be a list of pairs or a dictionary");
                return -1;
            }
        }

        else if (!PyObject_TypeCheck((PyObject *)node, &PySyckScalar_Type)) {
            PyErr_SetString(PyExc_TypeError, "Node instance is required");
            return -1;
        }   
    }
    return 0;
}

static PyObject *
PySyckEmitter_emit(PySyckEmitterObject *self, PyObject *args)
{
    PyObject *node;

    if (self->emitting) {
        PyErr_SetString(PyExc_RuntimeError, "do not call Emitter.emit while it is already emitting");
        return NULL;
    }

    if (self->halt) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (!PyArg_ParseTuple(args, "O", &node))
        return NULL;

    self->emitting = 1;


    self->symbols = PyList_New(0);
    if (!self->symbols) {
        return NULL;
    }
    self->nodes = PyDict_New();
    if (!self->nodes) {
        Py_DECREF(self->symbols);
        self->symbols = NULL;
        return NULL;
    }

    self->emitter = syck_new_emitter();
    self->emitter->bonus = self;
    self->emitter->headless = self->headless;
    self->emitter->use_header = self->use_header;
    self->emitter->use_version = self->use_version;
    self->emitter->explicit_typing = self->explicit_typing;
    self->emitter->style = self->style;
    self->emitter->best_width = self->best_width;
    self->emitter->indent = self->indent;

    syck_emitter_handler(self->emitter, PySyckEmitter_node_handler);
    syck_output_handler(self->emitter, PySyckEmitter_write_handler);

    if (PySyckEmitter_mark(self, node) < 0) {
        Py_DECREF(self->symbols);
        self->symbols = NULL;
        Py_DECREF(self->nodes);
        self->nodes = NULL;
        self->emitting = 0;
        self->halt = 1;
        syck_free_emitter(self->emitter);
        self->emitter = NULL;
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    syck_emit(self->emitter, 0);
    syck_emitter_flush(self->emitter, 0);
    Py_END_ALLOW_THREADS

    syck_free_emitter(self->emitter);
    self->emitter = NULL;

    self->emitting = 0;

    Py_DECREF(self->symbols);
    self->symbols = NULL;
    Py_DECREF(self->nodes);
    self->nodes = NULL;

    if (self->halt) return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}

PyDoc_STRVAR(PySyckEmitter_emit_doc,
    "emit(root_node) -> None\n\n"
    "Emits the Node tree to the output.\n");

static PyMethodDef PySyckEmitter_methods[] = {
    {"emit",  (PyCFunction)PySyckEmitter_emit,
        METH_VARARGS, PySyckEmitter_emit_doc},
    {NULL}  /* Sentinel */
};

static PyTypeObject PySyckEmitter_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
    "_syck.Emitter",                            /* tp_name */
    sizeof(PySyckEmitterObject),                /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor)PySyckEmitter_dealloc,          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE|Py_TPFLAGS_HAVE_GC,  /* tp_flags */
    PySyckEmitter_doc,                          /* tp_doc */
    (traverseproc)PySyckEmitter_traverse,       /* tp_traverse */
    (inquiry)PySyckEmitter_clear,               /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    PySyckEmitter_methods,                      /* tp_methods */
    0,                                          /* tp_members */
    PySyckEmitter_getsetters,                   /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PySyckEmitter_init,               /* tp_init */
    0,                                          /* tp_alloc */
    PySyckEmitter_new,                          /* tp_new */
};

/****************************************************************************
 * The module _syck.
 ****************************************************************************/

static PyMethodDef PySyck_methods[] = {
    {NULL}  /* Sentinel */
};

PyDoc_STRVAR(PySyck_doc,
    "_syck is a low-level wrapper for the Syck YAML parser and emitter.\n"
    "Do not use it directly, use the module 'syck' instead.\n");

static int
add_slotnames(PyTypeObject *type)
{
    PyObject *slotnames;
    PyObject *name;
    PyGetSetDef *getsetter;

    if (!type->tp_getset) return 0;
    if (!type->tp_dict) return 0;

    slotnames = PyList_New(0);
    if (!slotnames) return -1;

    for (getsetter = type->tp_getset; getsetter->name; getsetter++) {
        if (!getsetter->set) continue;
        name = PyString_FromString(getsetter->name);
        if (!name) {
           Py_DECREF(slotnames);
           return -1;
        }
        if (PyList_Append(slotnames, name) < 0) {
            Py_DECREF(name);
            Py_DECREF(slotnames);
            return -1;
        }
        Py_DECREF(name);
    }

    if (PyDict_SetItemString(type->tp_dict, "__slotnames__", slotnames) < 0) {
        Py_DECREF(slotnames);
        return -1;
    }

    Py_DECREF(slotnames);
    return 0;
}

PyMODINIT_FUNC
init_syck(void)
{
    PyObject *m;

    PyEval_InitThreads();   /* Fix segfault for Python 2.3 */

    if (PyType_Ready(&PySyckNode_Type) < 0)
        return;
    if (PyType_Ready(&PySyckScalar_Type) < 0)
        return;
    if (add_slotnames(&PySyckScalar_Type) < 0)
        return;
    if (PyType_Ready(&PySyckSeq_Type) < 0)
        return;
    if (add_slotnames(&PySyckSeq_Type) < 0)
        return;
    if (PyType_Ready(&PySyckMap_Type) < 0)
        return;
    if (add_slotnames(&PySyckMap_Type) < 0)
        return;
    if (PyType_Ready(&PySyckParser_Type) < 0)
        return;
    if (PyType_Ready(&PySyckEmitter_Type) < 0)
        return;
    
    PySyck_Error = PyErr_NewException("_syck.error", NULL, NULL);
    if (!PySyck_Error) return;

    PySyck_ScalarKind = PyString_FromString("scalar");
    if (!PySyck_ScalarKind) return;
    PySyck_SeqKind = PyString_FromString("seq");
    if (!PySyck_SeqKind) return;
    PySyck_MapKind = PyString_FromString("map");
    if (!PySyck_MapKind) return;

    PySyck_1QuoteStyle = PyString_FromString("1quote");
    if (!PySyck_1QuoteStyle) return;
    PySyck_2QuoteStyle = PyString_FromString("2quote");
    if (!PySyck_2QuoteStyle) return;
    PySyck_FoldStyle = PyString_FromString("fold");
    if (!PySyck_FoldStyle) return;
    PySyck_LiteralStyle = PyString_FromString("literal");
    if (!PySyck_LiteralStyle) return;
    PySyck_PlainStyle = PyString_FromString("plain");
    if (!PySyck_PlainStyle) return;

    PySyck_StripChomp = PyString_FromString("-");
    if (!PySyck_StripChomp) return;
    PySyck_KeepChomp = PyString_FromString("+");
    if (!PySyck_KeepChomp) return;

    m = Py_InitModule3("_syck", PySyck_methods, PySyck_doc);

    Py_INCREF(PySyck_Error);
    if (PyModule_AddObject(m, "error", (PyObject *)PySyck_Error) < 0)
        return;
    Py_INCREF(&PySyckNode_Type);
    if (PyModule_AddObject(m, "Node", (PyObject *)&PySyckNode_Type) < 0)
        return;
    Py_INCREF(&PySyckScalar_Type);
    if (PyModule_AddObject(m, "Scalar", (PyObject *)&PySyckScalar_Type) < 0)
        return;
    Py_INCREF(&PySyckSeq_Type);
    if (PyModule_AddObject(m, "Seq", (PyObject *)&PySyckSeq_Type) < 0)
        return;
    Py_INCREF(&PySyckMap_Type);
    if (PyModule_AddObject(m, "Map", (PyObject *)&PySyckMap_Type) < 0)
        return;
    Py_INCREF(&PySyckParser_Type);
    if (PyModule_AddObject(m, "Parser", (PyObject *)&PySyckParser_Type) < 0)
        return;
    Py_INCREF(&PySyckEmitter_Type);
    if (PyModule_AddObject(m, "Emitter", (PyObject *)&PySyckEmitter_Type) < 0)
        return;
}

