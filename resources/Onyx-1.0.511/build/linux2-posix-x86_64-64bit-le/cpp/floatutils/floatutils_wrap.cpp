///////////////////////////////////////////////////////////////////////////
//
// File:         floatutils_wrap.cpp
// Date:         Wed 1 Oct 2008 11:45
// Author:       Ken Basye
// Description:  C side of a wrapper for the floatutils library
//
// This file is part of Onyx   http://onyxtools.sourceforge.net
//
// Copyright 2008 The Johns Hopkins University
//
// Licensed under the Apache License, Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//   http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied.  See the License for the specific language governing
// permissions and limitations under the License.
//
///////////////////////////////////////////////////////////////////////////

#include<Python.h>
#include"floatutils.h"

// double readable_string_to_float(const char *s);
static PyObject *
wrap_readable_string_to_float(PyObject *self, PyObject *args)
{
    const char *readable;

    if (!PyArg_ParseTuple(args, "s", &readable))
    {
        PyErr_BadArgument();
        return NULL;
    }
    double d = readable_string_to_float(readable);
    // A Nan return means something went wrong in the parsing.
    // Figuring out what went wrong is left to the Python part of the wrapper
    if(fpclassify(d) == FP_NAN)
    {
        PyErr_Format(PyExc_ValueError, "String %s could not be decoded", readable);
        return NULL;
    }
    return Py_BuildValue("d", d);
}

// void float_to_readable_string(double f, char* buf);
static PyObject *
wrap_float_to_readable_string(PyObject *self, PyObject *args)
{
    double d;
    if (!PyArg_ParseTuple(args, "d", &d))
        return NULL;
    char buffer[FLOATUTILS_READABLE_LEN];
    float_to_readable_string(d, buffer);
    return Py_BuildValue("s", buffer);
}

// fpclassify
static PyObject *
wrap_fpclassify(PyObject *self, PyObject *args)
{
    double d;
    if (!PyArg_ParseTuple(args, "d", &d))
        return NULL;
    int fpc = fpclassify(d);
    return Py_BuildValue("i", fpc);
}


static PyMethodDef methods[] = {
    {"readable_string_to_float",  wrap_readable_string_to_float, METH_VARARGS, "Decode a readable string to float."},
    {"float_to_readable_string",  wrap_float_to_readable_string, METH_VARARGS, "Encode a float as readable string."},
    {"fpclassify",  wrap_fpclassify, METH_VARARGS, "Classify a float as C99's fpclassify does."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
init_floatutils(void)
{
    PyObject *m, *d;
    PyObject *tmp;

    m = Py_InitModule("_floatutils", methods);
    d = PyModule_GetDict(m);

    // For use with fpclassify, we place the following symbols and their values into the dict for
    // this module.  

    // From the fpclassify man page:
    // FP_INFINITE   Indicates that x is an infinite number.
    // FP_NAN        Indicates that x is not a number (NaN).
    // FP_NORMAL     Indicates that x is a normalized number.
    // FP_SUBNORMAL  Indicates that x is a denormalized number.
    // FP_ZERO       Indicates that x is zero (0 or -0).

#define ADD_CONSTANT(_C_)\
    tmp = PyInt_FromLong(_C_);\
    PyDict_SetItemString(d, #_C_, tmp);\
    Py_DECREF(tmp);

    ADD_CONSTANT(FP_INFINITE)
    ADD_CONSTANT(FP_NAN)
    ADD_CONSTANT(FP_NORMAL)
    ADD_CONSTANT(FP_SUBNORMAL)
    ADD_CONSTANT(FP_ZERO)
#undef ADD_CONSTANT
}


