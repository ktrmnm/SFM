#ifndef PYTHON_SFM_GENCUT_H
#define PYTHON_SFM_GENCUT_H

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL SFM_GENCUT_PyArray_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

namespace submodular {

PyObject* graph_prox_grid_1d(PyArrayObject* y, double alpha, bool directed, double tol);
PyObject* graph_prox_grid_2d(PyArrayObject* y, double alpha, bool directed, double tol);
PyObject* graph_prox(PyArrayObject* y, double alpha, PyObject* edge_list, bool directed, double tol);

}

#endif
