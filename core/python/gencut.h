// Copyright 2018 Kentaro Minami. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

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
