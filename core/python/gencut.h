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

#include <vector>
#include <utility>
#include <iostream>
#include "core/utils.h"
#include "core/graph/stcut.h"
#include "core/graph/hypergraph_cut.h"
#include "core/graph/divide_conquer.h"
#include "core/python/py_utils.h"

namespace submodular {

PyObject* graph_prox_grid_1d(PyArrayObject* y, double alpha, bool directed, double tol);
PyObject* graph_prox_grid_2d(PyArrayObject* y, double alpha, bool directed, double tol);
PyObject* graph_prox(PyArrayObject* y, double alpha, PyObject* edge_list, PyObject* capacities, bool directed, double tol);
PyObject* hypergraph_prox(PyArrayObject* y, double alpha, PyObject* hyperedge_list, PyObject* capacities, double tol);

}


namespace submodular {

PyObject* graph_prox_grid_1d(PyArrayObject* y, double alpha, bool directed, double tol) {
  if (!PyArray_Check(y)) {
    py_utils::set_value_error("numpy.ndarray");
    return NULL;
  }

  // If |alpha| is smaller than given tolerance, return a copy of y
  if (utils::is_abs_close(alpha, 0.0, tol)) {
    return PyArray_NewCopy(y, NPY_CORDER);
  }
  std::vector<double> y_data((double *)PyArray_DATA(y), ((double *)PyArray_DATA(y)) + PyArray_DIM(y, 0));
  auto n = y_data.size();

  std::vector<std::pair<std::size_t, std::size_t>> edges;
  std::vector<double> capacities(n - 1, alpha);
  for (std::size_t i = 0; i < n - 1; ++i) {
    edges.push_back(std::make_pair(i, i + 1));
  }
  for (auto& y_i: y_data) {
    y_i = -y_i;
  }

  // alpha * F_cut - y
  auto F = CutPlusModular<double>::FromEdgeList(
    n, (directed ? DIRECTED : UNDIRECTED), edges, capacities, y_data
  );
  DivideConquerMNP<double> solver;
  auto solution = solver.Solve(F);
  std::vector<double> base(n);
  for (std::size_t i = 0; i < n; ++i) {
    base[i] = - solution[i];
  }

  npy_intp dims[1] = { (npy_intp) n };
  PyObject* py_base = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
  std::copy(base.begin(), base.end(), (double *)PyArray_DATA(py_base));
  return py_base;
}

PyObject* graph_prox_grid_2d(PyArrayObject* y, double alpha, bool directed, double tol) {
  if (!PyArray_Check(y)) {
    py_utils::set_value_error("numpy.ndarray");
    return nullptr;
  }
  if (PyArray_NDIM(y) != 2) {
    PyErr_SetObject(PyExc_ValueError,
                    PyUnicode_FromFormat(
                        "Invalid NDIM of array (expected: 2): %d", PyArray_NDIM(y))
                    );
    return nullptr;
  }

  // If |alpha| is smaller than given tolerance, return a copy of y
  if (utils::is_close(alpha, 0.0, tol)) {
    return PyArray_NewCopy(y, NPY_CORDER);
  }

  auto y_dims = PyArray_DIMS(y);
  auto n_0 = static_cast<std::size_t>(y_dims[0]);
  auto n_1 = static_cast<std::size_t>(y_dims[1]);
  auto y_flatten = PyArray_Flatten(y, NPY_CORDER);
  std::vector<double> y_data((double *)PyArray_DATA(y_flatten),
                            ((double *)PyArray_DATA(y_flatten)) + PyArray_DIM(y_flatten, 0));
  auto n = y_data.size();

  // Make a 2-d grid graph
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (std::size_t i = 0; i < n_0; ++i) {
    for (std::size_t j = 0; j < n_1; ++j) {
      int node_id = n_0 * i + j;
      if (j < n_1 - 1) {
        edges.push_back(std::make_pair(node_id, node_id + 1));
        //std::cout << "add edge (" << node_id << ", " << node_id + 1 << ")" << std::endl;
      }
      if (i < n_0 - 1) {
        edges.push_back(std::make_pair(node_id, node_id + n_0));
        //std::cout << "add edge (" << node_id << ", " << node_id + n_0 << ")" << std::endl;
      }
    }
  }

  std::vector<double> capacities(edges.size(), alpha);
  for (auto& y_i: y_data) {
    y_i = -y_i;
  }

  auto F = CutPlusModular<double>::FromEdgeList(
    n, (directed ? DIRECTED : UNDIRECTED), edges, capacities, y_data
  );
  DivideConquerMNP<double> solver;
  auto solution = solver.Solve(F);
  std::vector<double> base(n);
  for (std::size_t i = 0; i < n; ++i) {
    base[i] = - solution[i];
  }

  npy_intp dims[1] = { (npy_intp) n };
  PyObject* py_base = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
  std::copy(base.begin(), base.end(), (double*)PyArray_DATA(py_base));
  PyArray_Dims shape = { (npy_intp*) y_dims, 2 };
  py_base = PyArray_Newshape((PyArrayObject*)py_base, &shape, NPY_CORDER);
  Py_XDECREF(y_flatten);
  return py_base;
}

PyObject* graph_prox(PyArrayObject* y, double alpha, PyObject* edge_list, PyObject* capacities, bool directed, double tol) {
  if (!PyArray_Check(y)) {
    py_utils::set_value_error("numpy.ndarray");
    return nullptr;
  }

  // If |alpha| is smaller than given tolerance, return a copy of y
  if (utils::is_close(alpha, 0.0, tol)) {
    return PyArray_NewCopy(y, NPY_CORDER);
  }
  std::vector<double> y_data((double *)PyArray_DATA(y), ((double *)PyArray_DATA(y)) + PyArray_DIM(y, 0));
  auto n = y_data.size();
  auto edges = py_utils::py_list_to_vector_of_pairs(edge_list);
  auto caps = py_utils::py_list_to_std_vector(capacities);
  for (auto& y_i: y_data) {
    y_i = - y_i / alpha;
  }

  auto F = CutPlusModular<double>::FromEdgeList(
    n, (directed ? DIRECTED : UNDIRECTED), edges, caps, y_data
  );
  DivideConquerMNP<double> solver;
  auto solution = solver.Solve(F);
  std::vector<double> base(n);
  for (std::size_t i = 0; i < n; ++i) {
    base[i] = - solution[i] * alpha;
  }

  npy_intp dims[1] = { (npy_intp) n };
  PyObject* py_base = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
  std::copy(base.begin(), base.end(), (double *)PyArray_DATA(py_base));
  return py_base;
}

PyObject* hypergraph_prox(PyArrayObject* y, double alpha, PyObject* hyperedge_list, PyObject* capacities, double tol) {
  if (!PyArray_Check(y)) {
    py_utils::set_value_error("numpy.ndarray");
    return nullptr;
  }

  // If |alpha| is smaller than given tolerance, return a copy of y
  if (utils::is_close(alpha, 0.0, tol)) {
    return PyArray_NewCopy(y, NPY_CORDER);
  }
  std::vector<double> y_data((double *)PyArray_DATA(y), ((double *)PyArray_DATA(y)) + PyArray_DIM(y, 0));
  auto n = y_data.size();
  auto hyperedges = py_utils::py_list_to_vector_of_sets(hyperedge_list);
  auto caps = py_utils::py_list_to_std_vector(capacities);
  for (auto& y_i: y_data) {
    y_i = - y_i / alpha;
  }

  auto F = HypergraphCutPlusModular<double>::FromHyperEdgeList(
    n, hyperedges, caps, y_data
  );
  DivideConquerMNP<double> solver;
  auto solution = solver.Solve(F);
  std::vector<double> base(n);
  for (std::size_t i = 0; i < n; ++i) {
    base[i] = - solution[i];
  }

  npy_intp dims[1] = { (npy_intp) n };
  PyObject* py_base = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
  std::copy(base.begin(), base.end(), (double *)PyArray_DATA(py_base));
  return py_base;
}

}

#endif
