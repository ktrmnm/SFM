#include "core/python/gencut.h"

#include <vector>
#include <utility>
#include <iostream>
#include "core/utils.h"
#include "core/graph/stcut.h"
#include "core/graph/divide_conquer.h"
#include "core/python/py_utils.h"

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

  std::vector<std::pair<std::size_t, std::size_t>> edges;
  for (std::size_t i = 0; i < n_0; ++i) {
    for (std::size_t j = 0; j < n_1; ++j) {
      int node_id = n_0 * i + j;
      if (j < n_1 - 1) {
        edges.push_back(std::make_pair(node_id, node_id + 1));
      }
      if (i < n_0 - 1) {
        edges.push_back(std::make_pair(node_id, node_id + n_0));
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
  PyArray_Dims shape = { y_dims, (int) n };
  py_base = PyArray_Newshape((PyArrayObject*)py_base, &shape, NPY_CORDER);
  Py_XDECREF(y_flatten);
  return py_base;
}

PyObject* graph_prox(PyArrayObject* y, double alpha, PyObject* edge_list, bool directed, double tol) {
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

  if (!PyList_Check(edge_list)) {
    py_utils::set_value_error("list of tuple (int src, int dst, {capacity: float cap})");
    return nullptr;
  }
  Py_ssize_t m = PyList_Size(edge_list);
  std::vector<std::pair<std::size_t, std::size_t>> edges;
  std::vector<double> capacities;
  for (Py_ssize_t i = 0; i < m; ++i) {
    auto edge = PyList_GetItem(edge_list, i);
    if (!PyTuple_Check(edge) || PyTuple_Size(edge) < 3) {
      py_utils::set_value_error("list of tuple (int src, int dst, {capacity: float cap})");
      return nullptr;
    }
    auto py_src = PyTuple_GetItem(edge, 0);
    auto py_dst = PyTuple_GetItem(edge, 1);
    auto py_dict = PyTuple_GetItem(edge, 2);
    if (!PyLong_Check(py_src) || !PyLong_Check(py_dst) || !PyDict_Check(py_dict)) {
      py_utils::set_value_error("list of tuple (int src, int dst, {capacity: float cap})");
      return nullptr;
    }
    auto py_cap = PyDict_GetItemString(py_dict, "capacity");
    if (py_cap == NULL || !PyFloat_Check(py_cap)) {
      py_utils::set_value_error("list of tuple (int src, int dst, {capacity: float cap})");
      return nullptr;
    }
    auto src = (std::size_t)PyLong_AsLong(py_src);
    auto dst = (std::size_t)PyLong_AsLong(py_dst);
    double cap = (double)PyFloat_AsDouble(py_cap);
    edges.push_back(std::make_pair(src, dst));
    capacities.push_back(cap);
  }
  for (auto& y_i: y_data) {
    y_i = - y_i / alpha;
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
  std::copy(base.begin(), base.end(), (double *)PyArray_DATA(py_base));
  return py_base;
}

}
