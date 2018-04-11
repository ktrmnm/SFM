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

#include <vector>
#include <set>
#include <utility>

#include <Python.h>

namespace py_utils {

void set_value_error(const char* expected);
void set_value_error(const char* expected, PyObject* p);

std::vector<double> py_list_to_std_vector(PyObject* py_list);
std::vector<std::pair<std::size_t, std::size_t>> py_list_to_vector_of_pairs(PyObject* py_edges);
std::vector<std::set<std::size_t>> py_list_to_vector_of_sets(PyObject* py_edges);

}

namespace py_utils {

void set_value_error(const char* expected) {
  PyErr_SetObject(PyExc_ValueError,
                  PyUnicode_FromFormat(
                      "Invalid format of Python object (expected: %s)", expected)
                  );
}

void set_value_error(const char* expected, PyObject* p) {
  PyErr_SetObject(PyExc_ValueError,
                  PyUnicode_FromFormat(
                      "Invalid format of Python object (expected: %s): %S",
                      expected, PyObject_Str(p))
                  );
}

std::vector<double> py_list_to_std_vector(PyObject* py_list) {
  if (!PyList_Check(py_list)) {
    py_utils::set_value_error("list");
    return std::vector<double>();
  }
  Py_ssize_t n = PyList_Size(py_list);
  std::vector<double> data;
  for (Py_ssize_t i = 0; i < n; ++i) {
    double val = PyFloat_AsDouble(PyList_GET_ITEM(py_list, i));
    data.push_back(val);
  }
  return data;
}

std::vector<std::pair<std::size_t, std::size_t>> py_list_to_vector_of_pairs(PyObject* py_edges) {
  if (!PyList_Check(py_edges)) {
    py_utils::set_value_error("list");
    return std::vector<std::pair<std::size_t, std::size_t>>();
  }
  Py_ssize_t n = PyList_Size(py_edges);
  std::vector<std::pair<std::size_t, std::size_t>> data;
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* edge = PyList_GET_ITEM(py_edges, i);
    if (!PyTuple_Check(edge)) {
      py_utils::set_value_error("tuple(int, int)");
      return std::vector<std::pair<std::size_t, std::size_t>>();
    }
    PyObject* src = PyTuple_GET_ITEM(edge, 0);
    PyObject* dst = PyTuple_GET_ITEM(edge, 1);
    if (!PyLong_Check(src) || !PyLong_Check(dst)) {
      py_utils::set_value_error("tuple(int, int)");
      return std::vector<std::pair<std::size_t, std::size_t>>();
    }
    data.push_back(std::make_pair(
      (std::size_t)PyLong_AsLong(src), (std::size_t)PyLong_AsLong(dst)
    ));
  }
  return data;
}

std::vector<std::set<std::size_t>> py_list_to_vector_of_sets(PyObject* py_edges) {
  if (!PyList_Check(py_edges)) {
    py_utils::set_value_error("list");
    return std::vector<std::set<std::size_t>>();
  }
  Py_ssize_t n = PyList_Size(py_edges);
  std::vector<std::set<std::size_t>> data;
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* edge = PyList_GET_ITEM(py_edges, i);
    if (!PyList_Check(edge)) {
      py_utils::set_value_error("list[int]");
      return std::vector<std::set<std::size_t>>();
    }
    Py_ssize_t edge_card = PyList_Size(edge);
    std::set<std::size_t> e; // c++ hyperedge
    for (Py_ssize_t j = 0; j < edge_card; ++j) {
      std::size_t elem = (std::size_t)PyLong_AsLong(PyList_GET_ITEM(edge, j));
      e.insert(elem);
    }
    data.push_back(std::move(e));
  }
  return data;
}

}
