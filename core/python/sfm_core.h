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

#ifndef PYTHON_SFM_CORE_H
#define PYTHON_SFM_CORE_H

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL SFM_CORE_PyArray_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <string>
#include <vector>
#include <set>
#include <utility>
#include <memory>
#include <stdexcept>

#include "core/python/py_utils.h"

#include "core/set_utils.h"
#include "core/oracle.h"
#include "core/reporter.h"
#include "core/sfm_algorithm.h"

// Oracles
#include "core/oracles/modular.h"
#include "core/oracles/iwata_test_function.h"
#include "core/graph/generalized_cut.h"
#include "core/graph/stcut.h"
#include "core/graph/hypergraph_cut.h"

// Algorithms
#include "core/algorithms/brute_force.h"
#include "core/algorithms/sfm_fw.h"


namespace submodular {
// -------------------------------------
// Converters
// -------------------------------------

PyObject* set_to_py_set(const Set& X);
Set py_set_to_set(PyObject* set, int n_ground);
Set py_list_to_set(PyObject* members, int n_ground);
PyObject* reporter_to_py_dict(const SFMReporter& reporter);
std::vector<double> py_array_to_std_vector(PyArrayObject* py_array);

// -------------------------------------
// Oracle wrappers
// -------------------------------------

// NOTE: This class is used as a wrapper of any submodular oracle objects.
// The declaration is visible from python codes. A major purpose of this class is to
// avoid instanciatiation of the abstract SubmodularOracle class.
// Any oracle factory methods called from python codes must return instances of this wrapper.
class OracleWrapper;
class ReducibleOracleWrapper;
class GraphOracleWrapper;

OracleWrapper factory_modular(PyArrayObject* x);
ReducibleOracleWrapper r_factory_modular(PyArrayObject* x);
OracleWrapper factory_iwata_test_function(int n);
ReducibleOracleWrapper r_factory_iwata_test_function(int n);
OracleWrapper factory_groupwise_iwata_test_function(int n, int k);
ReducibleOracleWrapper r_factory_groupwise_iwata_test_function(int n, int k);

GraphOracleWrapper factory_stcut(int n, int s, int t, PyObject* edge_list, PyObject* capacities);
GraphOracleWrapper factory_stcut_plus_modular(int n, int s, int t, PyObject* edge_list, PyObject* capacities, PyArrayObject* x);
GraphOracleWrapper factory_cut_plus_modular(int n, bool directed, PyObject* edge_list, PyObject* capacities, PyArrayObject* x);
GraphOracleWrapper factory_hypergraph_cut_plus_modular(int n, PyObject* hyper_edge_list, PyObject* capacities, PyArrayObject* x);

// -------------------------------------
// Algorithm wrappers
// -------------------------------------

SFMReporter minimize_bf(OracleWrapper F_wrap, SFMReporter* reporter);
SFMReporter minimize_bf(ReducibleOracleWrapper F_wrap, SFMReporter* reporter);
SFMReporter minimize_bf(GraphOracleWrapper F_wrap, SFMReporter* reporter);

SFMReporter minimize_fw(OracleWrapper F_wrap, SFMReporter* reporter, double precision);
SFMReporter minimize_fw(ReducibleOracleWrapper F_wrap, SFMReporter* reporter, double precision);
SFMReporter minimize_fw(GraphOracleWrapper F_wrap, SFMReporter* reporter, double precision);

SFMReporter minimize_graph(GraphOracleWrapper F_wrap, SFMReporter* reporter);

}


namespace submodular {
// -------------------------------------
// Converters impl
// -------------------------------------

PyObject* set_to_py_set(const Set& X) {
  PyObject* py_set = PySet_New(NULL);
  for (const auto& i: X.GetMembers()) {
    PyObject* elem = PyLong_FromSize_t(i);
    PySet_Add(py_set, elem);
    Py_XDECREF(elem);
  }
  return py_set;
}

Set py_set_to_set(PyObject* set, int n_ground) {
  if (!PySet_Check(set)) {
    py_utils::set_value_error("set");
  }
  std::size_t n = static_cast<std::size_t>(n_ground);
  Set X = Set::MakeEmpty(n);
  PyObject* set_copy = PySet_New(set);
  PyObject* py_elem = PySet_Pop(set_copy);
  while (py_elem != NULL) {
    py_elem = PySet_Pop(set_copy); // NOTE: this makes a new reference
    if (PyLong_Check(py_elem)) {
      std::size_t i = static_cast<std::size_t>(PyLong_AsLong(py_elem));
      X.AddElement(i);
    }
    Py_XDECREF(py_elem);
  }
  Py_XDECREF(set_copy);
  return X;
}

Set py_list_to_set(PyObject* members, int n_ground) {
  if (!PyList_Check(members)) {
    py_utils::set_value_error("list");
  }
  std::size_t n = static_cast<std::size_t>(n_ground);
  Set X = Set::MakeEmpty(n);
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(members); ++i) {
    PyObject* py_elem = PyList_GET_ITEM(members, i); // NOTE: this is a borrowed reference
    if (PyLong_Check(py_elem)) {
      std::size_t i = static_cast<std::size_t>(PyLong_AsLong(py_elem));
      X.AddElement(i);
    }
  }
  return X;
}

PyObject* reporter_to_py_dict(const SFMReporter& reporter) {
  PyObject* py_dict = PyDict_New();
  PyDict_SetItemString(py_dict, "algorithm", PyUnicode_FromString(reporter.algorithm_name_.data()));
  PyDict_SetItemString(py_dict, "oracle", PyUnicode_FromString(reporter.oracle_name_.data()));
  PyDict_SetItemString(py_dict, "minimum_value", PyFloat_FromDouble(reporter.minimum_value_));

  PyObject* py_minimizer = set_to_py_set(reporter.minimizer_);
  PyDict_SetItemString(py_dict, "minimizer", py_minimizer);
  Py_XDECREF(py_minimizer);

  if (reporter.times_.count(ReportKind::TOTAL) == 1) {
    PyDict_SetItemString(py_dict, "total_time",
                        PyLong_FromLong(reporter.times_.at(ReportKind::TOTAL).count()));
  }
  if (reporter.times_.count(ReportKind::PREPROCESSING) == 1) {
    PyDict_SetItemString(py_dict, "preprocessing_time",
                        PyLong_FromLong(reporter.times_.at(ReportKind::PREPROCESSING).count()));
  }
  if (reporter.counts_.count(ReportKind::ORACLE) == 1) {
    PyDict_SetItemString(py_dict, "oracle_count",
                        PyLong_FromLong(reporter.counts_.at(ReportKind::ORACLE)));
  }
  if (reporter.times_.count(ReportKind::ORACLE) == 1) {
    PyDict_SetItemString(py_dict, "oracle_time",
                        PyLong_FromLong(reporter.times_.at(ReportKind::ORACLE).count()));
  }
  if (reporter.counts_.count(ReportKind::BASE) == 1) {
    PyDict_SetItemString(py_dict, "base_count",
                        PyLong_FromLong(reporter.counts_.at(ReportKind::BASE)));
  }
  if (reporter.times_.count(ReportKind::BASE) == 1) {
    PyDict_SetItemString(py_dict, "base_time",
                        PyLong_FromLong(reporter.times_.at(ReportKind::BASE).count()));
  }
  if (reporter.counts_.count(ReportKind::ITERATION) == 1) {
    PyDict_SetItemString(py_dict, "iterations",
                        PyLong_FromLong(reporter.counts_.at(ReportKind::ITERATION)));
  }
  if (!reporter.msg_.empty()) {
    PyObject* msg_list = PyList_New((Py_ssize_t)reporter.msg_.size());
    for (std::size_t i = 0; i < reporter.msg_.size(); ++i) {
      PyList_SET_ITEM(msg_list, (Py_ssize_t)i,
                      PyUnicode_FromString(reporter.msg_[i].data()));
    }
    PyDict_SetItemString(py_dict, "messages", msg_list);
    Py_XDECREF(msg_list);
  }

  return py_dict;
}

std::vector<double> py_array_to_std_vector(PyArrayObject* py_array) {
  if (!PyArray_Check(py_array)) {
    py_utils::set_value_error("numpy.ndarray");
    return std::vector<double>();
  }
  std::vector<double> data((double *)PyArray_DATA(py_array),
                          ((double *)PyArray_DATA(py_array)) + PyArray_DIM(py_array, 0));
  return data;
}


// -------------------------------------
// Oracle wrappers impl
// -------------------------------------

class OracleWrapper {
public:
  OracleWrapper()
    : F_ptr(nullptr),
      is_reducible(false),
      is_graph(false)
  {}

  std::shared_ptr<SubmodularOracle<double>> F_ptr;
  bool is_reducible;
  bool is_graph;
};

class ReducibleOracleWrapper {
public:
  ReducibleOracleWrapper()
    : F_ptr(nullptr),
      is_reducible(true),
      is_graph(false)
  {}

  std::shared_ptr<ReducibleOracle<double>> F_ptr;
  bool is_reducible;
  bool is_graph;
};

class GraphOracleWrapper {
public:
  GraphOracleWrapper()
    : F_ptr(nullptr),
      is_reducible(false),
      is_graph(true)  
  {}

  std::shared_ptr<GeneralizedCutOracle<double>> F_ptr;
  bool is_reducible;
  bool is_graph;
};

OracleWrapper factory_modular(PyArrayObject* x) {
  auto data = py_array_to_std_vector(x);
  OracleWrapper ow;
  ow.F_ptr = std::make_shared<ModularOracle<double>>(data);
  return ow;
}

ReducibleOracleWrapper r_factory_modular(PyArrayObject* x) {
  auto data = py_array_to_std_vector(x);
  ReducibleOracleWrapper ow;
  ModularOracle<double> modular(data);
  ow.F_ptr = std::make_shared<ReducibleOracle<double>>(std::move(modular));
  return ow;
}

OracleWrapper factory_iwata_test_function(int n) {
  OracleWrapper ow;
  ow.F_ptr = std::make_shared<IwataTestFunction<double>>(n);
  return ow;
}

ReducibleOracleWrapper r_factory_iwata_test_function(int n) {
  ReducibleOracleWrapper ow;
  IwataTestFunction<double> F(n);
  ow.F_ptr = std::make_shared<ReducibleOracle<double>>(std::move(F));
  return ow;
}

OracleWrapper factory_groupwise_iwata_test_function(int n, int k) {
  OracleWrapper ow;
  ow.F_ptr = std::make_shared<GroupwiseIwataTestFunction<double>>(n, k);
  return ow;
}

ReducibleOracleWrapper r_factory_groupwise_iwata_test_function(int n, int k) {
  ReducibleOracleWrapper ow;
  GroupwiseIwataTestFunction<double> F(n, k);
  ow.F_ptr = std::make_shared<ReducibleOracle<double>>(std::move(F));
  return ow;
}


GraphOracleWrapper factory_stcut(int n, int s, int t, PyObject* edge_list, PyObject* capacities) {
  GraphOracleWrapper ow;
  ow.is_graph = true;
  auto edges = py_utils::py_list_to_vector_of_pairs(edge_list);
  auto caps = py_utils::py_list_to_std_vector(capacities);
  auto F = STCut<double>::FromEdgeList(n, s, t, edges, caps);
  ow.F_ptr = std::make_shared<STCut<double>>(std::move(F)); //NOTE: ムーブコンストラクタは明示的に定義していない
  return ow;
}

GraphOracleWrapper factory_stcut_plus_modular(int n, int s, int t, PyObject* edge_list, PyObject* capacities, PyArrayObject* x) {
  GraphOracleWrapper ow;
  ow.is_graph = true;
  auto edges = py_utils::py_list_to_vector_of_pairs(edge_list);
  auto caps = py_utils::py_list_to_std_vector(capacities);
  auto modular = py_array_to_std_vector(x);
  auto F = STCutPlusModular<double>::FromEdgeList(n, s, t, edges, caps, modular);
  ow.F_ptr = std::make_shared<STCutPlusModular<double>>(std::move(F)); //NOTE: ムーブコンストラクタは明示的に定義していない
  return ow;
}

GraphOracleWrapper factory_cut_plus_modular(int n, bool directed, PyObject* edge_list, PyObject* capacities, PyArrayObject* x) {
  GraphOracleWrapper ow;
  ow.is_graph = true;
  auto edges = py_utils::py_list_to_vector_of_pairs(edge_list);
  auto caps = py_utils::py_list_to_std_vector(capacities);
  auto modular = py_array_to_std_vector(x);
  DirectionKind kind = directed ? DIRECTED : UNDIRECTED;
  auto F = CutPlusModular<double>::FromEdgeList(n, kind, edges, caps, modular);
  ow.F_ptr = std::make_shared<CutPlusModular<double>>(std::move(F)); //NOTE: ムーブコンストラクタは明示的に定義していない
  return ow;
}

GraphOracleWrapper factory_hypergraph_cut_plus_modular(int n, PyObject* hyper_edge_list, PyObject* capacities, PyArrayObject* x) {
  GraphOracleWrapper ow;
  ow.is_graph = true;
  auto edges = py_utils::py_list_to_vector_of_sets(hyper_edge_list);
  auto caps = py_utils::py_list_to_std_vector(capacities);
  auto modular = py_array_to_std_vector(x);
  auto F = HypergraphCutPlusModular<double>::FromHyperEdgeList(n, edges, caps, modular);
  ow.F_ptr = std::make_shared<HypergraphCutPlusModular<double>>(std::move(F));
  return ow;
}

// -------------------------------------
// Algorithm wrappers impl
// -------------------------------------

SFMReporter minimize_bf(OracleWrapper F_wrap, SFMReporter* reporter) {
  BruteForce<double> solver;
  if (reporter != nullptr) {
    solver.SetReporter(*reporter);
  }
  solver.Minimize(*(F_wrap.F_ptr));
  return solver.GetReporter();
}

SFMReporter minimize_bf(ReducibleOracleWrapper F_wrap, SFMReporter* reporter) {
  BruteForce<double> solver;
  if (reporter != nullptr) {
    solver.SetReporter(*reporter);
  }
  solver.Minimize(*(F_wrap.F_ptr));
  return solver.GetReporter();
}

SFMReporter minimize_bf(GraphOracleWrapper F_wrap, SFMReporter* reporter) {
  BruteForce<double> solver;
  if (reporter != nullptr) {
    solver.SetReporter(*reporter);
  }
  solver.Minimize(*(F_wrap.F_ptr));
  return solver.GetReporter();
}

SFMReporter minimize_fw(OracleWrapper F_wrap, SFMReporter* reporter, double precision) {
  FWRobust<double> solver(precision);
  if (reporter != nullptr) {
    solver.SetReporter(*reporter);
  }
  solver.Minimize(*(F_wrap.F_ptr));
  return solver.GetReporter();
}

SFMReporter minimize_fw(ReducibleOracleWrapper F_wrap, SFMReporter* reporter, double precision) {
  FWRobust<double> solver(precision);
  if (reporter != nullptr) {
    solver.SetReporter(*reporter);
  }
  solver.Minimize(*(F_wrap.F_ptr));
  return solver.GetReporter();
}

SFMReporter minimize_fw(GraphOracleWrapper F_wrap, SFMReporter* reporter, double precision) {
  FWRobust<double> solver(precision);
  if (reporter != nullptr) {
    solver.SetReporter(*reporter);
  }
  solver.Minimize(*(F_wrap.F_ptr));
  return solver.GetReporter();
}

SFMReporter minimize_graph(GraphOracleWrapper F_wrap, SFMReporter* reporter) {
  SFMAlgorithmGeneralizedCut<double> solver;
  if (reporter != nullptr) {
    solver.SetReporter(*reporter);
  }
  solver.Minimize(*(F_wrap.F_ptr));
  return solver.GetReporter();
}

}


#endif
