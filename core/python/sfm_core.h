#ifndef PYTHON_SFM_CORE_H
#define PYTHON_SFM_CORE_H

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL SFM_CORE_PyArray_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "core/python/py_utils.h"

#include "core/set_utils.h"
#include "core/oracle.h"
#include "core/reporter.h"
#include "core/sfm_algorithm.h"

#include "core/oracles/modular.h"
#include "core/oracles/iwata_test_function.h"

#include "core/algorithms/brute_force.h"
#include "core/algorithms/sfm_fw.h"

namespace submodular {

//using CSet = Set;

PyObject* set_to_py_set(const Set& X);
Set py_set_to_set(PyObject* set, PyObject* n_ground);
Set py_list_to_set(PyObject* members, PyObject* n_ground);

PyObject* reporter_to_py_dict(const SFMReporter& reporter);

// Oracle defs
using SubmodularOracleD = SubmodularOracle<double>;
using ModularOracleD = ModularOracle<double>;
using IwataTestFunctionD = IwataTestFunction<double>;

// Algorithm defs
using BruteForceD = BruteForce<double>;
using FWRobustD = FWRobust<double>;

}

namespace submodular {

PyObject* set_to_py_set(const Set& X) {
  PyObject* py_set = PySet_New(NULL);
  for (const auto& i: X.GetMembers()) {
    PyObject* elem = PyLong_FromSize_t(i);
    PySet_Add(py_set, elem);
    Py_XDECREF(elem);
  }
  return py_set;
}

Set py_set_to_set(PyObject* set, PyObject* n_ground) {
  if (!PySet_Check(set) || !PyLong_Check(n_ground)) {
    py_utils::set_value_error("set, int");
  }
  std::size_t n = static_cast<std::size_t>(PyLong_AsLong(n_ground));
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

Set py_list_to_set(PyObject* members, PyObject* n_ground) {
  if (!PyList_Check(members) || !PyLong_Check(n_ground)) {
    py_utils::set_value_error("list, int");
  }
  std::size_t n = static_cast<std::size_t>(PyLong_AsLong(n_ground));
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

  return py_dict;
}

}


#endif
