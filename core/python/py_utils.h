#include <Python.h>

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

}
