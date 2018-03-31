import numpy as np

cdef extern from "core/python/pyarray_symbol_core.h":
    pass

cimport cython
cimport numpy as np
np.import_array()
from libcpp cimport bool as bool_t


"""
Import C++ classes
"""

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        vector()
        T& operator[](int)
        T& at(int)


cdef extern from "core/python/sfm_core.h" namespace "submodular":
    cdef cppclass Set:
        Set()
        Set(int)
        Set MakeDense(int)
        Set MakeEmpty(int)
        Set FromIndices(int, vector[int])
        bool_t operator[](int)
        vector[int] GetMembers()


    cdef cppclass SFMReporter:
        SFMReporter()
        void SetMessage(str)


    # todo: わかりにくいのでなんとかする
    cdef set set_to_py_set(Set) except +
    cdef Set py_set_to_set(set, int) except +
    cdef Set py_list_to_set(list, int) except +
    cdef dict reporter_to_py_dict(SFMReporter) except +
    cdef vector[double] py_array_to_std_vector(np.ndarray) except +


    cdef cppclass OracleWrapper:
        bool_t is_reducible_
        bool_t is_graph_


    cdef OracleWrapper factory_modular(np.ndarray, bool_t)
    #cdef OracleWrapper r_factory_modular(np.ndarray)
    cdef OracleWrapper factory_iwata_test_function(int, bool_t)
    #cdef OracleWrapper r_factory_iwata_test_function(int)

    cdef SFMReporter minimize_bf(OracleWrapper, SFMReporter*)
    cdef SFMReporter minimize_fw(OracleWrapper, SFMReporter*, double)


methods = ["fw", "bf"]
oracles = [
    "modular",
    "iwata_test_function"
]

cpdef dict _minimize(object F, str method, dict kwargs):
    cdef SFMReporter* reporter = new SFMReporter()
    cdef dict result
    cdef bool_t reducible = False

    cdef np.ndarray x
    cdef int n
    cdef double precision
    #cdef OracleWrapper oracle
    cdef SFMReporter sol

    if method not in methods:
        #reporter.SetMessage("invalid method name")
        result = reporter_to_py_dict(reporter[0])

    else:

        if (F.name is None) or (F.name not in oracles):
            raise ValueError("unknown oracle")

        if F.name is "modular":
            x = F.data.get('x')
            oracle = factory_modular(x, reducible)
        elif F.name is "iwata_test_function":
            n = F.data.get('n')
            oracle = factory_iwata_test_function(n, reducible)

        if method is "fw":
            precision = kwargs.get('precision', 0.5)
            sol = minimize_fw(oracle, reporter, precision)
            result = reporter_to_py_dict(sol)
        elif method is "bf":
            sol = minimize_bf(oracle, reporter)
            result = reporter_to_py_dict(sol)

    del reporter
    return result
