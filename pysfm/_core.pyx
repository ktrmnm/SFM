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

    cdef Set Set_to_py_Set(Set) except +
    cdef Set py_Set_to_Set(Set, int) except +
    cdef Set py_list_to_Set(list, int) except +
    cdef dict reporter_to_py_dict(SFMReporter) except +

    cdef cppclass SubmodularOracleD:
        pass

    cdef cppclass ModularOracleD:
        ModularOracleD(vector[float])
        float Call(Set)

    cdef cppclass IwataTestFunctionD:
        IwataTestFunctionD(int)
        float Call(Set)


    cdef cppclass BruteForceD:
        BruteForceD()
        Minimize(SubmodularOracleD)
        SFMReporter GetReport()

    cdef cppclass FWRobustD:
        FWRobustD()
        FWRobustD(float)
        Minimize(SubmodularOracleD)
        SFMReporter GetReport()
