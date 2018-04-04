import numpy as np

cdef extern from "core/python/pyarray_symbol_core.h":
    pass

cimport cython
cimport numpy as np
np.import_array()
from libcpp cimport bool as bool_t
from libcpp.string cimport string


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
        void SetMessage(string)


    cdef set set_to_py_set(Set) except +
    cdef Set py_set_to_set(set, int) except +
    cdef Set py_list_to_set(list, int) except +
    cdef dict reporter_to_py_dict(SFMReporter) except +
    cdef vector[double] py_array_to_std_vector(np.ndarray) except +


    cdef cppclass OracleWrapper:
        bool_t is_reducible_
        bool_t is_graph_

    cdef cppclass ReducibleOracleWrapper:
        bool_t is_reducible_
        bool_t is_graph_

    cdef cppclass GraphOracleWrapper:
        bool_t is_reducible_
        bool_t is_graph_


    cdef OracleWrapper factory_modular(np.ndarray)
    cdef ReducibleOracleWrapper r_factory_modular(np.ndarray)
    cdef OracleWrapper factory_iwata_test_function(int)
    cdef ReducibleOracleWrapper r_factory_iwata_test_function(int)
    cdef OracleWrapper factory_groupwise_iwata_test_function(int, int)
    cdef ReducibleOracleWrapper r_factory_groupwise_iwata_test_function(int, int)

    cdef GraphOracleWrapper factory_stcut(int, int, int, list, list)
    cdef GraphOracleWrapper factory_stcut_plus_modular(int, int, int, list, list, np.ndarray)
    cdef GraphOracleWrapper factory_cut_plus_modular(int, bool_t, list, list, np.ndarray)

    cdef SFMReporter minimize_bf(OracleWrapper, SFMReporter*)
    cdef SFMReporter minimize_bf(ReducibleOracleWrapper, SFMReporter*)
    cdef SFMReporter minimize_bf(GraphOracleWrapper, SFMReporter*)
    cdef SFMReporter minimize_fw(OracleWrapper, SFMReporter*, double)
    cdef SFMReporter minimize_fw(ReducibleOracleWrapper, SFMReporter*, double)
    cdef SFMReporter minimize_fw(GraphOracleWrapper, SFMReporter*, double)
    cdef SFMReporter minimize_graph(GraphOracleWrapper, SFMReporter*)


"""
Converter from Python object to C++ OracleWrapper object
"""

_oracles = {
    "modular": ["o", "r"],
    "iwata_test_function": ["o", "r"],
    "groupwise_iwata_test_function": ["o", "r"],
    "stcut": ["g"],
    "stcut_plus_modular": ["g"],
    "cut_plus_modular": ["g"]
}


cdef OracleWrapper _make_o(object F):
    if F.name is "modular":
        return make_modular(F)
    elif F.name is "iwata_test_function":
        return make_iwata_test_function(F)
    elif F.name is "groupwise_iwata_test_function":
        return make_groupwise_iwata_test_function(F)


cdef ReducibleOracleWrapper _make_r(object F):
    if F.name is "modular":
        return r_make_modular(F)
    elif F.name is "iwata_test_function":
        return r_make_iwata_test_function(F)
    elif F.name is "groupwise_iwata_test_function":
        return r_make_groupwise_iwata_test_function(F)


cdef GraphOracleWrapper _make_g(object F):
    if F.name is "stcut":
        return g_make_stcut(F)
    elif F.name is "stcut_plus_modular":
        return g_make_stcut_plus_modular(F)
    elif F.name is "cut_plus_modular":
        return g_make_cut_plus_modular(F)


cdef OracleWrapper make_modular(object F):
    cdef np.ndarray x = F.data.get('x')
    return factory_modular(x)


cdef ReducibleOracleWrapper r_make_modular(object F):
    cdef np.ndarray x = F.data.get('x')
    return r_factory_modular(x)


cdef OracleWrapper make_iwata_test_function(object F):
    cdef int n = F.data.get('n')
    return factory_iwata_test_function(n)


cdef ReducibleOracleWrapper r_make_iwata_test_function(object F):
    cdef int n = F.data.get('n')
    return r_factory_iwata_test_function(n)


cdef OracleWrapper make_groupwise_iwata_test_function(object F):
    cdef int n = F.data.get('n')
    cdef int k = F.data.get('k')
    return factory_groupwise_iwata_test_function(n, k)


cdef ReducibleOracleWrapper r_make_groupwise_iwata_test_function(object F):
    cdef int n = F.data.get('n')
    cdef int k = F.data.get('k')
    return r_factory_groupwise_iwata_test_function(n, k)


cdef GraphOracleWrapper g_make_stcut(object F):
    cdef int n = F.data.get('n')
    cdef int s = F.data.get('s')
    cdef int t = F.data.get('t')
    cdef list edges = F.data.get('edge_list')
    cdef list caps = F.data.get('capacities')
    return factory_stcut(n, s, t, edges, caps)


cdef GraphOracleWrapper g_make_stcut_plus_modular(object F):
    cdef int n = F.data.get('n')
    cdef int s = F.data.get('s')
    cdef int t = F.data.get('t')
    cdef list edges = F.data.get('edge_list')
    cdef list caps = F.data.get('capacities')
    cdef np.ndarray x = F.data.get('x')
    return factory_stcut_plus_modular(n, s, t, edges, caps, x)


cdef GraphOracleWrapper g_make_cut_plus_modular(object F):
    cdef int n = F.data.get('n')
    cdef bool_t directed = F.data.get('directed')
    cdef list edges = F.data.get('edge_list')
    cdef list caps = F.data.get('capacities')
    cdef np.ndarray x = F.data.get('x')
    return factory_cut_plus_modular(n, directed, edges, caps, x)


"""
Main function of SFM
"""


cdef dict _methods = {
    "fw": ["o", "g"],
    "bf": ["o", "g"],
    "maxflow": ["g"]
}


cdef SFMReporter _minimize_o(str method, OracleWrapper oracle, SFMReporter* reporter, dict kwargs):
    if method is "fw":
        return _minimize_fw(oracle, reporter, kwargs)
    elif method is "bf":
        return _minimize_bf(oracle, reporter, kwargs)


cdef SFMReporter _minimize_g(str method, GraphOracleWrapper oracle, SFMReporter* reporter, dict kwargs):
    if method is "maxflow":
        return _g_minimize_graph(oracle, reporter, kwargs)
    elif method is "fw":
        return _g_minimize_fw(oracle, reporter, kwargs)
    elif method is "bf":
        return _g_minimize_bf(oracle, reporter, kwargs)


cdef SFMReporter _minimize_fw(OracleWrapper oracle, SFMReporter* reporter, dict kwargs):
    cdef double precision = kwargs.get('precision', 0.5)
    return minimize_fw(oracle, reporter, precision)


cdef SFMReporter _g_minimize_fw(GraphOracleWrapper oracle, SFMReporter* reporter, dict kwargs):
    cdef double precision = kwargs.get('precision', 0.5)
    return minimize_fw(oracle, reporter, precision)


cdef SFMReporter _minimize_bf(OracleWrapper oracle, SFMReporter* reporter, dict kwargs):
    return minimize_bf(oracle, reporter)


cdef SFMReporter _g_minimize_bf(GraphOracleWrapper oracle, SFMReporter* reporter, dict kwargs):
    return minimize_bf(oracle, reporter)


cdef SFMReporter _g_minimize_graph(GraphOracleWrapper oracle, SFMReporter* reporter, dict kwargs):
    return minimize_graph(oracle, reporter)


cpdef dict _minimize(object F, str method, dict kwargs):
    cdef SFMReporter* reporter = new SFMReporter()
    cdef dict result
    cdef OracleWrapper oracle
    cdef ReducibleOracleWrapper r_oracle
    cdef GraphOracleWrapper g_oracle
    cdef SFMReporter sol

    if method not in _methods.keys():
        reporter.SetMessage("Error: Unknown algorithm name.")
        result = reporter_to_py_dict(reporter[0])
        del reporter
        return result

    if (F.name is None) or (F.name not in _oracles.keys()):
        reporter.SetMessage("Error: Unknown oracle name.")
        result = reporter_to_py_dict(reporter[0])
        del reporter
        return result

    if ("g" in _methods[method]) and ("g" in _oracles[F.name]):
        g_oracle = _make_g(F)
        sol = _minimize_g(method, g_oracle, reporter, kwargs)
    elif ("o" in _methods[method]) and ("o" in _oracles[F.name]):
        oracle = _make_o(F)
        sol = _minimize_o(method, oracle, reporter, kwargs)
    #elif ("r" in _methods[method]) and ("r" in _oracles[F.name]):
    #    r_oracle = _make_r(F)
    #    sol = _minimize_r(method, r_oracle, reporter, kwargs)
    else:
        reporter.SetMessage("Error: Method " + method + " cannot be applied to oracle " + F.name)
        result = reporter_to_py_dict(reporter[0])
        del reporter
        return result

    result = reporter_to_py_dict(sol)
    del reporter
    return result
