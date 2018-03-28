import numpy as np

cdef extern from "core/python/pyarray_symbol_gencut.h":
    pass

cimport cython
cimport numpy as np
np.import_array()

from libcpp cimport bool as bool_t


cdef extern from "core/python/gencut.h":
    cdef np.ndarray c_graph_prox_grid_1d "submodular::graph_prox_grid_1d"(np.ndarray, float, bool_t, float) except +

    cdef np.ndarray c_graph_prox_grid_2d "submodular::graph_prox_grid_1d"(np.ndarray, float, bool_t, float) except +

    cdef np.ndarray c_graph_prox "submodular::graph_prox"(np.ndarray, float, object, bool_t, float) except +


def _graph_prox_grid_1d(np.ndarray y, float alpha, bool_t directed, **kwargs):
    tol = kwargs.get('tol', 1e-8)
    return c_graph_prox_grid_1d(y, alpha, directed, tol)


def _graph_prox_grid_2d(np.ndarray y, float alpha, bool_t directed, **kwargs):
    # todo: type check?
    tol = kwargs.get('tol', 1e-8)
    return c_graph_prox_grid_2d(y, alpha, directed, tol)


def _graph_prox(np.ndarray y, float alpha, object edge_list, bool_t directed, **kwargs):
    tol = kwargs.get('tol', 1e-8)
    return c_graph_prox(y, alpha, edge_list, directed, tol)
