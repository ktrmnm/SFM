import numpy as np
import networkx as nx
from ._gencut import _graph_prox, _graph_prox_grid_1d, _graph_prox_grid_2d
from .utils import check_graph, to_edge_list


def graph_prox_grid_1d(y, alpha, directed=False, **kwargs):
    """
    Proximal operator of 1-dimensional grid graphs.

    TODO: document

    """
    alpha = float(alpha)
    if alpha < 0:
        raise ValueError('alpha must be non-negative.')

    return _graph_prox_grid_1d(y, alpha, directed, **kwargs)


def graph_prox_grid_2d(y, alpha, directed=False, **kwargs):
    """
    Proximal operator of 2-dimensional grid graphs.

    TODO: document

    """
    alpha = float(alpha)
    if alpha < 0:
        raise ValueError('alpha must be non-negative.')

    return _graph_prox_grid_2d(y, alpha, directed, **kwargs)


def graph_prox(y, alpha, graph=None, **kwargs):
    """
    Proximal operator of general cut function.

    TODO: document
    """
    n = len(y)

    if graph is not None:
        if kwargs.get('check_graph', False):
            if not check_graph(graph, node_number=n):
                raise ValueError('Invalid graph format')
        edge_list, capacities = to_edge_list(graph, **kwargs)
        directed = nx.is_directed(graph)
    else:
        if 'edge_list' in kwargs:
            edge_list = kwargs['edge_list']
            directed = kwargs.get('directed', True)
        else:
            raise ValueError('edge_list must be specified if graph is set to None.')

    alpha = float(alpha)
    if alpha < 0:
        raise ValueError('alpha must be non-negative.')

    return _graph_prox(y, alpha, edge_list, capacities, directed, **kwargs)
