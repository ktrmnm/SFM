import networkx as nx
import numpy as np
from ..utils import to_edge_list


class STCut:
    def __init__(self, graph, s, t):
        self.graph = graph
        self.s = s
        self.t = t

    @property
    def name(self):
        return "stcut"

    @property
    def data(self):
        edge_list, capacities = to_edge_list(self.graph)
        data =  {
            'n': len(self.graph) - 2,
            's': self.s,
            't': self.t,
            'edge_list': edge_list,
            'capacities': capacities
        }
        return data


class STCutPlusModular:
    def __init__(self, graph, s, t, x):
        self.graph = graph
        self.s = s
        self.t = t
        self.x = x

    @property
    def name(self):
        return "stcut_plus_modular"

    @property
    def data(self):
        edge_list, capacities = to_edge_list(self.graph)
        data =  {
            'n': len(self.x),
            's': self.s,
            't': self.t,
            'x': self.x,
            'edge_list': edge_list,
            'capacities': capacities
        }
        return data


class CutPlusModular:
    def __init__(self, graph, x, directed=False):
        self.graph = graph
        self.x = x
        self.directed = directed

    @property
    def name(self):
        return "cut_plus_modular"

    @property
    def data(self):
        edge_list, capacities = to_edge_list(self.graph)
        data =  {
            'n': len(self.x),
            'x': self.x,
            'directed': self.directed,
            'edge_list': edge_list,
            'capacities': capacities
        }
        return data
