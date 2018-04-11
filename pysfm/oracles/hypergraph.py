class Hypergraph:
    def __init__(self):
        self.hyperedge_list = []
        self.capacities = []

    def add_hyperedge(self, hyperedge, capacity=1.0):
        self.hyperedge_list.append(hyperedge)
        self.capacities.append(capacity)


class HypergraphCutPlusModular:
    def __init__(self, hypergraph, x):
        self.hypergraph = hypergraph
        self.x = x

    @property
    def name(self):
        return "hypergraph_cut_plus_modular"

    @property
    def data(self):
        data =  {
            'n': len(self.x),
            'hyperedge_list': self.hypergraph.hyperedge_list,
            'capacities': self.hypergraph.capacities,
            'x': self.x
        }
        return data
