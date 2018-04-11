import networkx as nx


def check_graph(G, **kwargs):
    out = True
    if 'node_number' in kwargs:
        if len(G) is not kwargs['node_number']:
            out = False
    if 'directed' in kwargs:
        if nx.is_directed(G) is not kwargs['directed']:
            out = False
    return out


def to_edge_list(graph, **kwargs):
    capacity_name = kwargs.get('capacity', 'capacity')
    fill_value = kwargs.get('fill_value', 1)

    edge_list = []
    capacities = []
    for src, dst, data in graph.edges.data():
        cap = data.get(capacity_name, fill_value)
        edge_list.append((src, dst))
        capacities.append(float(cap))

    return edge_list, capacities
