import numpy
import networkx as nx


def _to_networkx(filepath, name):
    G = nx.Graph()
    # node feature, node label
    with open("{}{}.content".format(filepath, name)) as f:
        compressor = {}
        acc = 0
        for line in f:
            lis = line.split()
            key, val = lis[0], lis[-1]
            if val in compressor:
                val = compressor[val]
            else:
                compressor[val] = acc
                val = acc
                acc += 1
            G.add_node(key,
                       x=numpy.array(lis[1:-1], dtype=numpy.float32),
                       y=val)
            # G.nodes[key]['x'] = numpy.array(lis[1:-1], dtype=numpy.float32)
            # G.nodes[key]['y'] = val
        G.graph['label_num'] = acc

    # edge
    with open("{}{}.cites".format(filepath, name)) as f:
        for line in f:
            u, v = line.split()
            if u not in G.nodes.keys():
                print("Warning: {} does not appear in {}{}.content".format(
                    u, filepath, name))
            elif v not in G.nodes.keys():
                print("Warning: {} does not appear in {}{}.content".format(
                    v, filepath, name))
            else:
                G.add_edge(u, v)

    G = nx.convert_node_labels_to_integers(G)
    print("Finished loading graph: {}".format(filepath))
    print("number of nodes: {}, number of edges: {}".format(
        G.number_of_nodes(), G.number_of_edges()
    ))
    return G


def cora_to_networkx():
    return _to_networkx(
        "chainer_chemistry/datasets/citation_network/cora/",
        "cora"
    )


def citeseer_to_network():
    return _to_networkx(
        "chainer_chemistry/datasets/citation_network/citeseer/",
        "citeseer"
    )
