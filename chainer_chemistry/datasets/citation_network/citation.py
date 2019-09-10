import numpy
import networkx as nx


def _to_networkx(filepath):
    G = nx.Graph()
    # edge
    with open(filepath + "cora.cites") as f:
        for line in f:
            u, v = map(int, line.split())
            G.add_edge(u, v)

    # node feature, node label
    with open(filepath + "cora.content") as f:
        compressor = {}
        acc = 0
        for line in f:
            lis = line.split()
            key, val = int(lis[0]), lis[-1]
            if val in compressor:
                val = compressor[val]
            else:
                compressor[val] = acc
                val = acc
                acc += 1
            G.nodes[key]['x'] = numpy.array(lis[1:-1], dtype=numpy.float32)
            G.nodes[key]['y'] = val
        G.graph['label_num'] = acc
    G = nx.convert_node_labels_to_integers(G)
    print("Finished loading graph: {}".format(filepath))
    return G


def cora_to_networkx():
    return _to_networkx("chainer_chemistry/datasets/citation_network/cora/")
