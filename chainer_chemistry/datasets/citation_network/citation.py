import os

import networkx as nx
import numpy
from tqdm import tqdm


def citation_to_networkx(dirpath, name):
    G = nx.Graph()
    # node feature, node label
    with open(os.path.join(dirpath, "{}.content".format(name))) as f:
        lines = f.readlines()
        compressor = {}
        acc = 0
        for line in tqdm(lines):
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
        G.graph['label_num'] = acc

    # edge
    with open(os.path.join(dirpath, "{}.cites".format(name))) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            u, v = line.split()
            if u not in G.nodes.keys():
                print("Warning: {} does not appear in {}{}.content".format(
                    u, dirpath, name))
            elif v not in G.nodes.keys():
                print("Warning: {} does not appear in {}{}.content".format(
                    v, dirpath, name))
            else:
                G.add_edge(u, v)

    G = nx.convert_node_labels_to_integers(G)
    print("Finished loading graph: {}".format(dirpath))
    print("number of nodes: {}, number of edges: {}".format(
        G.number_of_nodes(), G.number_of_edges()
    ))
    return G
