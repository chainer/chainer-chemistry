import numpy
import networkx as nx
import scipy


def reddit_to_networkx(dirpath):
    print("Loading graph data")
    coo_adj = scipy.sparse.load_npz(dirpath + "reddit_graph.npz")
    G = nx.from_scipy_sparse_matrix(coo_adj)

    print("Loading node feature and label")
    # node feature, edge label
    reddit_data = numpy.load(dirpath + "reddit_data.npz")
    G.graph['x'] = reddit_data['feature'].astype(numpy.float32)
    G.graph['y'] = reddit_data['label'].astype(numpy.int32)

    G.graph['label_num'] = 41
    # G = nx.convert_node_labels_to_integers(G)
    print("Finish loading graph: {}".format(dirpath))
    return G
