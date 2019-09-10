import numpy
import networkx as nx
import scipy


def _to_networkx(filepath):
    print("Loading graph data")
    coo_adj = scipy.sparse.load_npz(filepath + "reddit_graph.npz")
    G = nx.from_scipy_sparse_matrix(coo_adj)

    print("Loading node feature and label")
    # 頂点feature, 頂点label
    reddit_data = numpy.load(filepath + "reddit_data.npz")
    G.graph['x'] = reddit_data['feature']
    G.graph['y'] = reddit_data['label']

    G.graph['label_num'] = 41
    # G = nx.convert_node_labels_to_integers(G)
    print("Finish loading graph: {}".format(filepath))
    return G


def reddit_to_networkx():
    return _to_networkx("chainer_chemistry/datasets/reddit/")
