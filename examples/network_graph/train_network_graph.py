import numpy
from chainer import optimizers, functions
from chainer_chemistry.datasets.citation_network.citation import cora_to_networkx, citeseer_to_network  # NOQA
from chainer_chemistry.dataset.networkx_preprocessors.base_networkx import BaseSparseNetworkx  # NOQA
from chainer_chemistry.dataset.networkx_preprocessors.gin_networkx import GINSparseNetworkx  # NOQA
from chainer_chemistry.models.gin import GINSparse


def generate_random_mask(n, train_num, seed=0):
    numpy.random.seed(seed)
    mask = numpy.zeros(n, dtype=bool)
    mask[:train_num] = True
    numpy.random.shuffle(mask)
    return mask, numpy.logical_not(mask)  # (train_mask, val_mask)


if __name__ == '__main__':
    # networkx_graph = cora_to_networkx()
    networkx_graph = citeseer_to_network()
    data = GINSparseNetworkx().construct_sparse_data(networkx_graph)
    print('label num: {}'.format(data.label_num))
    gin = GINSparse(out_dim=None, node_embedding=True,
                    out_channels=data.label_num, n_update_layers=2)
    train_mask, val_mask = generate_random_mask(data.n_nodes, 800)
    optimizer = optimizers.Adam().setup(gin)
    for epoch in range(500):
        y = gin(data)
        train_loss = functions.softmax_cross_entropy(
            y[train_mask], data.y[train_mask])
        val_loss = functions.softmax_cross_entropy(
            y[val_mask], data.y[val_mask])
        print("epoch: {}, train loss: {}, val loss: {}".format(
            epoch, train_loss, val_loss))
        gin.cleargrads()
        train_loss.backward()
        optimizer.update()
