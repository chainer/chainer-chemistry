import numpy
import pytest

from chainer import cuda

from chainer_chemistry.links.update.cgcnn_update import CGCNNUpdate


# node_size_list means the first moleculae has three nodes,
# and the seconde molecule has five nodes
node_size_list = [3, 5]
max_num_nbr = 6
node_feature_dim = 10
edge_feature_dim = 15
out_dim = node_feature_dim
batch_size = 2


@pytest.fixture
def update():
    return CGCNNUpdate(n_site_features=node_feature_dim)


@pytest.fixture
def data():
    if len(node_size_list) != batch_size:
        raise ValueError("Invalid fixture data for CGCNN")

    numpy.random.seed(0)
    total_node_size = sum(node_size_list)
    atom_feat = numpy.random.rand(total_node_size,
                                  node_feature_dim).astype(numpy.float32)
    nbr_feat = numpy.random.rand(total_node_size, max_num_nbr,
                                 edge_feature_dim).astype(numpy.float32)
    # nbr_idx
    curr_idx = 0
    nbr_idx = []
    for val in node_size_list:
        for _ in range(val):
            max_val = curr_idx + val
            nbr_idx.append(numpy.random.randint(curr_idx,
                                                max_val, max_num_nbr))
        curr_idx += val
    nbr_idx = numpy.array(nbr_idx, dtype=numpy.int32)

    y_grad = numpy.random.uniform(-1, 1,
                                  (batch_size, out_dim)).astype(numpy.float32)
    return atom_feat, nbr_feat, nbr_idx, y_grad


def check_forward(update, data):
    y_actual = cuda.to_cpu(update(*data).data)
    assert y_actual.shape == (sum(node_size_list), out_dim)


def test_forward_cpu(update, data):
    atom_feat, nbr_feat, nbr_idx = data[:-1]
    check_forward(update, (atom_feat, nbr_feat, nbr_idx))


@pytest.mark.gpu
def test_forward_gpu(update, data):
    input_data = [cuda.to_gpu(d) for d in data[:-1]]
    update.to_gpu()
    check_forward(update, tuple(input_data))


# def test_backward_cpu(update, data):
#     input_data, y_grad = data[0:-1], data[-1]
#     gradient_check.check_backward(update, tuple(input_data), y_grad,
#                                   atol=5e-1, rtol=1e-1)


# @pytest.mark.gpu
# def test_backward_gpu(update, data):
#     atom_data, adj_data, y_grad = [cuda.to_gpu(d) for d in data]
#     update.to_gpu()
#     gradient_check.check_backward(update, (atom_data, adj_data), y_grad,
#                                   atol=5e-1, rtol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
