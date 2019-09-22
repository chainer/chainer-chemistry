from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.models.cgcnn import CGCNN


# node_size_list means the first moleculae has three nodes,
# and the seconde molecule has five nodes
node_size_list = [3, 5]
max_num_nbr = 6
node_feature_dim = 5
edge_feature_dim = 10
out_dim = 4
batch_size = 2


@pytest.fixture
def model():
    return CGCNN(out_dim=out_dim)


@pytest.fixture
def data():
    if len(node_size_list) != batch_size:
        raise ValueError("Invalid fixture data for CGCNN")

    numpy.random.seed(0)
    total_node_size = sum(node_size_list)
    # one-hot vector
    atom_feat = numpy.random.choice(
        [0, 1], (total_node_size, node_feature_dim)).astype(numpy.float32)
    nbr_feat = numpy.random.rand(total_node_size, max_num_nbr,
                                 edge_feature_dim).astype(numpy.float32)
    # atom_idx & nbr_idx
    curr_idx = 0
    atom_idx = []
    nbr_idx = []
    for val in node_size_list:
        atom_idx.append(numpy.arange(curr_idx, val))
        for _ in range(val):
            max_val = curr_idx + val
            nbr_idx.append(numpy.random.randint(curr_idx,
                                                max_val, max_num_nbr))
        curr_idx += val
    atom_idx = numpy.asarray(atom_idx)
    nbr_idx = numpy.array(nbr_idx, dtype=numpy.int32)

    y_grad = numpy.random.uniform(-1, 1,
                                  (batch_size, out_dim)).astype(numpy.float32)
    return atom_feat, nbr_feat, atom_idx, nbr_idx, y_grad


def check_forward(model, data):
    y_actual = cuda.to_cpu(model(*data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(model, data):
    atom_feat, nbr_feat, atom_idx, nbr_idx = data[:-1]
    check_forward(model, (atom_feat, nbr_feat, atom_idx, nbr_idx))


@pytest.mark.gpu
def test_forward_gpu(model, data):
    input_data = [cuda.to_gpu(d) for d in data[:-1]]
    model.to_gpu()
    check_forward(model, tuple(input_data))


# def test_backward_cpu(model, data):
#     input_data, y_grad = data[0:-1], data[-1]
#     gradient_check.check_backward(model, tuple(input_data), y_grad,
#                                   atol=5e-1, rtol=1e-1)


# @pytest.mark.gpu
# def test_backward_gpu(model, data):
#     atom_data, adj_data, y_grad = [cuda.to_gpu(d) for d in data]
#     model.to_gpu()
#     gradient_check.check_backward(model, (atom_data, adj_data), y_grad,
#                                   atol=5e-1, rtol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
