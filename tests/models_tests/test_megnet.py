from chainer import cuda
from chainer import gradient_check
import numpy
import pytest


from chainer_chemistry.models.megnet import MEGNet


# node_size_list means the first moleculae has six nodes,
# and the seconde molecule has four nodes
node_size_list = [6, 4]
# edge_size_list means the first moleculae has eight edges,
# and the seconde molecule has four edges
edge_size_list = [8, 4]
node_feature_dim = 5
edge_feature_dim = 10
global_feature_dim = 2
out_dim = 4
batch_size = 2


@pytest.fixture
def model():
    return MEGNet(out_dim=out_dim)


@pytest.fixture
def data():
    if len(node_size_list) != batch_size or len(edge_size_list) != batch_size:
        raise ValueError("In valid seed data for MEGNet")

    numpy.random.seed(0)
    total_node_size = sum(node_size_list)
    total_edge_size = sum(edge_size_list)
    atom_feat = numpy.random.rand(total_node_size,
                                  node_feature_dim).astype(numpy.float32)
    pair_feat = numpy.random.rand(total_edge_size,
                                  edge_feature_dim).astype(numpy.float32)
    global_feat = numpy.random.rand(batch_size,
                                    global_feature_dim).astype(numpy.float32)

    # atom idx
    atom_idx = numpy.hstack([[i] * node_size_list[i]
                             for i in range(batch_size)]).astype(numpy.int32)
    # pair idx
    pair_idx = numpy.hstack([[i] * edge_size_list[i]
                             for i in range(batch_size)]).astype(numpy.int32)
    # create start and end idx
    edge_idx = []
    acc_node_size = [sum(node_size_list[:i+1]) for i in range(batch_size)]
    low = numpy.roll(acc_node_size + [0], 1)[0:batch_size+1]
    high = numpy.array(acc_node_size)
    for i in range(batch_size):
        idx = [numpy.random.choice(numpy.arange(low[i], high[i]), 2,
                                   replace=False)
               for _ in range(edge_size_list[i])]
        edge_idx.extend(idx)

    start_idx = numpy.array(edge_idx, dtype=numpy.int32)[:, 0]
    end_idx = numpy.array(edge_idx, dtype=numpy.int32)[:, 1]

    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, out_dim)).astype(numpy.float32)

    return atom_feat, pair_feat, global_feat, \
        atom_idx, pair_idx, start_idx, end_idx, y_grad


def check_forward(model, data):
    y_actual = cuda.to_cpu(model(*data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(model, data):
    atom_feat, pair_feat, global_feat, \
        atom_idx, pair_idx, start_idx, end_idx = data[:-1]
    check_forward(model, (atom_feat, pair_feat, global_feat, atom_idx,
                          pair_idx, start_idx, end_idx))


@pytest.mark.gpu
def test_forward_gpu(model, data):
    input_data = [cuda.to_gpu(d) for d in data[:-1]]
    model.to_gpu()
    check_forward(model, tuple(input_data))


def test_backward_cpu(model, data):
    input_data, y_grad = data[0:-1], data[-1]
    gradient_check.check_backward(model, tuple(input_data), y_grad,
                                  atol=5e-1, rtol=1e-1)


@pytest.mark.gpu
def test_backward_gpu(model, data):
    data = [cuda.to_gpu(d) for d in data]
    input_data, y_grad = data[0:-1], data[-1]
    model.to_gpu()
    gradient_check.check_backward(model, tuple(input_data), y_grad,
                                  atol=5e-1, rtol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
