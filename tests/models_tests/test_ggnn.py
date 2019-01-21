from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.ggnn import GGNN
from chainer_chemistry.utils.extend import extend_node, extend_adj
from chainer_chemistry.utils.permutation import permute_adj
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
out_dim = 4
batch_size = 2
num_edge_type = 3


@pytest.fixture
def model():
    return GGNN(out_dim=out_dim, num_edge_type=num_edge_type)


@pytest.fixture
def data():
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
    ).astype(numpy.int32)
    adj_data = numpy.random.randint(
        0, high=2, size=(batch_size, num_edge_type, atom_size, atom_size)
    ).astype(numpy.float32)
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, out_dim)).astype(numpy.float32)
    return atom_data, adj_data, y_grad


def check_forward(model, atom_data, adj_data):
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(model, data):
    atom_data, adj_data = data[0], data[1]
    check_forward(model, atom_data, adj_data)


@pytest.mark.gpu
def test_forward_gpu(model, data):
    atom_data, adj_data = cuda.to_gpu(data[0]), cuda.to_gpu(data[1])
    model.to_gpu()
    check_forward(model, atom_data, adj_data)


def test_backward_cpu(model, data):
    atom_data, adj_data, y_grad = data
    gradient_check.check_backward(model, (atom_data, adj_data), y_grad,
                                  atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
def test_backward_gpu(model, data):
    atom_data, adj_data, y_grad = [cuda.to_gpu(d) for d in data]
    model.to_gpu()
    gradient_check.check_backward(model, (atom_data, adj_data), y_grad,
                                  atol=1e-3, rtol=1e-3)


def test_forward_cpu_graph_invariant(model, data):
    atom_data, adj_data = data[0], data[1]
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index)
    permute_adj_data = permute_adj(adj_data, permutation_index)
    permute_y_actual = cuda.to_cpu(model(
        permute_atom_data, permute_adj_data).data)
    assert numpy.allclose(y_actual, permute_y_actual, rtol=1e-5, atol=1e-6)


def test_forward_cpu_input_size_invariant(model, data):
    atom_data, adj_data = data[0], data[1]
    is_real_node = numpy.ones(atom_data.shape, dtype=numpy.float32)
    y_actual = cuda.to_cpu(model(atom_data, adj_data, is_real_node).data)

    atom_data_ex = extend_node(atom_data, out_size=8)
    adj_data_ex = extend_adj(adj_data, out_size=8)
    is_real_node_ex = extend_node(is_real_node, out_size=8)
    y_actual_ex = cuda.to_cpu(model(
        atom_data_ex, adj_data_ex, is_real_node_ex).data)
    assert numpy.allclose(y_actual, y_actual_ex, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
