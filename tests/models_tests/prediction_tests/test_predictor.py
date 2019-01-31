from typing import Tuple  # NOQA

from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.ggnn import GGNN
from chainer_chemistry.models.mlp import MLP
from chainer_chemistry.models.prediction import GraphConvPredictor
from chainer_chemistry.utils.permutation import permute_adj
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
class_num = 7
n_unit = 11
out_dim = 4
batch_size = 2
num_edge_type = 3


@pytest.fixture
def model():
    # type: () -> GraphConvPredictor
    mlp = MLP(out_dim=class_num, hidden_dim=n_unit)
    ggnn = GGNN(
        out_dim=out_dim, hidden_dim=n_unit, num_edge_type=num_edge_type)
    return GraphConvPredictor(ggnn, mlp)


@pytest.fixture
def data():
    # type: () -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)).astype('i')
    adj_data = numpy.random.randint(
        0, high=2, size=(batch_size, num_edge_type, atom_size,
                         atom_size)).astype('f')
    y_grad = numpy.random.uniform(-1, 1, (batch_size, class_num)).astype('f')
    return atom_data, adj_data, y_grad


def check_forward(model, atom_data, adj_data):
    # type: (GraphConvPredictor, numpy.ndarray, numpy.ndarray) -> None
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)
    assert y_actual.shape == (batch_size, class_num)


def test_forward_cpu(model, data):
    # type: (GraphConvPredictor, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    check_forward(model, *data[:2])


@pytest.mark.gpu
def test_forward_gpu(model, data):
    # type: (GraphConvPredictor, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = map(cuda.to_gpu, data[:2])
    model.to_gpu()
    check_forward(model, atom_data, adj_data)


def test_backward_cpu(model, data):
    # type: (GraphConvPredictor, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = data
    gradient_check.check_backward(
        model, (atom_data, adj_data), y_grad, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_backward_gpu(model, data):
    # type: (GraphConvPredictor, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = map(cuda.to_gpu, data)
    model.to_gpu()
    gradient_check.check_backward(
        model, (atom_data, adj_data), y_grad, atol=1e-4, rtol=1e-4)


def test_forward_cpu_graph_invariant(model, data):
    # type: (GraphConvPredictor, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = data[:2]
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index)
    permute_adj_data = permute_adj(adj_data, permutation_index)
    permute_y_actual = cuda.to_cpu(
        model(permute_atom_data, permute_adj_data).data)
    assert numpy.allclose(y_actual, permute_y_actual, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
