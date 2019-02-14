# from typing import List  # NOQA
# from typing import Tuple  # NOQA

from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.mpnn import MPNN
from chainer_chemistry.utils.permutation import permute_adj
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
out_dim = 4
batch_size = 2
num_edge_type = 3


@pytest.fixture(params=[('edgenet', 'set2set'), ('edgenet', 'ggnn'),
                        ('ggnn', 'set2set'), ('ggnn', 'ggnn')])
def model(request):
    # type: (pytest.fixture.SubRequest) -> MPNN
    message_func, readout_func = request.param
    return MPNN(
        out_dim=out_dim,
        num_edge_type=num_edge_type,
        message_func=message_func,
        readout_func=readout_func)


@pytest.fixture
def data():
    # type: () -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size,
                                      atom_size)).astype(numpy.int32)
    adj_data = numpy.random.randint(
        0, high=2, size=(batch_size, num_edge_type, atom_size,
                         atom_size)).astype(numpy.float32)
    y_grad = numpy.random.uniform(-1, 1,
                                  (batch_size, out_dim)).astype(numpy.float32)
    return atom_data, adj_data, y_grad


def check_forward(model, atom_data, adj_data):
    # type: (MPNN, numpy.ndarray, numpy.ndarray) -> None
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(model, data):
    # type: (MPNN, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None
    atom_data, adj_data = data[0], data[1]
    check_forward(model, atom_data, adj_data)


@pytest.mark.gpu
def test_forward_gpu(model, data):
    # type: (MPNN, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None
    atom_data, adj_data = cuda.to_gpu(data[0]), cuda.to_gpu(data[1])
    model.to_gpu()
    check_forward(model, atom_data, adj_data)


def test_backward_cpu(model, data):
    # type: (MPNN, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None
    atom_data, adj_data, y_grad = data
    gradient_check.check_backward(
        model, (atom_data, adj_data), y_grad, atol=1e-1, rtol=1e-1)


@pytest.mark.gpu
def test_backward_gpu(model, data):
    # type: (MPNN, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None
    atom_data, adj_data, y_grad = [cuda.to_gpu(d) for d in data]
    model.to_gpu()
    gradient_check.check_backward(
        model, (atom_data, adj_data), y_grad, atol=1e-1, rtol=1e-1)


def test_forward_cpu_graph_invariant(model, data):
    # type: (MPNN, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None
    if model.message_func == 'edgenet':
        return
        # Because EdgeNet uses NN for expanding edge vector dimension,
        # graph invariant is not ensured.
    atom_data, adj_data = data[0], data[1]
    y_actual = cuda.to_cpu(model(atom_data, adj_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index)
    permute_adj_data = permute_adj(adj_data, permutation_index)
    permute_y_actual = cuda.to_cpu(
        model(permute_atom_data, permute_adj_data).data)
    assert numpy.allclose(y_actual, permute_y_actual, rtol=1e-3, atol=1e-3)


def test_invalid_message_funcion():
    # type: () -> None
    with pytest.raises(ValueError):
        MPNN(out_dim=out_dim, message_func='invalid')


def test_invalid_readout_funcion():
    # type: () -> None
    with pytest.raises(ValueError):
        MPNN(out_dim=out_dim, readout_func='invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
