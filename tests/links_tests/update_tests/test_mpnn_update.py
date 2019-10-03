from typing import Tuple  # NOQA

import chainer  # NOQA
from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.update.mpnn_update import EdgeNet
from chainer_chemistry.links.update.mpnn_update import MPNNUpdate

atom_size = 5
hidden_channels = 4
batch_size = 3
num_edge_type = 7


@pytest.fixture
def message():
    # type: () -> EdgeNet
    return EdgeNet(out_channels=hidden_channels)


@pytest.fixture
def update():
    # type: () -> MPNNUpdate
    return MPNNUpdate(hidden_channels=hidden_channels)


@pytest.fixture
def data():
    # type: () -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    numpy.random.seed(0)
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)).astype('i')
    adj_data = numpy.random.randint(
        0, high=2, size=(batch_size, num_edge_type, atom_size,
                         atom_size)).astype('f')
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, atom_size, hidden_channels)).astype('f')
    y_grad_ = numpy.random.uniform(
        -1, 1, (batch_size, atom_size, hidden_channels)).astype('f')
    embed = EmbedAtomID(in_size=MAX_ATOMIC_NUM, out_size=hidden_channels)
    embed_atom_data = embed(atom_data).data
    return embed_atom_data, adj_data, y_grad, y_grad_


# Test Message Function
def check_message_forward(message, atom_data, adj_data):
    # type: (EdgeNet, numpy.ndarray, numpy.ndarray) -> None
    y_actual = cuda.to_cpu(message(atom_data, adj_data).data)
    assert y_actual.shape == (batch_size, atom_size, hidden_channels * 2)


def test_message_forward_cpu(message, data):
    # type: (EdgeNet, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = data[:2]
    check_message_forward(message, atom_data, adj_data)


@pytest.mark.gpu
def test_message_forward_gpu(message, data):
    # type: (EdgeNet, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = map(cuda.to_gpu, data[:2])
    message.to_gpu()
    check_message_forward(message, atom_data, adj_data)


def test_message_backward_cpu(message, data):
    # type: (EdgeNet, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad, y_grad_ = data
    y_grad = numpy.concatenate([y_grad, y_grad_], axis=2)
    gradient_check.check_backward(
        message, (atom_data, adj_data), y_grad, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_message_backward_gpu(message, data):
    # type: (EdgeNet, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad, y_grad_ = map(cuda.to_gpu, data)
    xp = cuda.get_array_module(atom_data)
    y_grad = xp.concatenate([y_grad, y_grad_], axis=2)
    message.to_gpu()
    gradient_check.check_backward(
        message, (atom_data, adj_data), y_grad, atol=1e-1, rtol=1e-1)


# Test Update Function
def check_forward(update, atom_data, adj_data):
    # type: (MPNNUpdate, numpy.ndarray, numpy.ndarray) -> None
    y_actual = cuda.to_cpu(update(atom_data, adj_data).data)
    assert y_actual.shape == (batch_size, atom_size, hidden_channels)


def test_forward_cpu(update, data):
    # type: (MPNNUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = data[:2]
    check_forward(update, atom_data, adj_data)


@pytest.mark.gpu
def test_forward_gpu(update, data):
    # type: (MPNNUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data = map(cuda.to_gpu, data[:2])
    update.to_gpu()
    check_forward(update, atom_data, adj_data)


def check_backward(update, atom_data, adj_data, y_grad):
    # type: (MPNNUpdate, numpy.ndarray, numpy.ndarray, numpy.ndarray) -> None
    """Check gradient of MPNNUpdate.

    This function is different from other backward tests.
    Because of GRU, reset_state method has to be called explicitly
    before gradient calculation.

    Args:
        update (callable):
        atom_data (numpy.ndarray):
        adj_data (numpy.ndarray):
        y_grad (numpy.ndarray):
    """
    def f(*args, **kwargs):
        update.reset_state()
        return update(*args, **kwargs)
    gradient_check.check_backward(
        f, (atom_data, adj_data), y_grad, atol=1e-1, rtol=1e-1)


def test_backward_cpu(update, data):
    # type: (MPNNUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = data[:3]
    check_backward(update, atom_data, adj_data, y_grad)


@pytest.mark.gpu
def test_backward_gpu(update, data):
    # type: (MPNNUpdate, Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]) -> None  # NOQA
    atom_data, adj_data, y_grad = map(cuda.to_gpu, data[:3])
    update.to_gpu()
    # gradient_check.check_backward(update, (atom_data, adj_data), y_grad,
    #                               atol=1e-1, rtol=1e-1)
    check_backward(update, atom_data, adj_data, y_grad)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-x'])
