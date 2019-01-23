from chainer import cuda
from chainer import functions
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.general_readout import GeneralReadout
from chainer_chemistry.utils.permutation import permute_node

atom_size = 5
hidden_dim = 7
batch_size = 2


@pytest.fixture
def readouts():
    modes = ['sum', 'max', 'summax']
    return (GeneralReadout(mode=mode) for mode in modes)


@pytest.fixture
def data():
    numpy.random.seed(0)
    atom_data = numpy.random.uniform(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size, hidden_dim)
    ).astype('f')
    y_grad = numpy.random.uniform(-1, 1, (batch_size, hidden_dim)).astype('f')
    return atom_data, y_grad


def check_forward(readout, atom_data):
    y_actual = cuda.to_cpu(readout(atom_data).data)
    if readout.mode == ('sum' and 'max'):
        assert y_actual.shape == (batch_size, hidden_dim)
    elif readout.mode == 'summax':
        assert y_actual.shape == (batch_size, hidden_dim * 2)


def test_forward_cpu(readouts, data):
    atom_data = data[0]
    for readout in readouts:
        check_forward(readout, atom_data)


@pytest.mark.gpu
def test_forward_gpu(readouts, data):
    atom_data = cuda.to_gpu(data[0])
    for readout in readouts:
        readout.to_gpu()
        check_forward(readout, atom_data)


def test_forward_cpu_assert_raises(data):
    atom_data = data[0]
    readout = GeneralReadout(mode='invalid')
    with pytest.raises(ValueError):
        cuda.to_cpu(readout(atom_data).data)


def test_backward_cpu(readouts, data):
    atom_data, y_grad = data
    for readout in readouts:
        if readout.mode == 'summax':
            y_grad = functions.concat((y_grad, y_grad), axis=1).data
        gradient_check.check_backward(
            readout, atom_data, y_grad, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_backward_gpu(readouts, data):
    atom_data, y_grad = map(cuda.to_gpu, data)
    for readout in readouts:
        readout.to_gpu()
        if readout.mode == 'summax':
            y_grad = functions.concat((y_grad, y_grad), axis=1).data
        # TODO (nakago): check why tolerance is so high.
        gradient_check.check_backward(
            readout, atom_data, y_grad, atol=1e-1, rtol=1e-1)


def test_forward_cpu_graph_invariant(readouts, data):
    atom_data = data[0]
    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index, axis=1)
    for readout in readouts:
        y_actual = cuda.to_cpu(readout(atom_data).data)
        permute_y_actual = cuda.to_cpu(readout(permute_atom_data).data)
        numpy.testing.assert_allclose(
            y_actual, permute_y_actual, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
