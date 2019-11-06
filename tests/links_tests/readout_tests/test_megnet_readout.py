from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.links.readout.megnet_readout import MEGNetReadout


max_node_num = 6
max_edge_num = 10
# This value is the same as the atom and pair feature dimension
in_channels = 10
global_feature_dim = 5
out_dim = 4
batch_size = 2


@pytest.fixture
def readout():
    return MEGNetReadout(in_channels=in_channels, out_dim=out_dim)


@pytest.fixture
def data():
    numpy.random.seed(0)
    atom_feat = numpy.random.rand(batch_size, max_node_num,
                                  in_channels).astype(numpy.float32)
    pair_feat = numpy.random.rand(batch_size, max_edge_num,
                                  in_channels).astype(numpy.float32)
    global_feat = numpy.random.rand(batch_size,
                                    global_feature_dim).astype(numpy.float32)
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, out_dim)).astype(numpy.float32)

    return atom_feat, pair_feat, global_feat, y_grad


def check_forward(readout, data):
    y_actual = cuda.to_cpu(readout(*data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(readout, data):
    atom_feat, pair_feat, global_feat = data[:-1]
    check_forward(readout, (atom_feat, pair_feat, global_feat))


@pytest.mark.gpu
def test_forward_gpu(readout, data):
    input_data = [cuda.to_gpu(d) for d in data[:-1]]
    readout.to_gpu()
    check_forward(readout, tuple(input_data))


def test_backward_cpu(readout, data):
    input_data, y_grad = data[0:-1], data[-1]
    gradient_check.check_backward(readout, tuple(input_data), y_grad,
                                  atol=5e-1, rtol=1e-1)


@pytest.mark.gpu
def test_backward_gpu(readout, data):
    data = [cuda.to_gpu(d) for d in data]
    input_data, y_grad = data[0:-1], data[-1]
    readout.to_gpu()
    gradient_check.check_backward(readout, tuple(input_data), y_grad,
                                  atol=5e-1, rtol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
