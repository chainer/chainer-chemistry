import numpy
import pytest

from chainer import cuda

from chainer_chemistry.links.readout.cgcnn_readout import CGCNNReadout


# node_size_list means the first moleculae has three nodes,
# and the seconde molecule has five nodes
node_size_list = [3, 5]
node_feature_dim = 32
out_dim = 4
batch_size = 2


@pytest.fixture
def readout():
    return CGCNNReadout(out_dim=out_dim)


@pytest.fixture
def data():
    if len(node_size_list) != batch_size:
        raise ValueError("Invalid fixture data for CGCNN")

    numpy.random.seed(0)
    total_node_size = sum(node_size_list)
    # atom_feat
    atom_feat = numpy.random.rand(
        total_node_size, node_feature_dim).astype(numpy.float32)
    # atom_idx
    curr_idx = 0
    atom_idx = []
    for val in node_size_list:
        atom_idx.append(numpy.arange(curr_idx, val))
        curr_idx += val
    atom_idx = numpy.asarray(atom_idx)

    y_grad = numpy.random.uniform(-1, 1,
                                  (batch_size, out_dim)).astype(numpy.float32)
    return atom_feat, atom_idx, y_grad


def check_forward(readout, data):
    y_actual = cuda.to_cpu(readout(*data).data)
    assert y_actual.shape == (batch_size, out_dim)


def test_forward_cpu(readout, data):
    atom_feat, atom_idx = data[:-1]
    check_forward(readout, (atom_feat, atom_idx))


@pytest.mark.gpu
def test_forward_gpu(readout, data):
    atom_feat, atom_idx, _ = data
    # atom_idx is list format... use numpy array
    input_data = (cuda.to_gpu(atom_feat), atom_idx)
    readout.to_gpu()
    check_forward(readout, tuple(input_data))


# def test_backward_cpu(readout, data):
#     input_data, y_grad = data[0:-1], data[-1]
#     gradient_check.check_backward(readout, tuple(input_data), y_grad,
#                                   atol=5e-1, rtol=1e-1)


# @pytest.mark.gpu
# def test_backward_gpu(readout, data):
#     data = [cuda.to_gpu(d) for d in data]
#     input_data, y_grad = data[0:-1], data[-1]
#     readout.to_gpu()
#     gradient_check.check_backward(readout, tuple(input_data), y_grad,
#                                   atol=5e-1, rtol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
