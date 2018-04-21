import chainer
from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.models.rsgcn import RSGCN
from chainer_chemistry.utils.permutation import permute_node, permute_adj

atom_size = 5
out_dim = 4
batch_size = 2


@pytest.fixture
def model():
    return RSGCN(out_dim=out_dim)


@pytest.fixture
def model_no_dropout():
    # To check backward gradient by `gradient_check`,
    # we need to skip stochastic dropout function.
    return RSGCN(out_dim=out_dim, dropout_ratio=0.)


@pytest.fixture
def data():
    atom_data = numpy.random.randint(
        0, high=MAX_ATOMIC_NUM, size=(batch_size, atom_size)
    ).astype(numpy.int32)
    # adj_data = numpy.random.randint(
    #     0, high=2, size=(batch_size, atom_size, atom_size)
    # ).astype(numpy.float32)

    # adj_data is symmetric matrix
    adj_data = numpy.random.uniform(
        0, high=1, size=(batch_size, atom_size, atom_size)
    ).astype(numpy.float32)
    adj_data = adj_data + adj_data.swapaxes(-1, -2)
    # adj_data = (adj_data > 1.5).astype(numpy.float32)

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


# TODO(nakago): check why tolerance is high
def test_backward_cpu(model_no_dropout, data):
    atom_data, adj_data, y_grad = data
    gradient_check.check_backward(
        model_no_dropout, (atom_data, adj_data), y_grad,
        params=(model_no_dropout.embed.W, ),
        atol=1e-1, rtol=1e-1, no_grads=[True, True])

    gradient_check.check_backward(
        model_no_dropout, (model_no_dropout.embed(atom_data).data, adj_data),
        y_grad, atol=1e-1, rtol=1e-1, no_grads=[False, True])


# TODO(nakago): check why tolerance is high
@pytest.mark.gpu
def test_backward_gpu(model_no_dropout, data):
    atom_data, adj_data, y_grad = [cuda.to_gpu(d) for d in data]
    model_no_dropout.to_gpu()
    gradient_check.check_backward(
        model_no_dropout, (atom_data, adj_data), y_grad,
        params=(model_no_dropout.embed.W, ),
        atol=1e-1, rtol=1e-1, no_grads=[True, True])
    gradient_check.check_backward(
        model_no_dropout, (model_no_dropout.embed(atom_data).data, adj_data),
        y_grad, atol=1e-1, rtol=1e-1, no_grads=[False, True])


def test_forward_cpu_graph_invariant(model, data):
    # This RSGCN uses dropout, so we need to forward with test mode
    # to remove stochastic calculation.
    atom_data, adj_data = data[0], data[1]
    with chainer.using_config('train', False):
        y_actual = cuda.to_cpu(model(atom_data, adj_data).data)

    permutation_index = numpy.random.permutation(atom_size)
    permute_atom_data = permute_node(atom_data, permutation_index)
    permute_adj_data = permute_adj(adj_data, permutation_index)
    with chainer.using_config('train', False):
        permute_y_actual = cuda.to_cpu(model(
            permute_atom_data, permute_adj_data).data)
    assert numpy.allclose(y_actual, permute_y_actual)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
