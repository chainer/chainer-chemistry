from chainer import cuda
from chainer import gradient_check
import numpy
import pytest

from chainerchem import links

in_size = 3
channel_size = 5
out_size = 4
batch_size = 2

@pytest.fixture
def model():
    return links.EmbedAtomID(in_size, out_size)


@pytest.fixture
def data():
    x_data = numpy.random.randint(
        in_size, size=(batch_size, channel_size)).astype(numpy.int32)
    y_grad = numpy.random.uniform(
        -1, 1, (batch_size, out_size, channel_size)).astype(numpy.float32)
    return x_data, y_grad


def check_forward(model, x_data):
    def forward(W, x):
        y = W[x]
        return numpy.transpose(y, (0, 2, 1))

    y_expect = forward(cuda.to_cpu(model.W.data),
                       cuda.to_cpu(x_data))
    y_actual = cuda.to_cpu(model(x_data).data)
    numpy.testing.assert_equal(y_actual, y_expect)


def test_forward_cpu(model, data):
    x_data = data[0]
    check_forward(model, x_data)


@pytest.mark.gpu
def test_forward_gpu(model, data):
    x_data = cuda.to_gpu(data[0])
    model.to_gpu()
    check_forward(model, x_data)


def test_backward_cpu(model, data):
    x_data, y_grad = data
    gradient_check.check_backward(model, x_data, y_grad, model.W,
                                  atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
def test_backward_gpu(model, data):
    x_data, y_grad = [cuda.to_gpu(d) for d in data]
    model.to_gpu()
    gradient_check.check_backward(model, x_data, y_grad, model.W,
                                  atol=1e-3, rtol=1e-3)
