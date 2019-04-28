import os

import numpy
import pytest
from chainer import serializers, Variable, cuda

from chainer_chemistry.links.scaler.standard_scaler import StandardScaler


@pytest.fixture
def data():
    x = numpy.array(
        [[0.1, 10., 0.3],
         [0.2, 20., 0.1],
         [0.3, 30., 0.],
         [0.4, 40., 0.]],
        dtype=numpy.float32)
    expect_x_scaled = numpy.array(
        [[-1.3416407, -1.3416408, 1.6329931],
         [-0.44721353, -0.4472136, 0.],
         [0.44721368, 0.4472136, -0.8164965],
         [1.3416407, 1.3416408, -0.8164965]],
        dtype=numpy.float32)
    return x, expect_x_scaled


@pytest.mark.parametrize('indices', [None, [0], [1, 2]])
def test_standard_scaler_transform(data, indices):
    x, expect_x_scaled = data
    scaler = StandardScaler()
    scaler.fit(x, indices=indices)
    x_scaled = scaler.transform(x)

    if indices is None:
        indices = numpy.arange(x.shape[1])
    for index in range(x.shape[1]):
        if index in indices:
            assert numpy.allclose(x_scaled[:, index],
                                  expect_x_scaled[:, index])
        else:
            assert numpy.allclose(x_scaled[:, index], x[:, index])


def test_standard_scaler_transform_variable(data):
    x, expect_x_scaled = data
    xvar = Variable(x)
    scaler = StandardScaler()
    scaler.fit(xvar)
    x_scaled = scaler.transform(xvar)

    assert isinstance(x_scaled, Variable)
    assert numpy.allclose(x_scaled.array, expect_x_scaled)


@pytest.mark.gpu
def test_standard_scaler_transform_gpu(data):
    x, expect_x_scaled = data
    scaler = StandardScaler()
    scaler.to_gpu()
    x = cuda.to_gpu(x)
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    assert isinstance(x_scaled, cuda.cupy.ndarray)
    assert numpy.allclose(cuda.to_cpu(x_scaled), expect_x_scaled)


@pytest.mark.parametrize('indices', [None, [0], [1, 2]])
def test_standard_scaler_inverse_transform(data, indices):
    x, expect_x_scaled = data
    scaler = StandardScaler()
    scaler.fit(x, indices=indices)
    x_inverse = scaler.inverse_transform(expect_x_scaled)

    if indices is None:
        indices = numpy.arange(x.shape[1])
    for index in range(x.shape[1]):
        if index in indices:
            assert numpy.allclose(x_inverse[:, index], x[:, index])
        else:
            assert numpy.allclose(x_inverse[:, index],
                                  expect_x_scaled[:, index])


def test_standard_scaler_fit_transform(data):
    x, expect_x_scaled = data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    assert numpy.allclose(x_scaled, expect_x_scaled)


@pytest.mark.parametrize('indices', [None, [0]])
def test_standard_scaler_serialize(tmpdir, data, indices):
    x, expect_x_scaled = data
    scaler = StandardScaler()
    scaler.fit(x, indices=indices)

    scaler_filepath = os.path.join(str(tmpdir), 'scaler.npz')
    serializers.save_npz(scaler_filepath, scaler)

    scaler2 = StandardScaler()
    serializers.load_npz(scaler_filepath, scaler2)

    # print('scaler2 attribs:', scaler2.mean, scaler2.std, scaler2.indices)
    assert numpy.allclose(scaler.mean, scaler2.mean)
    assert numpy.allclose(scaler.std, scaler2.std)
    assert scaler.indices == scaler2.indices


def test_standard_scaler_assert_raises():
    x = numpy.array([[0.1, 0.2, 0.3], [0.5, 0.3, 0.1]],
                    dtype=numpy.float32)
    scaler = StandardScaler()

    # call transform before fit raises error
    with pytest.raises(AttributeError):
        scaler.transform(x)
    with pytest.raises(AttributeError):
        scaler.inverse_transform(x)


def test_standard_scaler_transform_zero_std():
    x = numpy.array([[1, 2], [1, 2], [1, 2]], dtype=numpy.float32)
    expect_x_scaled = numpy.array([[0, 0], [0, 0], [0, 0]],
                                  dtype=numpy.float32)
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    assert numpy.allclose(x_scaled, expect_x_scaled)


def test_standard_scaler_forward(data):
    # test `forward` and `__call__` method.
    indices = [0]
    x, expect_x_scaled = data
    scaler = StandardScaler()
    scaler.fit(x, indices=indices)
    x_scaled_transform = scaler.transform(x)
    x_scaled_forward = scaler.forward(x)
    x_scaled_call = scaler(x)

    assert numpy.allclose(x_scaled_transform, x_scaled_forward)
    assert numpy.allclose(x_scaled_transform, x_scaled_call)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
