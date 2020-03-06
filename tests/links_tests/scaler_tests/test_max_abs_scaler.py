import os

import numpy
import pytest
from chainer import serializers, Variable, cuda  # NOQA

from chainer_chemistry.links.scaler.max_abs_scaler import MaxAbsScaler


@pytest.fixture
def data():
    x = numpy.array(
        [[0.1, 10., 0.3],
         [0.2, 20., 0.1],
         [-0.3, 30., 0.],
         [0.4, -40., 0.]],
        dtype=numpy.float32)
    expect_x_scaled = numpy.array(
        [[0.25, 0.25, 1.],
         [0.5, 0.5, 0.3333333],
         [-0.75, 0.75, 0.],
         [1., -1., 0.]],
        dtype=numpy.float32)
    return x, expect_x_scaled


@pytest.mark.parametrize('indices', [None, [0], [1, 2]])
def test_max_abs_scaler_transform(data, indices):
    x, expect_x_scaled = data
    scaler = MaxAbsScaler()
    scaler.fit(x, indices=indices)

    x_scaled = scaler.transform(x)

    if indices is None:
        indices = numpy.arange(x.shape[1])

    numpy.allclose(scaler.max_abs, numpy.array([0.4, 40, 0.3])[indices])

    for index in range(x.shape[1]):
        if index in indices:
            assert numpy.allclose(x_scaled[:, index],
                                  expect_x_scaled[:, index])
        else:
            assert numpy.allclose(x_scaled[:, index], x[:, index])


def test_max_abs_scaler_transform_variable(data):
    x, expect_x_scaled = data
    xvar = Variable(x)
    scaler = MaxAbsScaler()
    scaler.fit(xvar)
    x_scaled = scaler.transform(xvar)

    assert isinstance(x_scaled, Variable)
    assert numpy.allclose(x_scaled.array, expect_x_scaled)


@pytest.mark.gpu
def test_max_abs_scaler_transform_gpu(data):
    x, expect_x_scaled = data
    scaler = MaxAbsScaler()
    scaler.to_gpu()
    x = cuda.to_gpu(x)
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    assert isinstance(x_scaled, cuda.cupy.ndarray)
    assert numpy.allclose(cuda.to_cpu(x_scaled), expect_x_scaled)


@pytest.mark.parametrize('indices', [None, [0], [1, 2]])
def test_max_abs_scaler_inverse_transform(data, indices):
    x, expect_x_scaled = data
    scaler = MaxAbsScaler()
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


@pytest.mark.parametrize('axis', [1, 2])
def test_max_abs_scaler_3darray(data, axis):
    x, expect_x_scaled = data
    s0, s1 = x.shape
    if axis == 1:
        # feature axis is 1, insert other axis to 2nd axis
        x = numpy.broadcast_to(x[:, :, None], (s0, s1, 2))
        expect_x_scaled = numpy.broadcast_to(
            expect_x_scaled[:, :, None], (s0, s1, 2))
    elif axis == 2:
        # feature axis is 2, insert other axis to 1st axis
        x = numpy.broadcast_to(x[:, None, :], (s0, 3, s1))
        expect_x_scaled = numpy.broadcast_to(
            expect_x_scaled[:, None, :], (s0, 3, s1))
    assert x.ndim == 3
    indices = None
    scaler = MaxAbsScaler()
    scaler.fit(x, indices=indices, axis=axis)
    x_scaled = scaler.transform(x, axis=axis)
    assert x_scaled.shape == expect_x_scaled.shape
    assert numpy.allclose(x_scaled, expect_x_scaled, atol=1e-7)

    x_inverse = scaler.inverse_transform(expect_x_scaled, axis=axis)

    for index in numpy.arange(x.shape[1]):
        assert numpy.allclose(x_inverse[:, index], x[:, index], atol=1e-7)


def test_max_abs_scaler_fit_transform(data):
    x, expect_x_scaled = data
    scaler = MaxAbsScaler()
    x_scaled = scaler.fit_transform(x)
    assert numpy.allclose(x_scaled, expect_x_scaled)


# TODO(nakago): fix Chainer serializer.
# Behavior changed from numpy versioin 1.16.3.
# allow_pickle=True must be passed to numpy.load function,
# in order to load `None`.
# For now, skip test for serialize `None`.
# @pytest.mark.parametrize('indices', [None, [0]])
@pytest.mark.parametrize('indices', [[0]])
def test_max_abs_scaler_serialize(tmpdir, data, indices):
    x, expect_x_scaled = data
    scaler = MaxAbsScaler()
    scaler.fit(x, indices=indices)

    scaler_filepath = os.path.join(str(tmpdir), 'scaler.npz')
    serializers.save_npz(scaler_filepath, scaler)

    scaler2 = MaxAbsScaler()
    serializers.load_npz(scaler_filepath, scaler2)

    # print('scaler2 attribs:', scaler2.max_abs, scaler2.indices)
    assert numpy.allclose(scaler.max_abs, scaler2.max_abs)
    assert scaler.indices == scaler2.indices


def test_max_abs_scaler_assert_raises():
    x = numpy.array([[0.1, 0.2, 0.3], [0.5, 0.3, 0.1]],
                    dtype=numpy.float32)
    scaler = MaxAbsScaler()

    # call transform before fit raises error
    with pytest.raises(AttributeError):
        scaler.transform(x)
    with pytest.raises(AttributeError):
        scaler.inverse_transform(x)


def test_max_abs_scaler_transform_zero_max():
    x = numpy.array([[0, 2], [0, 2], [0, 2]], dtype=numpy.float32)
    expect_x_scaled = numpy.array([[0, 1], [0, 1], [0, 1]],
                                  dtype=numpy.float32)
    scaler = MaxAbsScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    # print('max_abs', scaler.max_abs)
    assert numpy.allclose(x_scaled, expect_x_scaled)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
