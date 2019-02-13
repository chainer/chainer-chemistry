import os

import numpy
import pytest
import scipy.stats
from chainer import serializers, Variable, cuda, testing

from chainer_chemistry.links.scaler.flow_scaler import FlowScaler


@testing.with_requires('chainer>=5.0.0')
def test_flow_scaler_transform_uniform():
    x = numpy.random.uniform(50, 100, size=100).astype(numpy.float32)

    scaler = FlowScaler(5)
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    assert scipy.stats.kstest(x_scaled, 'norm').pvalue > 0.05


@testing.with_requires('chainer>=5.0.0')
def test_flow_scaler_transform_mix_gaussian():
    plus = numpy.random.binomial(n=1, p=0.6, size=100).astype(numpy.float32)
    x = plus * numpy.random.normal(10, 5, size=100).astype(numpy.float32)
    x += (1 - plus) * numpy.random.normal(
        -10, 5, size=100).astype(numpy.float32)

    scaler = FlowScaler(5)
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    assert scipy.stats.kstest(x_scaled, 'norm').pvalue > 0.05


@testing.with_requires('chainer>=5.0.0')
def test_flow_scaler_transform_variable():
    x = numpy.random.uniform(50, 100, size=100).astype(numpy.float32)
    xvar = Variable(x)
    scaler = FlowScaler(5)
    scaler.fit(xvar)
    x_scaled = scaler.transform(xvar)

    assert isinstance(x_scaled, Variable)
    assert scipy.stats.kstest(x_scaled.array, 'norm').pvalue > 0.05


@testing.with_requires('chainer>=5.0.0')
@pytest.mark.gpu
def test_flow_scaler_transform_gpu():
    x = numpy.random.uniform(50, 100, size=100).astype(numpy.float32)

    scaler = FlowScaler(5)
    scaler.to_gpu()
    x = cuda.to_gpu(x)
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    assert isinstance(x_scaled, cuda.cupy.ndarray)
    assert scipy.stats.kstest(cuda.to_cpu(x_scaled), 'norm').pvalue > 0.05


@testing.with_requires('chainer>=5.0.0')
def test_flow_scaler_serialize(tmpdir):
    x = numpy.random.uniform(50, 100, size=100).astype(numpy.float32)
    scaler = FlowScaler(5)
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    scaler_filepath = os.path.join(str(tmpdir), 'scaler.npz')
    serializers.save_npz(scaler_filepath, scaler)

    scaler2 = FlowScaler(5)
    serializers.load_npz(scaler_filepath, scaler2)
    x_scaled2 = scaler2.transform(x)

    assert numpy.allclose(scaler.W1.array, scaler2.W1.array)
    assert numpy.allclose(scaler.b1.array, scaler2.b1.array)
    assert numpy.allclose(scaler.W2.array, scaler2.W2.array)
    assert numpy.allclose(scaler.b2.array, scaler2.b2.array)
    assert numpy.allclose(x_scaled, x_scaled2)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
