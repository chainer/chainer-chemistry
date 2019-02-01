import numpy
import scipy.stats

from chainer_chemistry.links.scaler.flow_scaler import FlowScaler


def test_flow_scaler_transform_uniform():
    x = numpy.random.uniform(50, 100, size=100).astype(numpy.float32)

    scaler = FlowScaler(5)
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    assert scipy.stats.kstest(x_scaled.reshape(-1), 'norm').pvalue > 0.05


def test_flow_scaler_transform_mix_gaussian():
    plus = numpy.random.binomial(n=1, p=0.6, size=100).astype(numpy.float32)
    x = plus * numpy.random.normal(10, 5, size=100).astype(numpy.float32)
    x += (1 - plus) * numpy.random.normal(
        -10, 5, size=100).astype(numpy.float32)

    scaler = FlowScaler(5)
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    assert scipy.stats.kstest(x_scaled.reshape(-1), 'norm').pvalue > 0.05
