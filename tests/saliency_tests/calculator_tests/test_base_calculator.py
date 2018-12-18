import numpy
import pytest

import chainer
from chainer.links import Linear

from chainer_chemistry.link_hooks import is_link_hooks_available
if is_link_hooks_available:
    from chainer_chemistry.link_hooks import VariableMonitorLinkHook
    from chainer_chemistry.saliency.calculator import GaussianNoiseSampler
    from chainer_chemistry.saliency.calculator.base_calculator import BaseCalculator  # NOQA

    class DummyCalculator(BaseCalculator):
        """Dummy calculator which returns target_var"""

        def _compute_core(self, *inputs):
            self.model(*inputs)
            return self.get_target_var(inputs)


class DummyModel(chainer.Chain):
    def __init__(self):
        super(DummyModel, self).__init__()
        with self.init_scope():
            self.l1 = Linear(
                3, 1, initialW=numpy.array([[1, 3, 2]]),
                nobias=True)
        self.h = None

    def forward(self, x):
        self.h = self.l1(x)
        out = self.h * 3
        return out


@pytest.fixture
def model():
    return DummyModel()


@pytest.mark.skipif(not is_link_hooks_available,
                    reason='Link Hook is not available')
def test_base_calculator_compute(model):
    calculator = DummyCalculator(model)
    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)

    saliency = calculator.compute(x)
    # DummyCalculator returns `saliency` as input `x`.
    assert numpy.allclose(saliency, x)


@pytest.mark.skipif(not is_link_hooks_available,
                    reason='Link Hook is not available')
def test_base_calculator_compute_noise_sampler(model):
    calculator = DummyCalculator(model)

    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)
    saliency = calculator.compute(x, M=2, noise_sampler=GaussianNoiseSampler())
    assert saliency.shape == (2, 3)
    # noise is added, should be different from original input
    assert not numpy.allclose(saliency[0], x)
    assert not numpy.allclose(saliency[1], x)


@pytest.mark.skipif(not is_link_hooks_available,
                    reason='Link Hook is not available')
def test_base_calculator_compute_target_extractor(model):
    # It should extract `target_var` as after `l1`, which is `model.h`.
    calculator = DummyCalculator(
        model, target_extractor=VariableMonitorLinkHook(model.l1))

    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)
    saliency = calculator.compute(x)
    assert numpy.allclose(saliency, model.h.array)


@pytest.mark.skipif(not is_link_hooks_available,
                    reason='Link Hook is not available')
def test_base_calculator_aggregate():
    model = DummyModel()
    calculator = DummyCalculator(model)

    saliency = numpy.array([[-1, -1, -1], [2, 2, 2]], dtype=numpy.float32)
    saliency_raw = calculator.aggregate(saliency, method='raw', ch_axis=None)
    assert numpy.allclose(saliency_raw,
                          numpy.array([[0.5, 0.5, 0.5]], dtype=numpy.float32))
    saliency_abs = calculator.aggregate(saliency, method='abs', ch_axis=None)
    assert numpy.allclose(saliency_abs,
                          numpy.array([[1.5, 1.5, 1.5]], dtype=numpy.float32))
    saliency_square = calculator.aggregate(saliency, method='square',
                                           ch_axis=None)
    assert numpy.allclose(saliency_square,
                          numpy.array([[2.5, 2.5, 2.5]], dtype=numpy.float32))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
