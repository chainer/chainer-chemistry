import numpy
import pytest

import chainer
from chainer.links import Linear

from chainer_chemistry.link_hooks import VariableMonitorLinkHook
from chainer_chemistry.saliency.calculator.integrated_gradients_calculator import IntegratedGradientsCalculator  # NOQA


class DummyModel(chainer.Chain):
    def __init__(self):
        super(DummyModel, self).__init__()
        with self.init_scope():
            self.l1 = Linear(
                3, 1, initialW=numpy.array([[1, 3, 2]]),
                nobias=True)

    def forward(self, x):
        return self.l1(x)


def test_integrated_gradient_calculator():
    model = DummyModel()
    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)
    calculator = IntegratedGradientsCalculator(model, steps=3)
    saliency = calculator.compute(x)
    # gradient is always [1, 3, 2] * (input - base) is [1, 5, 8]
    assert numpy.allclose(saliency, numpy.array([[1, 15, 16]]))


def test_integrated_gradient_calculator_target_extractor():
    model = DummyModel()
    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)
    calculator = IntegratedGradientsCalculator(
        model, steps=4,
        target_extractor=VariableMonitorLinkHook(model.l1, timing='pre'))
    saliency = calculator.compute(x)
    # gradient is always [1, 3, 2] * (input - base) is [1, 5, 8]
    assert numpy.allclose(saliency, numpy.array([[[1, 15, 16]]]))
    assert saliency.shape == (1, 1, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
