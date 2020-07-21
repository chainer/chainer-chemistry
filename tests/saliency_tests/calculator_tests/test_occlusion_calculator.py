import numpy
import pytest

import chainer
from chainer.links import Linear, Convolution2D  # NOQA

from chainer_chemistry.link_hooks import is_link_hooks_available
if is_link_hooks_available:
    from chainer_chemistry.link_hooks import VariableMonitorLinkHook
    from chainer_chemistry.saliency.calculator.occlusion_calculator import OcclusionCalculator  # NOQA


class DummyModel(chainer.Chain):
    def __init__(self):
        super(DummyModel, self).__init__()
        with self.init_scope():
            self.l1 = Linear(
                3, 1, initialW=numpy.array([[1, 3, 2]]),
                nobias=True)

    def forward(self, x):
        return self.l1(x)


class DummyCNNModel(chainer.Chain):
    def __init__(self):
        super(DummyCNNModel, self).__init__()
        with self.init_scope():
            self.l1 = Convolution2D(
                1, 1, ksize=3,
                initialW=numpy.ones((1, 1, 3, 3), numpy.float32), nobias=True)

    def forward(self, x):
        return self.l1(x)


@pytest.mark.skipif(not is_link_hooks_available,
                    reason='Link Hook is not available')
def test_occlusion_calculator():
    model = DummyModel()
    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)
    calculator = OcclusionCalculator(model, slide_axis=1)
    saliency = calculator.compute(x)
    assert numpy.allclose(saliency, numpy.array([[[1, 15, 16]]]))
    assert saliency.shape == (1, 1, 3)


@pytest.mark.skipif(not is_link_hooks_available,
                    reason='Link Hook is not available')
def test_occlusion_calculator_cnn():
    model = DummyCNNModel()
    # x (1, 1, 3, 3): (bs, ch, h, w)
    x = numpy.array([[[[1, 5, 8], [2, 4, 1], [3, 2, 9]]]], dtype=numpy.float32)
    calculator = OcclusionCalculator(model, slide_axis=(2, 3))
    saliency = calculator.compute(x)
    assert numpy.allclose(saliency, x)
    assert saliency.shape == (1, 1, 1, 3, 3)  # (M, bs, ch, h, w)


@pytest.mark.skipif(not is_link_hooks_available,
                    reason='Link Hook is not available')
def test_occlusion_calculator_target_extractor():
    model = DummyModel()
    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)
    calculator = OcclusionCalculator(
        model, slide_axis=1,
        target_extractor=VariableMonitorLinkHook(model.l1, timing='pre'))
    saliency = calculator.compute(x)
    assert numpy.allclose(saliency, numpy.array([[[1, 15, 16]]]))
    assert saliency.shape == (1, 1, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
