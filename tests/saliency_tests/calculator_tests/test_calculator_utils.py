import numpy
import pytest

from chainer_chemistry.saliency.calculator.calculator_utils import GaussianNoiseSampler  # NOQA


@pytest.mark.parametrize('mode', ['relative', 'absolute'])
def test_gaussian_noise_sampler(mode):
    shape = (3, 4, 5)
    target_array = numpy.random.uniform(0, 1, shape)
    sampler = GaussianNoiseSampler(mode=mode, scale=0.15)
    noise = sampler.sample(target_array)
    assert noise.shape == shape


def test_gaussian_noise_sampler_assert_raises():
    shape = (3, 4, 5)
    target_array = numpy.random.uniform(0, 1, shape)
    with pytest.raises(ValueError):
        sampler = GaussianNoiseSampler(mode='invalid_mode', scale=0.15)
        sampler.sample(target_array)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
