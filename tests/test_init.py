import pkg_resources

import chainer_chemistry
import pytest


def test_version():
    expect = pkg_resources.get_distribution('chainer_chemistry').version
    actual = chainer_chemistry.__version__
    assert expect == actual


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
