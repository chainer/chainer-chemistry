import numpy
import pytest

from chainer_chemistry.utils.extend import extend_node, extend_adj  # NOQA


batchsize = 2
num_node = 3
ch = 5

x_2d = numpy.arange(batchsize * num_node).reshape(
    (batchsize, num_node))
x_3d = numpy.arange(batchsize * num_node * ch).reshape(
    (batchsize, num_node, ch))
adj_3d = numpy.arange(batchsize * num_node * num_node).reshape(
    (batchsize, num_node, num_node))


@pytest.mark.parametrize('x', [x_2d, x_2d.astype(numpy.float32)])
def test_extend_node_2d(x):
    x_extended = extend_node(x, out_size=6)
    x_expected = numpy.array([[0, 1, 2, 0, 0, 0],
                              [3, 4, 5, 0, 0, 0]], dtype=x.dtype)

    print('x type', x_extended.dtype)
    assert x_extended.shape == (batchsize, 6)
    assert x_extended.dtype == x.dtype
    assert numpy.array_equal(x_extended, x_expected)


@pytest.mark.parametrize('x', [x_3d, x_3d.astype(numpy.float32)])
@pytest.mark.parametrize('axis', [-1, 2])
def test_extend_node_3d(x, axis):
    x_extended = extend_node(x, out_size=6, axis=axis)
    x_expected = numpy.array([
        [[0, 1, 2, 3, 4, 0],
         [5, 6, 7, 8, 9, 0],
         [10, 11, 12, 13, 14, 0]],
        [[15, 16, 17, 18, 19, 0],
         [20, 21, 22, 23, 24, 0],
         [25, 26, 27, 28, 29, 0]]])

    assert x_extended.shape == (batchsize, num_node, 6)
    assert x_extended.dtype == x.dtype
    assert numpy.array_equal(x_extended, x_expected)


def test_extend_node_assert_raises():
    with pytest.raises(ValueError):
        extend_node(x_2d, out_size=1)


@pytest.mark.parametrize('adj', [adj_3d, adj_3d.astype(numpy.float32)])
def test_extend_adj(adj):
    adj_extended = extend_adj(adj, out_size=6)
    assert adj_extended.shape == (batchsize, 6, 6)
    assert adj_extended.dtype == adj.dtype
    assert numpy.array_equal(adj_extended[:, :num_node, :num_node], adj)
    assert numpy.alltrue(adj_extended[:, num_node:, :] == 0)
    assert numpy.alltrue(adj_extended[:, :, num_node:] == 0)


def test_extend_adj_assert_raises():
    with pytest.raises(ValueError):
        extend_adj(adj_3d, out_size=1)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
