import numpy
import pytest


from chainer_chemistry.utils.permutation import permute_node
from chainer_chemistry.utils.permutation import permute_adj


batchsize = 1
num_node = 3
ch = 5


@pytest.mark.parametrize('x', [
    numpy.random.randint(10, size=(batchsize, num_node), dtype=numpy.int32),
    numpy.random.random(size=(batchsize, num_node))
])
def test_permute_node_2d(x):
    perm = numpy.random.permutation(num_node)
    x_perm = permute_node(x, perm)

    assert x.shape == x_perm.shape
    for i in range(num_node):
        assert numpy.allclose(x[:, perm[i]], x_perm[:, i])


@pytest.mark.parametrize('x', [
    numpy.random.randint(10, size=(batchsize, num_node, ch),
                         dtype=numpy.int32),
    numpy.random.random(size=(batchsize, num_node, ch))
])
@pytest.mark.parametrize('axis', [-1, -2, 1, 2])
def test_permute_node_3d(x, axis):
    perm = numpy.random.permutation(x.shape[axis])
    x_perm = permute_node(x, perm, axis=axis)

    assert x.shape == x_perm.shape
    if axis == -1 or axis == 2:
        for i in range(num_node):
            assert numpy.allclose(x[:, :, perm[i]], x_perm[:, :, i])
    else:
        for i in range(num_node):
            assert numpy.allclose(x[:, perm[i], :], x_perm[:, i, :])


@pytest.mark.parametrize('adj', [
    numpy.random.randint(10, size=(batchsize, num_node, num_node),
                         dtype=numpy.int32),
    numpy.random.randint(10, size=(batchsize, ch, num_node, num_node),
                         dtype=numpy.int32)
])
def test_permute_adj(adj):
    perm = numpy.random.permutation(num_node)
    adj_perm = permute_adj(adj, perm)

    assert adj.shape == adj_perm.shape
    # print('perm', perm)
    # print('adj, adj_perm', adj, adj_perm)
    for i in range(num_node):
        for j in range(num_node):
            assert numpy.allclose(
                adj[..., perm[i], perm[j]], adj_perm[..., i, j])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
