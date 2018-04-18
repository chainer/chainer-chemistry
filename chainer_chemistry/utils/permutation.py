import numpy


def permute_node(node, permutation_index, axis=-1):
    """Permute index of `node` array

    Args:
        node (numpy.ndarray): the array whose `axis` to be permuted.
        permutation_index (numpy.ndarray): 1d numpy array whose size should be
            same as permutation axis of `node`.
        axis (int): permutation axis.

    Returns (numpy.ndarray): permutated `node` array.

    """
    if node.shape[axis] != len(permutation_index):
        raise ValueError(
            'node.shape[{}] = {} and len(permutation_index) = {} do not match!'
            .format(axis, node.shape[axis], len(permutation_index)))
    out_node = numpy.take(node, permutation_index, axis=axis).copy()
    return out_node


def permute_adj(adj, permutation_index, axis=None):
    """Permute index of adjacency matrix array

    Args:
        adj (numpy.ndarray): the array whose `axis` to be permuted.
            It is considered as adjacency matrix.
        permutation_index (numpy.ndarray): 1d numpy array whose size should be
            same as permutation axis of `node`.
        axis (list or None): list of 2d int, indicates the permutation axis.
            When None is passed (default), it uses -1 and -2 as `axis`, it
            means that last 2 axis are considered to be permuted.

    Returns (numpy.ndarray): permutated `adj` array.

    """
    if axis is not None:
        raise NotImplementedError('Sorry, it is not implemented yet.')
    axis = [-1, -2]
    num_node = len(permutation_index)
    for ax in axis:
        if adj.shape[ax] != len(permutation_index):
            raise ValueError(
                'adj.shape[{}] = {} and len(permutation_index) = {} do not '
                'match!'.format(axis, adj.shape[axis], len(permutation_index)))

    # TODO(nakago): support arbitrary axis. not this is only for axis=-1, -2
    out_adj = numpy.zeros_like(adj)
    for i in range(num_node):
        for j in range(num_node):
            out_adj[..., i, j] = adj[..., permutation_index[i],
                                     permutation_index[j]]
    return out_adj
