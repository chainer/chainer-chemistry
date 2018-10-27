from collections import Iterable
from logging import getLogger
import six

from chainer import cuda


def _to_list(a):
    if isinstance(a, Iterable):
        a = list(a)
    else:
        a = [a]
    return a


def extend_node(node, out_size, axis=-1, value=0):
    """Extend size of `node` array

    For now, this function works same with `extend_array` method,
    this is just an alias function.

    Args:
        node (numpy.ndarray): the array whose `axis` to be extended.
            first axis is considered as "batch" axis.
        out_size (int): target output size for specified `axis`.
        axis (int): node feature axis to be extended.
            Default is `axis=-1`, which extends only last axis.
        value (int or float): value to be filled for extended place.

    Returns (numpy.ndarray): extended `node` array, extended place is filled
        with `value`

    """
    return extend_arrays_to_size(
        node, out_size=out_size, axis=axis, value=value)


def extend_adj(adj, out_size, axis=None, value=0):
    """Extend size of `adj` array

    For now, this function only differs default `axis` value from
    `extend_array` method, this is an alias function.

    Args:
        adj (numpy.ndarray): the array whose `axis` to be extended.
            first axis is considered as "batch" axis.
        out_size (int): target output size for specified `axis`.
        axis (list or None): node feature axis to be extended. Default is None,
            in this case `axis=[-1, -2]` is used to extend last 2 axes.
        value (int or float): value to be filled for extended place.

    Returns (numpy.ndarray): extended `adj` array, extended place is filled
        with `value`

    """
    axis = axis or [-1, -2]
    return extend_arrays_to_size(
        adj, out_size=out_size, axis=axis, value=value)


def extend_arrays_to_size(arrays, out_size, axis=-1, value=0):
    """Extend size of `arrays` array

    Args:
        arrays (numpy.ndarray): the array whose `axis` to be extended.
            first axis is considered as "batch" axis.
        out_size (int): target output size for specified `axis`.
        axis (int or list): node feature axis to be extended.
        value (int or float): value to be filled for extended place.

    Returns (numpy.ndarray): extended array, extended place is filled
        with `value`

    """
    batch_size = len(arrays)

    in_shape = _to_list(arrays[0].shape)
    out_shape = [batch_size] + in_shape

    axis = _to_list(axis)
    for ax in axis:
        if ax == 0:
            logger = getLogger(__name__)
            logger.warning('axis 0 detected, but axis=0 is expected to be '
                           'batch size dimension.')
        if out_shape[ax] > out_size:
            raise ValueError(
                'current size={} is larger than out_size={} at axis={}'
                .format(out_shape[ax], out_size, ax))
        out_shape[ax] = out_size
    return extend_arrays_to_shape(arrays, out_shape, value=value)


def extend_arrays_to_shape(arrays, out_shape, value=0):
    # Ref: `_concat_arrays_with_padding` method in chainer convert.py
    # https://github.com/chainer/chainer/blob/master/chainer/dataset/convert.py
    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        result = xp.full(out_shape, value, dtype=arrays[0].dtype)
        for i in six.moves.range(len(arrays)):
            src = arrays[i]
            slices = tuple(slice(dim) for dim in src.shape)
            result[(i,) + slices] = src
    return result
