from typing import Iterable

import numpy
import six
from chainer import cuda


def _to_list(a):
    if isinstance(a, Iterable):
        a = list(a)
    else:
        a = [a]
    return a


def extend_node(node, out_size, axis=-1, value=0):
    return extend_arrays_to_size(
        node, out_size=out_size, axis=axis, value=value)


def extend_adj(adj, out_size, axis=None, value=0):
    axis = axis or [-1, -2]
    return extend_arrays_to_size(
        adj, out_size=out_size, axis=axis, value=value)


def extend_arrays_to_size(arrays, out_size, axis=-1, value=0):
    """Extend size of `node` array

    Args:
        arrays (numpy.ndarray): the array whose `axis` to be extended.
            first axis is considered as "batch" axis.
        out_size (int): target output size for specified `axis`.
        axis (int): node feature axis to be extended.
        value (int or float): value to be filled for extended place.

    Returns (numpy.ndarray): extended `node` array, extended place is filled
        with `value`

    """
    batch_size = len(arrays)

    in_shape = _to_list(arrays[0].shape)
    out_shape = [batch_size] + in_shape

    axis = _to_list(axis)
    for ax in axis:
        if out_shape[ax] > out_size:
            raise ValueError(
                'current size={} is larger than out_size={} at axis={}'
                .format(out_shape[ax], out_size, ax))
        out_shape[ax] = out_size
    return extend_arrays_to_shape(arrays, out_shape, value=value)


def extend_arrays_to_shape(arrays, out_shape, value=0):
    # TODO: update ref
    # Ref: `_concat_arrays_with_padding` method in chainer convert.py
    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        result = xp.full(out_shape, value, dtype=arrays[0].dtype)
        for i in six.moves.range(len(arrays)):
            src = arrays[i]
            slices = tuple(slice(dim) for dim in src.shape)
            result[(i,) + slices] = src
    return result
