import numpy
from chainer.dataset.convert import _concat_arrays


# increase dimension by 1 and padding
def batch_with_padding(name, batch, device=None, pad=0):
    feat = _concat_arrays(
        [getattr(example, name) for example in batch], pad)
    return device.send(feat)


# increase dimension by 1 and no padding (all shape must be same)
def batch_without_padding(name, batch, device=None):
    feat = _concat_arrays(
        [getattr(example, name) for example in batch], None)
    return device.send(feat)


def concat_with_padding(name, batch, device=None, pad=0):
    feat = batch_with_padding(name, batch, device=device, pad=pad)
    a, b = feat.shape
    return feat.reshape((a * b))


# not increase dimension
def concat(name, batch, device=None, axis=0):
    feat = numpy.concatenate([getattr(data, name) for data in batch],
                             axis=axis)
    return device.send(feat)


def shift_concat(name, batch, device=None, shift_attr='x', shift_axis=1):
    shift_index_array = numpy.cumsum(
        numpy.array([0] + [getattr(data, shift_attr).shape[0] for data in batch]))
    feat = numpy.concatenate([
        getattr(data, name) + shift_index_array[i]
        for i, data in enumerate(batch)], axis=shift_axis)
    return device.send(feat)


def shift_concat_with_padding(name, batch, device=None, shift_attr='x', shift_axis=1):
    max_n_nodes = max([data.x.shape[0] for data in batch])
    shift_index_array = numpy.arange(0, len(batch) * max_n_nodes, max_n_nodes)
    feat = numpy.concatenate([
        getattr(data, name) + shift_index_array[i]
        for i, data in enumerate(batch)], axis=shift_axis)
    return device.send(feat)


def create_index(name, batch, device=None, shift_attr='x', shift_axis=0):
    # name is not used.
    batch_index = numpy.array([
        numpy.ones(getattr(data, shift_attr).shape[shift_axis],
                   dtype=numpy.int32) * i for i, data in enumerate(batch)])
    return device.send(batch_index)
