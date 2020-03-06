import numpy

from chainer.dataset.convert import _concat_arrays


def batch_with_padding(name, batch, device=None, pad=0):
    """Batch with padding (increase ndim by 1)

    Args:
        name (str): propaty name of graph data
        batch (list[BaseGraphData]): list of base graph data
        device (chainer.backend.Device, optional): device. Defaults to None.
        pad (int, optional): padding value. Defaults to 0.

    Returns:
        BaseGraphDataset: graph dataset sent to `device`
    """
    feat = _concat_arrays(
        [getattr(example, name) for example in batch], pad)
    return device.send(feat)


def batch_without_padding(name, batch, device=None):
    """Batch without padding (increase ndim by 1)

    Args:
        name (str): propaty name of graph data
        batch (list[BaseGraphData]): list of base graph data
        device (chainer.backend.Device, optional): device. Defaults to None.

    Returns:
        BaseGraphDataset: graph dataset sent to `device`
    """
    feat = _concat_arrays(
        [getattr(example, name) for example in batch], None)
    return device.send(feat)


def concat_with_padding(name, batch, device=None, pad=0):
    """Concat without padding (ndim does not increase)

    Args:
        name (str): propaty name of graph data
        batch (list[BaseGraphData]): list of base graph data
        device (chainer.backend.Device, optional): device. Defaults to None.
        pad (int, optional): padding value. Defaults to 0.

    Returns:
        BaseGraphDataset: graph dataset sent to `device`
    """
    feat = batch_with_padding(name, batch, device=device, pad=pad)
    a, b = feat.shape
    return feat.reshape((a * b))


def concat(name, batch, device=None, axis=0):
    """Concat with padding (ndim does not increase)

    Args:
        name (str): propaty name of graph data
        batch (list[BaseGraphData]): list of base graph data
        device (chainer.backend.Device, optional): device. Defaults to None.
        pad (int, optional): padding value. Defaults to 0.

    Returns:
        BaseGraphDataset: graph dataset sent to `device`
    """
    feat = numpy.concatenate([getattr(data, name) for data in batch],
                             axis=axis)
    return device.send(feat)


def shift_concat(name, batch, device=None, shift_attr='x', shift_axis=1):
    """Concat with index shift (ndim does not increase)

    Concatenate graphs into a big one.
    Used for sparse pattern batching.

    Args:
        name (str): propaty name of graph data
        batch (list[BaseGraphData]): list of base graph data
        device (chainer.backend.Device, optional): device. Defaults to None.

    Returns:
        BaseGraphDataset: graph dataset sent to `device`
    """
    shift_index_array = numpy.cumsum(
        numpy.array([0] + [getattr(data, shift_attr).shape[0]
                           for data in batch]))
    feat = numpy.concatenate([
        getattr(data, name) + shift_index_array[i]
        for i, data in enumerate(batch)], axis=shift_axis)
    return device.send(feat)


def shift_concat_with_padding(name, batch, device=None, shift_attr='x',
                              shift_axis=1):
    """Concat with index shift and padding (ndim does not increase)

    Concatenate graphs into a big one.
    Used for sparse pattern batching.

    Args:
        name (str): propaty name of graph data
        batch (list[BaseGraphData]): list of base graph data
        device (chainer.backend.Device, optional): device. Defaults to None.

    Returns:
        BaseGraphDataset: graph dataset sent to `device`
    """
    max_n_nodes = max([data.x.shape[0] for data in batch])
    shift_index_array = numpy.arange(0, len(batch) * max_n_nodes, max_n_nodes)
    feat = numpy.concatenate([
        getattr(data, name) + shift_index_array[i]
        for i, data in enumerate(batch)], axis=shift_axis)
    return device.send(feat)
