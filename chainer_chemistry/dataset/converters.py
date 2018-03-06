import chainer
from chainer.backends import cuda
import numpy


def concat_mols(batch, device=None, padding=0):
    """Concatenates a list of mols into array(s).

    Args:
        batch (list):
            A list of examples. This is typically given by a dataset
            iterator.
        device (int):
            Device ID to which each array is sent. Negative value
            indicates the host memory (CPU). If it is omitted, all arrays are
            left in the original device.
        padding:
            Scalar value for extra elements. If this is None (default),
            an error is raised on shape mismatch. Otherwise, an array of
            minimum dimensionalities that can accommodate all arrays is
            created, and elements outside of the examples are padded by this
            value.

    Returns:
        Array, a tuple of arrays, or a dictionary of arrays:
        The type depends on the type of each example in the batch.
    """
    return chainer.dataset.concat_examples(batch, device, padding=padding)


def to_device(device, x):
    """Send an array to a given device. """
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        return cuda.to_gpu(x, device)


def concat_sparse_rsgcn(batch, device=None, do_flatten=False):
    """Concatenates a list of data"""

    if len(batch) == 0:
        raise ValueError('batch is empty')
    if not isinstance(batch[0], tuple):
        raise ValueError('only tuple is supported')
    num_data = len(batch[0])
    if num_data != 5:
        raise ValueError('dataset must contain five data')
        # mol, adj_data, adj_row, adj_col, label

    padding = [0, 0, -1, -1, 0]
    result = []

    i = 0
    mol = chainer.dataset.convert._concat_arrays(
        [example[i] for example in batch], padding[i])
    result.append(to_device(device, mol))

    if do_flatten:
        total_nnz = 0
        for example in batch:
            total_nnz += example[1].shape[0]

        adj_data = numpy.empty((total_nnz), dtype=batch[0][1].dtype)
        adj_row = numpy.empty((total_nnz), dtype=batch[0][2].dtype)
        adj_col = numpy.empty((total_nnz), dtype=batch[0][3].dtype)

        head = 0
        for i, example in enumerate(batch):
            nnz = example[1].shape[0]
            adj_data[head:head+nnz] = example[1]
            adj_row[head:head+nnz] = example[2] + i * mol.shape[1]
            adj_col[head:head+nnz] = example[3] + i * mol.shape[1]
            head += nnz

        result.append(to_device(device, adj_data))
        result.append(to_device(device, adj_row))
        result.append(to_device(device, adj_col))
    else:
        i = 1
        adj_data = chainer.dataset.convert._concat_arrays(
            [example[i] for example in batch], padding[i])
        result.append(to_device(device, adj_data))

        i = 2
        adj_row = chainer.dataset.convert._concat_arrays(
            [example[i] for example in batch], padding[i])
        result.append(to_device(device, adj_row))

        i = 3
        adj_col = chainer.dataset.convert._concat_arrays(
            [example[i] for example in batch], padding[i])
        result.append(to_device(device, adj_col))

    i = 4
    label = chainer.dataset.convert._concat_arrays(
        [example[i] for example in batch], padding[i])
    result.append(to_device(device, label))

    return tuple(result)
