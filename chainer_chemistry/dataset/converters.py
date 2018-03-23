import chainer
from chainer.backends import cuda
import numpy

import cupy


def concat_mols(batch, device=None, padding=0):
    """Concatenates a list of molecules into array(s).

    This function converts an "array of tuples" into a "tuple of arrays".
    Specifically, given a list of examples each of which consists of
    a list of elements, this function first makes an array
    by taking the element in the same position from each example
    and concatenates them along the newly-inserted first axis
    (called `batch dimension`) into one array.
    It repeats this for all positions and returns the resulting arrays.

    The output type depends on the type of examples in ``batch``.
    For instance, consider each example consists of two arrays ``(x, y)``.
    Then, this function concatenates ``x`` 's into one array, and ``y`` 's
    into another array, and returns a tuple of these two arrays. Another
    example: consider each example is a dictionary of two entries whose keys
    are ``'x'`` and ``'y'``, respectively, and values are arrays. Then, this
    function concatenates ``x`` 's into one array, and ``y`` 's into another
    array, and returns a dictionary with two entries ``x`` and ``y`` whose
    values are the concatenated arrays.

    When the arrays to concatenate have different shapes, the behavior depends
    on the ``padding`` value. If ``padding`` is ``None``, it raises an error.
    Otherwise, it builds an array of the minimum shape that the
    contents of all arrays can be substituted to. The padding value is then
    used to the extra elements of the resulting arrays.

    The current implementation is identical to
    :func:`~chainer.dataset.concat_examples` of Chainer, except the default
    value of the ``padding`` option is changed to ``0``.

    .. admonition:: Example

       >>> import numpy
       >>> from chainer_chemistry.dataset.converters import concat_mols
       >>> x0 = numpy.array([1, 2])
       >>> x1 = numpy.array([4, 5, 6])
       >>> dataset = [x0, x1]
       >>> results = concat_mols(dataset)
       >>> print(results)
       [[1 2 0]
        [4 5 6]]

    .. seealso:: :func:`chainer.dataset.concat_examples`

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


def to_device(device, x, sync=False):
    """Send an array to a given device. """
    if device is None:
        return x
    elif device < 0:
        return cuda.to_cpu(x)
    else:
        with cuda.get_device_from_id(device) as _device:
            x_gpu = cupy.ndarray(x.shape, dtype=x.dtype)
            x_gpu.set(x)
            if sync:
                _device.synchronize()

        return x_gpu


def concat_sparse_rsgcn(batch, device=None, do_flatten=False):
    """Concatenator for Sparse RSGCN model"""

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
    result.append(to_device(device, label, sync=True))

    return tuple(result)
