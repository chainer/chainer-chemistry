import chainer


if int(chainer.__version__[0]) >= 3:
    _matmul_fn = chainer.functions.matmul
else:
    _matmul_fn = chainer.functions.batch_matmul


def matmul(a, b, transa=False, transb=False):
    """Computes the matrix multiplication of two arrays.

    Args:
        a (Variable): The left operand of the matrix multiplication.
            If ``a`` and ``b`` are both 1-D arrays, ``matmul`` returns a dot
            product of vector `a` and vector `b`. If 2-D arrays, ``matmul``
            returns matrix product of ``a`` and ``b``. If arrays' dimension is
            larger than 2, they are treated as a stack of matrices residing in
            the last two indexes. ``matmul`` returns a stack of each two
            arrays. ``a`` and ``b`` must have the same dimension.
        b (Variable): The right operand of the matrix multiplication.
            Its array is treated as a matrix in the same way as ``a``'s array.
        transa (bool): If ``True``, each matrices in ``a`` will be transposed.
            If ``a.ndim == 1``, do nothing.
        transb (bool): If ``True``, each matrices in ``b`` will be transposed.
            If ``b.ndim == 1``, do nothing.

    Returns:
        ~chainer.Variable: The result of the matrix multiplication.

    .. admonition:: Example

        >>> a = np.array([[1, 0], [0, 1]], 'f')
        >>> b = np.array([[4, 1], [2, 2]], 'f')
        >>> F.matmul(a, b).data
        array([[ 4.,  1.],
               [ 2.,  2.]], dtype=float32)

    """
    return _matmul_fn(a, b, transa=transa, transb=transb)
