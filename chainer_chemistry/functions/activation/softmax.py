from chainer import functions


def softmax(x, axis=1, mask=None, mask_value=1e10):
    """softmax function, which supports `mask`.

    Args:
        x (Variable): Input variable
        axis (int): The axis along which the softmax is to be computed.
        mask (Variable or None):
            Default value is `None` which does not use mask,
            this case the result is same with original `softmax` computation.
            When `mask` is not `None`, it is assumed to have value 1 or 0.
            1 indicates actual feature, and 0 indicates virtual feature to be
            masked.
        mask_value (int): The value used for masking.

    Returns:
        output (Variable): Output variable whose shape is same with `x`

    """
    if mask is None:
        h = x
    else:
        if x.shape != mask.shape:
            raise ValueError("x.shape={} and mask.shape={} must be same!"
                             .format(x.shape, mask.shape))
        h = x + (mask - 1.) * mask_value
    return functions.softmax(h, axis=axis)
