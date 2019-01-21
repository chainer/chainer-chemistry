from chainer import functions


def softmax(x, axis=1, mask=None, mask_value=1e10):
    if mask is None:
        h = x
    else:
        if x.shape != mask.shape:
            raise ValueError("x.shape={} and mask.shape={} must be same!"
                             .format(x.shape, mask.shape))
        h = x + (mask - 1.) * mask_value
    return functions.softmax(h, axis=axis)
