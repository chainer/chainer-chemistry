import numpy


# Default color function
def red_blue_cmap(x):
    # return in RGB order
    if x > 0:
        # Red for positive value
        # x=0 -> 1, 1, 1 (white)
        # x=1 -> 1, 0, 0 (red)
        return 1., 1. - x, 1. - x
    else:
        # Blue for negative value
        x *= -1
        return 1. - x, 1. - x, 1.


def min_max_scaler(saliency):
    """Normalize saliency to value 0-1"""
    maxv = numpy.max(saliency)
    minv = numpy.min(saliency)
    if maxv == minv:
        saliency = numpy.zeros_like(saliency)
    else:
        saliency = (saliency - minv) / (maxv - minv)
    return saliency


def abs_max_scaler(saliency):
    """Normalize saliency to value -1~+1"""
    maxv = numpy.max(numpy.abs(saliency))
    if maxv <= 0:
        return numpy.zeros_like(saliency)
    else:
        return saliency / maxv


def normalize_scaler(saliency, axis=None):
    """Normalize saliency to be sum=1"""
    vsum = numpy.sum(numpy.abs(saliency), axis=axis, keepdims=True)
    if vsum <= 0:
        return numpy.zeros_like(saliency)
    else:
        return saliency / vsum
