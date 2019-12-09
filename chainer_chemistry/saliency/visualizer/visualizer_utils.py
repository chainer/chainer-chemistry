from logging import getLogger
import numpy  # NOQA

from chainer import cuda


def red_blue_cmap(x):
    """Red to Blue color map

    Args:
        x (float): value between -1 ~ 1, represents normalized saliency score

    Returns (tuple): tuple of 3 float values representing R, G, B.
    """
    if x > 0:
        # Red for positive value
        # x=0 -> 1, 1, 1 (white)
        # x=1 -> 1, 0, 0 (red)
        return 1., 1. - x, 1. - x
    else:
        # Blue for negative value
        x *= -1
        return 1. - x, 1. - x, 1.


def min_max_scaler(saliency, logger=None):
    """Normalize saliency to value 0~1

    Args:
        saliency (numpy.ndarray or cupy.ndarray): saliency array
        logger:

    Returns (numpy.ndarray or cupy.ndarray): normalized saliency array

    """
    xp = cuda.get_array_module(saliency)
    maxv = xp.max(saliency)
    minv = xp.min(saliency)
    if maxv == minv:
        logger = logger or getLogger(__name__)
        logger.info('All saliency value is 0')
        saliency = xp.zeros_like(saliency)
    else:
        saliency = (saliency - minv) / (maxv - minv)
    return saliency


def abs_max_scaler(saliency, logger=None):
    """Normalize saliency to value -1~1

    Args:
        saliency (numpy.ndarray or cupy.ndarray): saliency array
        logger:

    Returns (numpy.ndarray or cupy.ndarray): normalized saliency array

    """
    xp = cuda.get_array_module(saliency)
    maxv = xp.max(xp.abs(saliency))
    if maxv <= 0:
        logger = logger or getLogger(__name__)
        logger.info('All saliency value is 0')
        return xp.zeros_like(saliency)
    else:
        return saliency / maxv


def normalize_scaler(saliency, axis=None, logger=None):
    """Normalize saliency to be sum=1

    Args:
        saliency (numpy.ndarray or cupy.ndarray): saliency array.
        axis (int): axis to take sum for normalization.
        logger:

    Returns (numpy.ndarray or cupy.ndarray): normalized saliency array

    """
    xp = cuda.get_array_module(saliency)
    if xp.sum(saliency < 0) > 0:
        logger = logger or getLogger(__name__)
        logger.warning('saliency array contains negative number, '
                       'which is unexpected!')
    vsum = xp.sum(xp.abs(saliency), axis=axis, keepdims=True)
    if vsum <= 0:
        logger = logger or getLogger(__name__)
        logger.info('All saliency value is 0')
        return xp.zeros_like(saliency)
    else:
        return saliency / vsum
