import json
from logging import getLogger
import numpy
try:
    from pathlib import PurePath
    _is_pathlib_available = True
except ImportError:
    _is_pathlib_available = False

from chainer import cuda


class JSONEncoderEX(json.JSONEncoder):
    """Encoder class used for `json.dump`"""

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, cuda.ndarray):
            return cuda.to_cpu(obj).tolist()
        elif _is_pathlib_available and isinstance(obj, PurePath):
            # save as str representation
            # convert windows path separator to linux format
            return str(obj).replace('\\', '/')
        else:
            return super(JSONEncoderEX, self).default(obj)


def save_json(filepath, params, ignore_error=False, indent=4, logger=None):
    """Save `params` to `filepath` in json format.

    It also supports `numpy` & `cupy` array serialization by converting them to
    `list` format.

    Args:
        filepath (str): filepath to save args
        params (dict or list): parameters to be saved.
        ignore_error (bool): If `True`, it will ignore exception with printing
            error logs, which prevents to stop.
        indent (int): Indent for saved file.
        logger:

    """
    try:
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=indent, cls=JSONEncoderEX)
    except Exception as e:
        if not ignore_error:
            raise e
        else:
            logger = logger or getLogger(__name__)
            logger.warning('Error occurred at save_json, but ignoring...')
            logger.warning('The file {} may not be saved or corrupted.'
                           .format(filepath))
            logger.warning(e)


def load_json(filepath):
    """Load params, which is stored in json format.

    Args:
        filepath (str): filepath to json file to load.

    Returns (dict or list): params
    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params
