import os
import json
import numpy
from chainer import cuda


class JSONEncoderEX(json.JSONEncoder):
    """Ref: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python"""
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, cuda.ndarray):
            return cuda.to_cpu(obj).tolist()
        else:
            return super(JSONEncoderEX, self).default(obj)


def save_json(filepath, params, ignore_error=True):
    """save params in json format.

    Args:
        filepath (str): filepath to save args
        params (dict or list): args to be saved 
        ignore_error (bool): if True, it will ignore exception with printing 
            error logs, which prevents to stop

    """
    try:
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4, cls=JSONEncoderEX)
    except Exception as e:
        if not ignore_error:
            raise e
        else:
            print('[WARNING] Error occurred at save_json, but ignoring...')
            print('The file {} may not be saved.'.format(filepath))
            print(e)


def load_json(filepath):
    """load params, whicch is stored in json format.

    Args:
        filepath (str): filepath to save args

    Returns (dict or list): params

    """
    with open(filepath, 'r') as f:
        params = json.load(f)
    return params
