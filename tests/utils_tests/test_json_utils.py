import os

import numpy
import pytest
from pathlib import Path

from chainer_chemistry.utils.json_utils import load_json
from chainer_chemistry.utils.json_utils import save_json

params = {
    'a_int': 1,
    'b_str': 'string',
    'c_list': [1, 2, 3],
    'd_tuple': (1, 2),
    'n_int_scalar': numpy.array(1),
    'n_int_array': numpy.array([1]),
    'n_float': numpy.array([[1.0, 2.0], [3.0, 4.0]]),
    'path': Path('/tmp/hoge')
}


params_invalid = {
    'lambda_function': lambda x: x * 2,
}


def test_save_json(tmpdir):
    filepath = os.path.join(str(tmpdir), 'tmp.json')
    save_json(filepath, params)
    assert os.path.exists(filepath)


def test_save_json_ignore_error(tmpdir):
    filepath = os.path.join(str(tmpdir), 'tmp.json')

    # 1. should raise error when ignore_error=False
    with pytest.raises(TypeError):
        save_json(filepath, params_invalid, ignore_error=False)

    # 2. should not raise error when ignore_error=False
    save_json(filepath, params_invalid, ignore_error=True)


def test_load_json(tmpdir):
    filepath = os.path.join(str(tmpdir), 'tmp.json')
    # TODO(nakago): better to remove `save_json` dependency for unittest.
    save_json(filepath, params)

    params_load = load_json(filepath)
    expected_params_load = {
        'a_int': 1,
        'b_str': 'string',
        'c_list': [1, 2, 3],
        'd_tuple': [1, 2],
        'n_float': [[1.0, 2.0], [3.0, 4.0]],
        'n_int_array': [1],
        'n_int_scalar': 1,
        'path': '/tmp/hoge'  # PurePath is converted to str
    }
    assert params_load == expected_params_load


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
