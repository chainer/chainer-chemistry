import os

import chainer
from chainer import cuda
import numpy
import pytest

from chainer_chemistry.models.prediction.base import BaseForwardModel


class DummyForwardModel(BaseForwardModel):

    def __init__(self, device=-1, dummy_str='dummy'):
        super(DummyForwardModel, self).__init__()
        with self.init_scope():
            self.l = chainer.links.Linear(3, 10)

        self.dummy_str = dummy_str
        self.initialize(device)

    def __call__(self, x):
        return self.l(x)


# test `_forward` is done by `Classifier` and `Regressor` concrete class.
def _test_save_load_pickle(device, tmpdir):
    model = DummyForwardModel(device=device, dummy_str='hoge')

    filepath = os.path.join(str(tmpdir), 'model.pkl')
    model.save_pickle(filepath)
    model_load = DummyForwardModel.load_pickle(filepath, device=device)

    # --- check model class ---
    assert isinstance(model_load, DummyForwardModel)
    # --- check model attribute is same ---
    assert model_load.dummy_str == model.dummy_str
    assert model_load.dummy_str == 'hoge'
    assert model_load.get_device() == model.get_device()
    assert model_load.get_device() == device

    # --- check model parameter is same ---
    params = model.namedparams()
    params_load = dict(model_load.namedparams())
    for k, v in params:
        v_load = params_load[k]
        assert cuda.get_device_from_array(v_load.data).id == device
        assert numpy.allclose(cuda.to_cpu(v.data), cuda.to_cpu(v_load.data))


def test_save_load_pickle_cpu(tmpdir):
    _test_save_load_pickle(device=-1, tmpdir=tmpdir)


@pytest.mark.gpu
def test_save_load_pickle_gpu(tmpdir):
    _test_save_load_pickle(device=0, tmpdir=tmpdir)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
