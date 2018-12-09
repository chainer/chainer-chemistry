import numpy
import pytest

import chainer
from chainer import Variable, cuda
from chainer.links import Linear

from chainer_chemistry.link_hooks import VariableMonitorLinkHook


class DummyModel(chainer.Chain):
    def __init__(self):
        super(DummyModel, self).__init__()
        with self.init_scope():
            self.l1 = Linear(
                3, 1, initialW=numpy.array([[1, 3, 2]]),
                nobias=True)
        self.h = None

    def forward(self, x):
        self.h = self.l1(x)
        out = self.h * 3
        return out


@pytest.fixture
def model():
    return DummyModel()


def test_variable_monitor_link_hook_pre(model):
    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)
    x = Variable(x)
    pre_hook = VariableMonitorLinkHook(target_link=model.l1, timing='pre')
    with pre_hook:
        model(x)
    var = pre_hook.get_variable()
    assert var is x


def test_variable_monitor_link_hook_post(model):
    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)
    x = Variable(x)
    pre_hook = VariableMonitorLinkHook(target_link=model.l1, timing='post')
    with pre_hook:
        model(x)
    var = pre_hook.get_variable()
    assert var is model.h


def test_variable_monitor_link_hook_process(model):
    x = numpy.array([[1, 5, 8]], dtype=numpy.float32)
    x = Variable(x)
    pre_hook = VariableMonitorLinkHook(target_link=model.l1, timing='post')

    # Add process
    def _process_zeros(hook, args, target_var):
        xp = cuda.get_array_module(target_var.array)
        target_var.array = xp.zeros(target_var.array.shape)
    pre_hook.add_process('_process_zeros', _process_zeros)
    with pre_hook:
        model(x)

    assert numpy.allclose(model.h.array, numpy.zeros(model.h.shape))
    assert '_process_zeros' in pre_hook.process_fns.keys()

    # Delete process
    pre_hook.delete_process('_process_zeros')
    assert '_process_zeros' not in pre_hook.process_fns.keys()


def test_variable_monitor_link_hook_assert_raises(model):
    with pytest.raises(TypeError):
        # target_link must be chainer.Link
        pre_hook = VariableMonitorLinkHook(target_link='hoge')

    with pytest.raises(ValueError):
        # check timing args
        pre_hook = VariableMonitorLinkHook(target_link=model.l1, timing='hoge')

    hook = VariableMonitorLinkHook(target_link=model.l1)

    def _process(hook, args, target_var):
        pass

    with pytest.raises(TypeError):
        # key is wrong
        hook.add_process(1, _process)

    with pytest.raises(TypeError):
        # fn is wrong
        hook.add_process('hoge', 'var')

    hook.add_process('hoge', _process)
    with pytest.raises(TypeError):
        # key is wrong
        hook.delete_process(1)
    hook.delete_process('hoge')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
