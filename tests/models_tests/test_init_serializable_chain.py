import tempfile

import chainer
import numpy
import pytest


from chainer_chemistry.models.init_serializable_chain import \
    InitSerializableChain, retain_args


class DummyNonRetainArgsChain(InitSerializableChain):
    """Forgot to use @retain_args, it should raise error when save"""

    def __init__(self, i, j=10):
        super(DummyNonRetainArgsChain, self).__init__()
        self.i = i
        self.j = j


class DummyChainParent(InitSerializableChain):

    @retain_args
    def __init__(self, i, child):
        super(DummyChainParent, self).__init__()
        self.i = i
        with self.init_scope():
            self.child = child


class DummyChainChild(InitSerializableChain):

    @retain_args
    def __init__(self, i, j=5, postprocess_fn=chainer.functions.softmax):
        super(DummyChainChild, self).__init__()
        self.i = i
        self.j = j
        self.postprocess_fn = postprocess_fn
        with self.init_scope():
            self.l = chainer.links.Linear(i, j)


def test_non_retain_args():
    net = DummyNonRetainArgsChain(3)
    with tempfile.TemporaryDirectory() as dirpath:
        with pytest.raises(AttributeError):
            net.save(dirpath)


def test_save_load():
    net_child = DummyChainChild(i=6)
    net = DummyChainParent(3, child=net_child)

    with tempfile.TemporaryDirectory() as dirpath:
        net.save(dirpath)
        net_load = InitSerializableChain.load(dirpath)

    assert isinstance(net_load, DummyChainParent)
    # --- attribute should be same ---
    assert net.i == net_load.i
    assert net.child.i == net_load.child.i
    assert net.child.j == net_load.child.j
    assert net.child.postprocess_fn == net_load.child.postprocess_fn
    # --- param should be same ---
    assert numpy.alltrue(net.child.l.W.data == net_load.child.l.W.data)
    assert numpy.alltrue(net.child.l.b.data == net_load.child.l.b.data)


if __name__ == '__main__':
    # print('NFP example')
    # nfp = NFP(10)
    # nfp.save('nfp')
    # nfp_load = InitSerializableChain.load('nfp')
    # print('nfp_load done', type(nfp_load))
    pytest.main([__file__, '-v', '-s'])
