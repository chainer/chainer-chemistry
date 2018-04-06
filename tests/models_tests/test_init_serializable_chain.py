import tempfile

import chainer
import pytest

from chainer_chemistry.models.nfp import NFP

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
    def __init__(self, i, j=10, child=None):
        super(DummyChainParent, self).__init__()
        self.i = i
        self.j = j
        with self.init_scope():
            if child is not None:
                self.child = child


class DummyChainChild(InitSerializableChain):

    @retain_args
    def __init__(self, j, k=5):
        super(DummyChainChild, self).__init__()
        self.j = j
        self.k = k


def test_non_retain_args():
    net = DummyNonRetainArgsChain(3)
    with tempfile.TemporaryDirectory() as dirpath:
        with pytest.raises(AttributeError):
            net.save(dirpath)


def test_save_load():
    net_child = DummyChainChild(6)
    net = DummyChainParent(3, child=net_child)

    with tempfile.TemporaryDirectory() as dirpath:
        net.save(dirpath)
        net_load = InitSerializableChain.load(dirpath)

    # assert isinstance(net_load, DummyChainParent)


if __name__ == '__main__':
    class A(InitSerializableChain):

        @retain_args
        def __init__(self, i, j=3, b=None):
            super(A, self).__init__()
            print('A init')
            self.i = i
            self.j = j
            if b is not None:
                with self.init_scope():
                    self.b = b


    from chainer import links as L


    class B(InitSerializableChain):

        @retain_args
        def __init__(self, i=10, j=30):
            super(B, self).__init__()
            self.i = i
            self.j = j
            with self.init_scope():
                self.l = L.Linear(i, j)

    # a = A(i=2, j=4)
    b = B()
    a = A(1, j='null', b=b)
    print('A')
    print(a.i)
    print(a.j)
    print('a._init_args_dict', a._init_args_dict)
    a.save('a')

    # a_load = A.load('a')  # type: A
    a_load = InitSerializableChain.load('a')  # type: A
    print('a_load...')
    print(a_load.i)
    print(a_load.j)
    print(a_load._init_args_dict)

    print('b', a_load.b.i, a_load.b.j)

    # b_load = B.load('a/b')
    # print('b_load...')
    # print(b_load.i)
    # print(b_load.j)

    print('NFP example')
    nfp = NFP(10)
    nfp.save('nfp')

    nfp_load = InitSerializableChain.load('nfp')
    print('nfp_load done', type(nfp_load))
