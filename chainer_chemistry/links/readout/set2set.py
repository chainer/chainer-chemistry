from typing import List, Optional  # NOQA

import chainer
from chainer import cuda
from chainer import functions
from chainer import links
import numpy  # NOQA


class Set2Set(chainer.Chain):
    r"""MPNN subsubmodule for readout part.

    See: Oriol Vinyals+, \
        Order Matters: Sequence to sequence for sets. November 2015.
        `arXiv:1511.06391 <https://arxiv.org/abs/1511.06391>`

    Args:
        in_channels (int): dimension of input feature vector
        n_layers (int): number of LSTM layers

    Returns (chainer.Variable):
        Output feature vector: (minibatch, in_channels * 2)

    """

    def __init__(self, in_channels, n_layers=1):
        # type: (int, int) -> None
        super(Set2Set, self).__init__()
        with self.init_scope():
            self.lstm_layer = links.NStepLSTM(
                n_layers=n_layers,
                in_size=in_channels * 2,
                out_size=in_channels,
                dropout=0)
        self.in_channels = in_channels
        self.n_layers = n_layers
        self.hx = None  # type: Optional[chainer.Variable]
        self.cx = None  # type: Optional[chainer.Variable]
        self.q_star = None  # type: Optional[List]

    def __call__(self, h):
        # type: (chainer.Variable) -> chainer.Variable
        xp = cuda.get_array_module(h)
        mb, node, ch = h.shape  # type: int, int, int
        if self.q_star is None:
            self.q_star = [
                xp.zeros((1, self.in_channels * 2)).astype('f')
                for _ in range(mb)
            ]
        self.hx, self.cx, q = self.lstm_layer(self.hx, self.cx, self.q_star)
        # self.hx: (mb, mb, ch)
        # self.cx: (mb, mb, ch)
        # q: List[(1, ch) * mb]
        q = functions.stack(q)  # q: (mb, 1, ch)
        q_ = functions.transpose(q, axes=(0, 2, 1))  # q_: (mb, ch, 1)
        e = functions.matmul(h, q_)  # e: (mb, node, 1)
        a = functions.softmax(e)  # a: (mb, node, 1)
        a = functions.broadcast_to(a, h.shape)  # a: (mb, node, ch)
        r = functions.sum((a * h), axis=1, keepdims=True)  # r: (mb, 1, ch)
        q_star_ = functions.concat((q, r), axis=2)  # q_star_: (mb, 1, ch*2)
        self.q_star = functions.separate(q_star_)
        return functions.reshape(q_star_, (mb, ch * 2))

    def reset_state(self):
        # type: () -> None
        self.hx = None
        self.cx = None
        self.q_star = None
