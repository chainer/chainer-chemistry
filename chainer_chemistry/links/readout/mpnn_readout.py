import chainer
from chainer import functions
from chainer import links

from chainer_chemistry.links.readout.set2set import Set2Set


class MPNNReadout(chainer.Chain):
    """MPNN submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector associated to
            each atom
        n_layers (int): number of LSTM layers for set2set
        processing_steps (int): number of processing for set2set
    """

    def __init__(self, out_dim, hidden_dim, n_layers, processing_steps=3):
        # type: (int, int, int, int) -> None
        super(MPNNReadout, self).__init__()
        with self.init_scope():
            self.set2set = Set2Set(in_channels=hidden_dim, n_layers=n_layers)
            self.linear1 = links.Linear(None, hidden_dim)
            self.linear2 = links.Linear(None, out_dim)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.processing_steps = processing_steps

    def __call__(self, h):
        # type: (chainer.Variable) -> chainer.Variable
        # h: (mb, node, ch)
        self.set2set.reset_state()
        for i in range(self.processing_steps):
            g = self.set2set(h)  # g: (mb, ch * 2)
        g = functions.relu(self.linear1(g))  # g: (mb, hidden_dim)
        g = self.linear2(g)  # g: (mb, out_dim)
        return g
