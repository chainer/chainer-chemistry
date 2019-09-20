import chainer
from chainer import functions, links


class CGCNNReadout(chainer.Chain):
    """CGCNN submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
    """

    def __init__(self, out_dim=128):
        super(CGCNNReadout, self).__init__()
        with self.init_scope():
            self.linear = links.Linear(None, out_dim)

    def __call__(self, atom_feat, atom_idx):
        average_pool = [functions.mean(atom_feat[idx], axis=0, keepdims=True)
                        for idx in atom_idx]
        h = functions.concat(average_pool, axis=0)
        h = self.linear(h)
        h = functions.softplus(h)
        return h
