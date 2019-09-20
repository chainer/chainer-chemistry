import chainer
from chainer import functions, links


from chainer_chemistry.functions import megnet_softplus
from chainer_chemistry.links.readout.set2set import Set2Set


class MEGNetReadout(chainer.Chain):
    """MEGNet submodule for readout part.

    Args:
        out_dim (int): dimension of output feature vector
        in_channels (int): dimension of feature vector associated to
            each node. Must not be `None`.
        n_layers (int): number of LSTM layers for set2set
        processing_steps (int): number of processing for set2set
        dropout_ratio (float): ratio of dropout
    """

    def __init__(self, out_dim=32, in_channels=32, n_layers=1,
                 processing_steps=3, dropout_ratio=-1):
        super(MEGNetReadout, self).__init__()
        if processing_steps <= 0:
            raise ValueError("[ERROR] Unexpected value processing_steps={}"
                             .format(processing_steps))

        self.processing_steps = processing_steps
        self.dropout_ratio = dropout_ratio
        with self.init_scope():
            self.set2set_for_atom = Set2Set(
                in_channels=in_channels, n_layers=n_layers)
            self.set2set_for_pair = Set2Set(
                in_channels=in_channels, n_layers=n_layers)
            self.linear = links.Linear(None, out_dim)

    def __call__(self, atoms_feat, pair_feat, global_feat):
        a_f = atoms_feat
        p_f = pair_feat
        g_f = global_feat

        # readout for atom and pair feature
        self.set2set_for_atom.reset_state()
        self.set2set_for_pair.reset_state()
        for i in range(self.processing_steps):
            a_f_r = self.set2set_for_atom(a_f)
            p_f_r = self.set2set_for_pair(p_f)

        # concating all features
        h = functions.concat((a_f_r, p_f_r, g_f), axis=1)
        if self.dropout_ratio > 0.0:
            h = functions.dropout(h, ratio=self.dropout_ratio)
        out = megnet_softplus(self.linear(h))
        return out
