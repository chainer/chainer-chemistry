import chainer
from chainer import functions, links


from chainer_chemistry.functions import improved_softplus
from chainer_chemistry.links.update.megnet_update import MEGNetUpdate
from chainer_chemistry.links.readout.megnet_readout import MEGNetReadout


def reshaped_feat(feat, idx):
    max_idx = max(idx)
    vec_list = [feat[idx == i] for i in range(max_idx+1)]
    return functions.pad_sequence(vec_list)


class MEGNet(chainer.Chain):
    """MEGNet

    See Chi Chen et al, \
        Graph Networks as a Universal Machine Learning Framework for Molecules
        and Crystals. \
        `arXiv:1812.05055 <https://arxiv.org/abs/1812.05055>`_

    Args:
        out_dim (int): dimension of output feature vector
        n_update_layers (int): the number of MEGNetUpdate layers
        dropout_ratio (float): ratio of dropout
    """

    def __init__(self, out_dim=32, n_update_layers=3, dropout_ratio=-1):
        super(MEGNet, self).__init__()
        if n_update_layers <= 0:
            raise ValueError('n_update_layers must be a positive integer, '
                             'but it was set to {}'.format(n_update_layers))

        self.n_update_layers = n_update_layers
        self.dropout_ratio = dropout_ratio
        with self.init_scope():
            self.update_layers = chainer.ChainList(
                *[MEGNetUpdate(
                    hidden_dim_for_dense=[64, 32],
                    hidden_dim_for_update=[64, 64, 32]
                ) for _ in range(n_update_layers)])
            self.readout_for_atom = MEGNetReadout(in_channels=32, n_layers=16,
                                                  processing_steps=3)
            self.readout_for_pair = MEGNetReadout(in_channels=32, n_layers=16,
                                                  processing_steps=3)
            self.linear = links.Linear(None, out_dim)

    def __call__(self, atoms_feat, pair_feat, global_feat, *args):
        a_f = atoms_feat
        p_f = pair_feat
        g_f = global_feat
        # --- MGENet update ---
        for i in range(self.n_update_layers):
            a_f, p_f, g_f = self.update_layers[i](a_f, p_f, g_f, *args)
            if self.dropout_ratio > 0.0:
                a_f = functions.dropout(a_f, ratio=self.dropout_ratio)
                p_f = functions.dropout(p_f, ratio=self.dropout_ratio)
                g_f = functions.dropout(g_f, ratio=self.dropout_ratio)
        # --- reshape ---
        atom_idx = args[0]
        pair_idx = args[1]
        a_f = reshaped_feat(a_f, atom_idx)
        p_f = reshaped_feat(p_f, pair_idx)
        # --- MGENet readout ---
        a_f_r = self.readout_for_atom(a_f)
        p_f_r = self.readout_for_pair(p_f)
        concated_v = functions.concat((a_f_r, p_f_r, g_f), axis=1)
        if self.dropout_ratio > 0.0:
            concated_v = functions.dropout(concated_v,
                                           ratio=self.dropout_ratio)
        # --- convert feature's dim to out_dim ---
        out = improved_softplus(self.linear(concated_v))
        return out
