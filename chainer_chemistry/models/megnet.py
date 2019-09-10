import chainer
from chainer import functions


from chainer_chemistry.links.update.megnet_update import MEGNetUpdate
from chainer_chemistry.links.readout.megnet_readout import MEGNetReadout


class MEGNet(chainer.Chain):
    """MEGNet

    See Chi Chen et al, \
        Graph Networks as a Universal Machine Learning Framework for Molecules
        and Crystals. \
        `arXiv:1812.05055 <https://arxiv.org/abs/1812.05055>`_

    Args:
        n_update_layers (int): number of MEGNetUpdate layers
    """

    def __init__(self, n_update_layers=3):
        super(MEGNet, self).__init__()
        if n_update_layers <= 0:
            raise ValueError('n_update_layers must be a positive integer, '
                             'but it was set to {}'.format(n_update_layers))

        self.n_update_layers = n_update_layers
        with self.init_scope():
            self.update_layers = chainer.ChainList(
                *[MEGNetUpdate(
                    hidden_dim_for_dense=[64, 32],
                    hidden_dim_for_update=[64, 64, 32]
                ) for _ in range(n_update_layers)])
            self.readout_for_atom = MEGNetReadout(
                in_channels=32, n_layers=16, processing_steps=3)
            self.readout_for_pair = MEGNetReadout(
                in_channels=32, n_layers=16, processing_steps=3)

    def __call__(self, atoms_feat, pair_feat, global_feat, *args):
        a_f = atoms_feat
        p_f = pair_feat
        g_f = global_feat
        # --- MGENet update ---
        for i in range(self.n_update_layers):
            a_f, p_f, g_f = self.update_layers[i](a_f, p_f, g_f, *args)
        # --- MGENet readout ---
        a_f_r = self.readout_for_atom(a_f)
        p_f_r = self.readout_for_pair(p_f)
        concated_v = functions.concat((a_f_r, p_f_r, g_f), axis=1)
        return concated_v
