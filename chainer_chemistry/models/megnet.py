import chainer
from chainer.backend import get_array_module
from chainer import functions

from chainer_chemistry.functions import megnet_softplus
from chainer_chemistry.links.readout.megnet_readout import MEGNetReadout
from chainer_chemistry.links.update.megnet_update import MEGNetUpdate


def reshaped_feat(feat, idx):
    """Convert node stack pattern into pad pattern

    This method is converting from node stack pattern to pad pattern
    about node and edge feature. This is because the current set2set
    implementation is only focus on pad pattern feature.
    """
    xp = get_array_module(idx)
    max_idx = int(xp.max(idx))
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
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function for megnet model
            `megnet_softplus` was used in original paper.
    """

    def __init__(self, out_dim=32, n_update_layers=3, dropout_ratio=-1,
                 activation=megnet_softplus):
        super(MEGNet, self).__init__()
        if n_update_layers <= 0:
            raise ValueError('n_update_layers must be a positive integer, '
                             'but it was set to {}'.format(n_update_layers))

        self.n_update_layers = n_update_layers
        with self.init_scope():
            self.update_layers = chainer.ChainList(
                *[MEGNetUpdate(
                    dim_for_dense=[64, 32], dim_for_update=[64, 64, 32],
                    dropout_ratio=dropout_ratio, activation=activation,
                    skip_intermediate=(i == 0)
                ) for i in range(n_update_layers)])
            self.readout = MEGNetReadout(out_dim=out_dim, in_channels=32,
                                         n_layers=1, processing_steps=3,
                                         dropout_ratio=dropout_ratio,
                                         activation=activation)

    def __call__(self, atoms_feat, pair_feat, global_feat, *args):
        a_f = atoms_feat
        p_f = pair_feat
        g_f = global_feat
        # --- MGENet update ---
        for i in range(self.n_update_layers):
            a_f, p_f, g_f = self.update_layers[i](a_f, p_f, g_f, *args)
        # --- reshape ---
        atom_idx = args[0]
        pair_idx = args[1]
        a_f = reshaped_feat(a_f, atom_idx)
        p_f = reshaped_feat(p_f, pair_idx)
        # --- MGENet readout ---
        out = self.readout(a_f, p_f, g_f)
        return out
