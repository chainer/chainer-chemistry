import chainer
from chainer import links

from chainer_chemistry.links.readout.cgcnn_readout import CGCNNReadout
from chainer_chemistry.links.update.cgcnn_update import CGCNNUpdate


class CGCNN(chainer.Chain):
    """CGCNN

    See Tian Xie et al, \
        Crystal Graph Convolutional Neural Networks for an Accurate and
        Interpretable Prediction of Material Properties. \
        `arXiv:1710.10324 <https://arxiv.org/abs/1710.10324>`_

    Args:
        out_dim (int): dimension of output feature vector
        n_update_layers (int): number of CGCNNUpdate layers
        n_atom_features (int): hidden dimension of atom feature vector
    """

    def __init__(self, out_dim=128, n_update_layers=3, n_atom_features=64):
        super(CGCNN, self).__init__()
        with self.init_scope():
            self.atom_feature_embedding = links.Linear(None, n_atom_features)
            self.crystal_convs = chainer.ChainList(
                *[CGCNNUpdate(n_atom_features) for _ in range(n_update_layers)]
            )
            self.readout = CGCNNReadout(out_dim=out_dim)

    def __call__(self, atom_feat, nbr_feat, atom_idx, feat_idx):
        # atom feature embedding
        atom_feat = self.atom_feature_embedding(atom_feat)
        # --- CGCNN update ---
        for conv_layer in self.crystal_convs:
            atom_feat = conv_layer(atom_feat, nbr_feat, feat_idx)
        # --- CGCNN readout ---
        pool = self.readout(atom_feat, atom_idx)
        return pool
