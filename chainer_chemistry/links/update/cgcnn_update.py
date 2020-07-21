import chainer
from chainer import links, functions  # NOQA


class CGCNNUpdate(chainer.Chain):
    """Update submodule for CGCNN

    Args:
        n_site_features (int): hidden dimension of atom feature vector.
            This value must be the same as n_site_feat.
    """

    def __init__(self, n_site_features=64):
        super(CGCNNUpdate, self).__init__()
        with self.init_scope():
            self.fc = links.Linear(None, 2*n_site_features)
            self.bn1 = links.BatchNormalization(2*n_site_features)
            self.bn2 = links.BatchNormalization(n_site_features)

    def __call__(self, site_feat, nbr_feat, nbr_feat_idx):
        n_site, n_nbr, n_nbr_feat = nbr_feat.shape
        _, n_site_feat = site_feat.shape
        site_nbr_feat = site_feat[nbr_feat_idx]
        total_feat = functions.concat([
            functions.broadcast_to(site_feat[:, None, :],
                                   (n_site, n_nbr, n_site_feat)),
            site_nbr_feat,
            nbr_feat
        ], axis=2)

        total_feat = self.fc(total_feat.reshape(
            n_site*n_nbr, 2*n_site_feat+n_nbr_feat))
        total_feat = self.bn1(total_feat).reshape(n_site, n_nbr, 2*n_site_feat)
        feat_gate, feat_core = functions.split_axis(total_feat, 2, axis=-1)
        feat_gate = functions.sigmoid(feat_gate)
        feat_core = functions.softplus(feat_core)
        feat_sum = functions.sum(feat_gate * feat_core, axis=1)
        feat_sum = self.bn2(feat_sum)
        out = functions.softplus(site_feat + feat_sum)
        return out
