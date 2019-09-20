import chainer
from chainer import functions, links


from chainer_chemistry.functions import megnet_softplus


class DenseLayer(chainer.Chain):
    def __init__(self, hidden_dim=[64, 32]):
        super(DenseLayer, self).__init__()
        self.n_layers = len(hidden_dim)
        with self.init_scope():
            self.update_layer = chainer.ChainList(
                *[links.Linear(None, hidden_dim[i])
                  for i in range(self.n_layers)])
        self.activation = megnet_softplus

    def __call__(self, v):
        for i in range(self.n_layers):
            v = self.activation(self.update_layer[i](v))
        return v


class UpdateLayer(chainer.Chain):
    def __init__(self, hidden_dim=[64, 64, 32]):
        super(UpdateLayer, self).__init__()
        self.n_layers = len(hidden_dim)
        with self.init_scope():
            self.update_layer = chainer.ChainList(
                *[links.Linear(None, hidden_dim[i])
                    for i in range(self.n_layers)])

    def __call__(self, concated_vector):
        v = concated_vector
        for i in range(self.n_layers):
            v = self.update_layer[i](v)
            # doesn't pass the activation at the last layer
            if i != (self.n_layers-1):
                v = megnet_softplus(v)
        return v


def get_mean_feat(feat, idx, out_shape, xp):
    zero = xp.zeros(out_shape, dtype=xp.float32)
    sum_vec = functions.scatter_add(zero, idx, feat)
    one = xp.ones(feat.shape, dtype=xp.float32)
    degree = functions.scatter_add(zero, idx, one)
    return sum_vec / degree


class MEGNetUpdate(chainer.Chain):
    """Update submodule for MEGNet

    Args:
        hidden_dim_for_dense (list): dimension list of dense layer
        hidden_dim_for_update (list): dimension list of update layer
        dropout_ratio (float): ratio of dropout
    """

    def __init__(self, hidden_dim_for_dense=[64, 32],
                 hidden_dim_for_update=[64, 64, 32],
                 dropout_ratio=-1):
        super(MEGNetUpdate, self).__init__()
        if len(hidden_dim_for_dense) != 2:
            raise ValueError('hidden_dim_for_dense must have 2 elements')

        if len(hidden_dim_for_update) != 3:
            raise ValueError('hidden_dim_for_dense must have 3 elements')

        self.dropout_ratio = dropout_ratio
        with self.init_scope():
            # for dense layer
            self.dense_for_atom = DenseLayer(hidden_dim_for_dense)
            self.dense_for_pair = DenseLayer(hidden_dim_for_dense)
            self.dense_for_global = DenseLayer(hidden_dim_for_dense)
            # for update layer
            self.update_for_atom = UpdateLayer(hidden_dim_for_update)
            self.update_for_pair = UpdateLayer(hidden_dim_for_update)
            self.update_for_global = UpdateLayer(hidden_dim_for_update)

    def __call__(self, atoms_feat, pair_feat, global_feat,
                 atom_idx, pair_idx, start_idx, end_idx):
        # 1) Pass the Dense layer
        a_f_d = self.dense_for_atom(atoms_feat)
        p_f_d = self.dense_for_pair(pair_feat)
        g_f_d = self.dense_for_global(global_feat)

        # 2) Update the edge vector
        start_node = a_f_d[start_idx]
        end_node = a_f_d[end_idx]
        g_f_extend_with_pair_idx = g_f_d[pair_idx]
        concat_p_v = functions.concat((p_f_d, start_node, end_node,
                                       g_f_extend_with_pair_idx))
        update_p = self.update_for_atom(concat_p_v)

        # 3) Update the node vector
        # 1. get sum edge feature of all nodes using scatter_add method
        zero = self.xp.zeros(a_f_d.shape, dtype=self.xp.float32)
        sum_edeg_vec = functions.scatter_add(zero, start_idx, update_p) + \
            functions.scatter_add(zero, end_idx, update_p)
        # 2. get degree of all nodes using scatter_add method
        one = self.xp.ones(p_f_d.shape, dtype=self.xp.float32)
        degree = functions.scatter_add(zero, start_idx, one) + \
            functions.scatter_add(zero, end_idx, one)
        # 3. get mean edge feature of all nodes
        mean_edge_vec = sum_edeg_vec / degree
        # 4. concating
        g_f_extend_with_atom_idx = g_f_d[atom_idx]
        concat_a_v = functions.concat((a_f_d, mean_edge_vec,
                                       g_f_extend_with_atom_idx))
        update_a = self.update_for_pair(concat_a_v)

        # 4) Update the global vector
        out_shape = g_f_d.shape
        ave_p = get_mean_feat(update_p, pair_idx, out_shape, self.xp)
        ave_a = get_mean_feat(update_a, atom_idx, out_shape, self.xp)
        concat_g_v = functions.concat((ave_a, ave_p, g_f_d), axis=1)
        update_g = self.update_for_global(concat_g_v)

        # 5) Skip connection
        new_a_f = functions.add(a_f_d, update_a)
        new_p_f = functions.add(p_f_d, update_p)
        new_g_f = functions.add(g_f_d, update_g)

        # 6) dropout
        if self.dropout_ratio > 0.0:
            new_a_f = functions.dropout(new_a_f, ratio=self.dropout_ratio)
            new_p_f = functions.dropout(new_p_f, ratio=self.dropout_ratio)
            new_g_f = functions.dropout(new_g_f, ratio=self.dropout_ratio)

        return new_a_f, new_p_f, new_g_f
