import chainer
from chainer import functions, links


from chainer_chemistry.links.connection.graph_linear import GraphLinear
from chainer_chemistry.functions.activation.softplus import improved_softplus


class DenseLayerWithGraphLinear(chainer.Chain):
    def __init__(self, hidden_dim=[64, 32]):
        super(DenseLayerWithGraphLinear, self).__init__()
        self.n_layers = len(hidden_dim)
        with self.init_scope():
            self.update_layer = chainer.ChainList(
                *[GraphLinear(None, hidden_dim[i]) for i in range(self.n_layers)])
        self.activation = improved_softplus

    def __call__(self, v):
        for i in range(self.n_layers):
            v = self.activation(self.update_layer[i](v))
        return v


class DenseLayerWithLinear(chainer.Chain):
    def __init__(self, hidden_dim=[64, 32]):
        super(DenseLayerWithLinear, self).__init__()
        self.n_layers = len(hidden_dim)
        with self.init_scope():
            self.update_layer = chainer.ChainList(
                *[links.Linear(None, hidden_dim[i]) for i in range(self.n_layers)])
        self.activation = improved_softplus

    def __call__(self, v):
        for i in range(self.n_layers):
            v = self.activation(self.update_layer[i](v))
        return v


class UpdateLayerWithGraphLinear(chainer.Chain):
    def __init__(self, hidden_dim=[64, 64, 32]):
        super(UpdateLayerWithGraphLinear, self).__init__()
        self.n_layers = len(hidden_dim)
        with self.init_scope():
            self.update_layer = chainer.ChainList(
                *[GraphLinear(None, hidden_dim[i]) for i in range(self.n_layers)])

    def __call__(self, concated_vector):
        v = concated_vector
        for i in range(self.n_layers):
            v = self.update_layer[i](v)
            # doesn't pass the activation at the last layer
            if i != (self.n_layers-1):
                v = improved_softplus(v)
        return v


class UpdateLayerWithLinear(chainer.Chain):
    def __init__(self, hidden_dim=[64, 64, 32]):
        super(UpdateLayerWithLinear, self).__init__()
        self.n_layers = len(hidden_dim)
        with self.init_scope():
            self.update_layer = chainer.ChainList(
                *[links.Linear(None, hidden_dim[i]) for i in range(self.n_layers)])

    def __call__(self, concated_vector):
        v = concated_vector
        for i in range(self.n_layers):
            v = self.update_layer[i](v)
            # doesn't pass the activation at the last layer
            if i != (self.n_layers-1):
                v = improved_softplus(v)
        return v


def get_concated_edge_vec(atom_feat, pair_feat, global_feat, bond_num, bond_idx, xp):
    a0, a1, a2 = atom_feat.shape
    p0, p1, p2 = pair_feat.shape
    g0, g1 = global_feat.shape

    reshaped_atom_feat = functions.reshape(atom_feat, (a0 * a1, a2))
    reshaped_pair_feat = functions.reshape(pair_feat, (p0 * p1, p2))

    # get source and target nodes of all edges
    base_idx = xp.broadcast_to(xp.arange(0, p0 * a1, a1).reshape(-1, 1), (p0, p1))
    start_idx = (bond_idx[:, 0, :] + base_idx).reshape(-1)
    end_idx = (bond_idx[:, 1, :] + base_idx).reshape(-1)
    start_node = reshaped_atom_feat[start_idx]
    end_node = reshaped_atom_feat[end_idx]

    # concating
    reshaped_global_feat = functions.tile(global_feat, (p1, 1))
    concated_vec = functions.concat((reshaped_pair_feat, 
                                     start_node, end_node, reshaped_global_feat))

    # masking
    mask = xp.zeros((p0 * p1), dtype=xp.bool)
    for i in range(p0): mask[xp.arange(p1 * i, (p1 * i + bond_num[i]))] = True
    mask = xp.broadcast_to(mask.reshape(-1, 1), concated_vec.shape)
    mask_val = xp.zeros(concated_vec.shape, dtype=xp.float32)
    masked_vec = functions.where(mask, concated_vec, mask_val)
    reshaped_masked_vec = masked_vec.reshape(p0, p1, -1)

    return reshaped_masked_vec


def get_concated_node_vec(atom_feat, pair_feat, global_feat, atom_num, bond_num, bond_idx, xp):
    a0, a1, a2 = atom_feat.shape
    p0, p1, p2 = pair_feat.shape
    g0, g1 = global_feat.shape

    reshaped_atom_feat = functions.reshape(atom_feat, (a0 * a1, a2))
    reshaped_pair_feat = functions.reshape(pair_feat, (p0 * p1, p2))

    # get mean edge feature of all nodes
    ## 1. get source and target node idx masked with -1 
    base_idx = xp.broadcast_to(xp.arange(0, p0 * a1, a1).reshape(-1, 1), (p0, p1))
    start_idx = (bond_idx[:, 0, :] + base_idx).reshape(-1)
    end_idx = (bond_idx[:, 1, :] + base_idx).reshape(-1)
    mask = xp.ones((p0 * p1), dtype=xp.bool)
    for i in range(p0): mask[xp.arange(p1 * i, (p1 * i + bond_num[i]))] = False
    start_idx[mask] = -1
    end_idx[mask] = -1
    ## 2. get sum edge feature of all nodes using scatter_add method
    zero = xp.zeros((a0 * a1 + 1, p2), dtype=xp.float32)
    sum_edeg_vec = functions.scatter_add(zero, start_idx, reshaped_pair_feat) + \
                functions.scatter_add(zero, end_idx, reshaped_pair_feat)
    ## 3. get degree of all nodes using scatter_add method
    degree_zero = xp.zeros((a0 * a1 + 1), dtype=xp.float32)
    degree_one = xp.ones((p0 * p1), dtype=xp.float32)
    degree = functions.scatter_add(degree_zero, start_idx, degree_one) + \
                 functions.scatter_add(degree_zero, end_idx, degree_one)
    ## 4. get mean edge feature of all nodes
    degree = degree[:-1] + 1e-10
    sum_edeg_vec = sum_edeg_vec[:-1]
    mean_edge_vec = sum_edeg_vec / functions.broadcast_to(degree.reshape(-1, 1), 
                                                          sum_edeg_vec.shape)

    # concat
    reshaped_global_feat = functions.tile(global_feat, (a1, 1))
    concated_vec = functions.concat((reshaped_atom_feat,
                                     mean_edge_vec, reshaped_global_feat))

    # concating
    mask = xp.zeros((a0 * a1), dtype=xp.bool)
    for i in range(a0): mask[xp.arange(a1 * i, (a1 * i + atom_num[i]))] = True
    mask = xp.broadcast_to(mask.reshape(-1, 1), concated_vec.shape)
    mask_val = xp.zeros(concated_vec.shape, dtype=xp.float32)
    masked_vec = functions.where(mask, concated_vec, mask_val)
    reshaped_masked_vec = masked_vec.reshape(a0, a1, -1)

    return reshaped_masked_vec


class MEGNetUpdate(chainer.Chain):
    """Update submodule for MEGNet

    Args:
        hidden_dim_for_dense (list): dimension list of dense layer
        hidden_dim_for_update (list): dimension list of update layer
    """

    def __init__(self, hidden_dim_for_dense=[64, 32], hidden_dim_for_update=[64, 64, 32]):
        super(MEGNetUpdate, self).__init__()
        if len(hidden_dim_for_dense) != 2:
            raise ValueError('hidden_dim_for_dense must have 2 elements')

        if len(hidden_dim_for_update) != 3:
            raise ValueError('hidden_dim_for_dense must have 3 elements')

        with self.init_scope():
            # for dense layer
            self.dense_for_atom = DenseLayerWithGraphLinear(hidden_dim=hidden_dim_for_dense)
            self.dense_for_pair = DenseLayerWithGraphLinear(hidden_dim=hidden_dim_for_dense)
            self.dense_for_global = DenseLayerWithLinear(hidden_dim=hidden_dim_for_dense)
            # for update layer
            self.update_for_atom = UpdateLayerWithGraphLinear(hidden_dim=hidden_dim_for_update)
            self.update_for_pair = UpdateLayerWithGraphLinear(hidden_dim=hidden_dim_for_update)
            self.update_for_global = UpdateLayerWithLinear(hidden_dim=hidden_dim_for_update)

    def __call__(self, atoms_feat, pair_feat, global_feat, atom_num, bond_num, bond_idx):
        # 1) Pass the Dense layer
        a_f_d = self.dense_for_atom(atoms_feat)
        p_f_d = self.dense_for_pair(pair_feat)
        g_f_d = self.dense_for_global(global_feat)

        # 2) Update the edge vector
        concat_p_v = get_concated_edge_vec(a_f_d, p_f_d, g_f_d, bond_num, bond_idx, self.xp)
        update_p = self.update_for_atom(concat_p_v)

        # 3) Update the node vector
        concat_a_v = get_concated_node_vec(a_f_d, update_p, g_f_d, atom_num, bond_num, bond_idx, self.xp)
        update_a = self.update_for_pair(concat_a_v)

        # 4) Update the global vector
        ave_p = functions.mean(update_p, axis=1)
        ave_a = functions.mean(update_a, axis=1)
        concat_g_v = functions.concat((ave_a, ave_p, g_f_d), axis=1)
        update_g = self.update_for_global(concat_g_v)

        # 5) Skip connection
        new_a_f = functions.add(a_f_d, update_a)
        new_p_f = functions.add(p_f_d, update_p)
        new_g_f = functions.add(g_f_d, update_g)

        return new_a_f, new_p_f, new_g_f
