import chainer
from chainer import functions
from chainer import links

from chainer_chemistry.links import GraphLinear


class WarpGateUnit(chainer.Chain):
    def __init__(self, output_type='graph', hidden_dim=16, n_layers=4,
                 dropout_ratio=-1):
        super(WarpGateUnit, self).__init__()
        if output_type == 'graph':
            LinearFunc = GraphLinear
        elif output_type == 'super':
            LinearFunc = links.Linear
        else:
            raise ValueError

        with self.init_scope():
            self.H = chainer.ChainList(
                *[LinearFunc(in_size=hidden_dim, out_size=hidden_dim)
                  for _ in range(n_layers)]
            )
            self.G = chainer.ChainList(
                *[LinearFunc(in_size=hidden_dim, out_size=hidden_dim)
                  for _ in range(n_layers)]
            )

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio
        self.output_type = output_type

    def __call__(self, h, g, step=0):
        z = self.H[step](h) + self.G[step](g)

        if self.dropout_ratio > 0.0:
            # TODO: fail backward test
            z = functions.dropout(z, ratio=self.dropout_ratio)
        z = functions.sigmoid(z)
        merged = (1 - z) * h + z * g
        return merged


class SuperNodeTransmitterUnit(chainer.Chain):
    def __init__(self, hidden_dim_super=16, hidden_dim=16, n_layers=4, dropout_ratio=-1):
        super(SuperNodeTransmitterUnit, self).__init__()
        with self.init_scope():
            self.F_super = chainer.ChainList(
                *[links.Linear(in_size=hidden_dim_super,
                               out_size=hidden_dim)
                  for _ in range(n_layers)]
            )
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio

    def __call__(self, g, n_atoms, step=0):
        mb = len(g)
        # for local updates
        g_trans = self.F_super[step](g)
        # intermediate_h_super.shape == (mb, self.hidden_dim)
        g_trans = functions.tanh(g_trans)
        # intermediate_h_super.shape == (mb, 1, self.hidden_dim)
        g_trans = functions.expand_dims(g_trans, 1)
        # intermediate_h_super.shape == (mb, atom, self.hidden_dim)
        g_trans = functions.broadcast_to(g_trans, (mb, n_atoms, self.hidden_dim))
        return g_trans


class GraphTransmitterUnit(chainer.Chain):
    def __init__(self, hidden_dim_super=16, hidden_dim=16, n_heads=8,
                 n_layers=4, dropout_ratio=-1):
        super(GraphTransmitterUnit, self).__init__()
        with self.init_scope():
            self.V_super = chainer.ChainList(
                *[links.Linear(hidden_dim * n_heads, hidden_dim * n_heads)
                  for _ in range(n_layers)]
            )
            self.W_super = chainer.ChainList(
                *[links.Linear(hidden_dim * n_heads, hidden_dim_super)
                  for _ in range(n_layers)]
            )
            self.B = chainer.ChainList(
                *[GraphLinear(n_heads * hidden_dim, n_heads * hidden_dim_super)
                  for _ in range(n_layers)]
            )
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.n_layers = n_layers
        self.dropout_ratio = dropout_ratio
        self.n_heads = n_heads

    def __call__(self, h, g, step=0):
        mb, atom, ch = h.shape
        # h1.shape == (mb, atom, 1, ch)
        h1 = functions.expand_dims(h, 2)
        # h1.shape == (mb, atom, n_heads, ch)
        h1 = functions.broadcast_to(h1, [mb, atom, self.n_heads, ch])
        # h1.shape == (mb, atom, n_heads * ch)
        h1 = functions.reshape(h1, [mb, atom, self.n_heads * ch])

        h_j = functions.expand_dims(h, 1)
        # h_j.shape == (mb, self.n_heads, atom, ch)
        h_j = functions.broadcast_to(h_j, (mb, self.n_heads, atom, ch))

        # expand h_super
        # g_extend.shape (mb, 1, self.hidden_dim_super)
        g_extend = functions.expand_dims(g, 1)
        # g_extend.shape == (mb, self.n_heads, self.hidden_dim_super)
        g_extend = functions.broadcast_to(g_extend, (mb, self.n_heads,
                                                     self.hidden_dim_super))
        # g_extend.shape == (mb, self.n_heads, 1, self.hidden_dim_super)
        g_extend = functions.expand_dims(g_extend, 2)

        # update for attention-message B h_i
        # h1.shape == (mb, atom, n_heads * ch)
        # Bh_i.shape == (mb, atom, self.n_heads * self.hidden_dim_super)
        Bh_i = self.B[step](h1)
        # Bh_i.shpae == (mb, atom, num_head, ch)
        Bh_i = functions.reshape(Bh_i, (mb, atom, self.n_heads,
                                        self.hidden_dim_super))
        # Bh_i.shape == (mb, num_head, atom, ch)
        Bh_i = functions.transpose(Bh_i, [0, 2, 1, 3])

        # take g^{T} * B * h_i
        # indexed by i
        # mb, self.n_haeds atom(i)
        # b_hi.shape == (mb, self.n_heads, 1, atom)
        # This will reduce the last hidden_dim_super axis
        b_hi = functions.matmul(g_extend, Bh_i, transb=True)

        # softmax. sum/normalize over the last axis.
        # mb, self.n_heda, atom(i-normzlied)
        # attention_i.shape == (mb, self.n_heads, 1, atom)
        attention_i = functions.softmax(b_hi, axis=3)
        if self.dropout_ratio > 0.0:
            attention_i = functions.dropout(attention_i,
                                            ratio=self.dropout_ratio)

        # element-wise product --> sum over i
        # mb, num_head, hidden_dim_super
        # attention_sum.shape == (mb, self.n_heads, 1, ch)
        attention_sum = functions.matmul(attention_i, h_j)
        # attention_sum.shape == (mb, self.n_heads * ch)
        attention_sum = functions.reshape(attention_sum,
                                          (mb, self.n_heads * ch))

        # weighting h for different heads
        # intermediate_h.shape == (mb, self.n_heads * ch)
        h_trans = self.V_super[step](attention_sum)
        # compress heads
        h_trans = self.W_super[step](h_trans)
        # intermediate_h.shape == (mb, self.hidden_dim_super)
        h_trans = functions.tanh(h_trans)
        return h_trans


class GWM(chainer.Chain):
    """
    Graph Warping Module (GWM)

    See: Ishiguro, Maeda, and Koyama. "Graph Warp Module: an Auxiliary Module
        for Boosting the Power of Graph NeuralNetworks", arXiv, 2019.

    Args:
        hidden_dim (default=16): dimension of hidden vectors
            associated to each atom (local node)
        hidden_dim_super(default=16); dimension of super-node hidden vector
        n_layers (default=4): number of layers
        n_heads (default=8): numbef of heads
        n_atom_types (default=MAX_ATOMIC_NUM): number of types of atoms
        n_super_feature (default: tuned according to gtn_preprocessor):
            number of super-node observation attributes
        n_edge_types (int): number of edge types witin graphs.
        dropout_ratio (default=0.5); if > 0.0, perform dropout
        tying_flag (default=false): enable if you want to share params across
            layers
    """
    NUM_EDGE_TYPE = 4

    def __init__(self, hidden_dim=16, hidden_dim_super=16, n_layers=4,
                 n_heads=8, dropout_ratio=-1, concat_hidden=False,
                 tying_flag=False):
        super(GWM, self).__init__()
        num_layer = n_layers
        if tying_flag:
            num_layer = 1

        with self.init_scope():
            self.update_super = chainer.ChainList(
                *[links.Linear(in_size=hidden_dim_super,
                               out_size=hidden_dim_super)
                  for _ in range(num_layer)]
            )

            # for Transmitter unit
            self.super_transmitter = SuperNodeTransmitterUnit(
                hidden_dim=hidden_dim, hidden_dim_super=hidden_dim_super,
                n_layers=n_layers, dropout_ratio=dropout_ratio)
            self.graph_transmitter = GraphTransmitterUnit(
                hidden_dim=hidden_dim, hidden_dim_super=hidden_dim_super,
                n_layers=n_layers, n_heads=n_heads, dropout_ratio=dropout_ratio)

            # for Warp Gate unit
            self.wgu_local = WarpGateUnit(output_type='graph', hidden_dim=hidden_dim,
                                          n_layers=n_layers, dropout_ratio=dropout_ratio)
            self.wgu_super = WarpGateUnit(output_type='super', hidden_dim=hidden_dim_super,
                                          n_layers=n_layers, dropout_ratio=dropout_ratio)

            # GRU's. not layer-wise (recurrent through layers)
            self.GRU_local = links.GRU(in_size=hidden_dim, out_size=hidden_dim)
            self.GRU_super = links.GRU(in_size=hidden_dim_super,
                                       out_size=hidden_dim_super)
        # end init_scope-with
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.concat_hidden = concat_hidden
        self.tying_flag = tying_flag

    def __call__(self, h, h_new, g, step=0):
        """
        Describes the module for a single layer update.
        Do not forget to rest GRU for each batch...

        :param h: minibatch by num_nodes by hidden_dim numpy array.
                current local node hidden states as input of the vanilla GNN
        :param h_new: minibatch by num_nodes by hidden_dim numpy array.
                updated local node hidden states as output from the vanilla GNN
        :param adj: minibatch by bond_types by num_nodes by num_nodes 1/0
                array. Adjacency matrices over several bond types
        :param g: minibatch by hidden_dim_super numpy array.
                current super node hiddden state
        :param step: integer, the layer index
        :return: updated h and g
        """
        # (minibatch, atom, ch)
        mb, atom, ch = h.shape
        # non linear update of the super node
        g_new = functions.relu(self.update_super[step](g))

        # Transmitter unit: inter-module message passing
        # original --> super transmission
        h_trans = self.graph_transmitter(h, g, step=step)
        # g_trans: super --> original transmission
        g_trans = self.super_transmitter(g, atom, step=step)

        # Warp Gate unit
        merged_h = self.wgu_local(h_new, g_trans, step=step)
        merged_g = self.wgu_super(h_trans, g_new, step=step)

        # Self recurrent
        out_h = functions.reshape(merged_h, (mb * atom, self.hidden_dim))
        out_h = self.GRU_local(out_h)
        out_h = functions.reshape(out_h, (mb, atom, self.hidden_dim))

        out_g = self.GRU_super(merged_g)

        return out_h, out_g

    def reset_state(self):
        self.GRU_local.reset_state()
        self.GRU_super.reset_state()
