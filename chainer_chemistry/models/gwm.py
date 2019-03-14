import chainer
from chainer import functions
from chainer import links

from chainer_chemistry.links import GraphLinear


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
                 n_heads=8, dropout_ratio=0.5, concat_hidden=False,
                 tying_flag=False):
        super(GWM, self).__init__()
        num_layer = n_layers
        if tying_flag:
            num_layer = 1

        with self.init_scope():
            #
            # for Transmitter unit
            #
            self.F_super = chainer.ChainList(
                *[links.Linear(in_size=hidden_dim_super,
                               out_size=hidden_dim_super)
                  for _ in range(num_layer)]
            )
            self.V_super = chainer.ChainList(
                *[links.Linear(hidden_dim * n_heads, hidden_dim * n_heads)
                  for _ in range(num_layer)]
            )
            self.W_super = chainer.ChainList(
                *[links.Linear(hidden_dim * n_heads, hidden_dim_super)
                  for _ in range(num_layer)]
            )
            self.B = chainer.ChainList(
                *[GraphLinear(n_heads * hidden_dim, n_heads * hidden_dim_super)
                  for _ in range(num_layer)]
            )

            #
            # for Warp Gate unit
            #
            self.gate_dim = hidden_dim
            self.H_local = chainer.ChainList(
                *[GraphLinear(in_size=hidden_dim, out_size=self.gate_dim)
                  for _ in range(num_layer)]
            )
            self.G_local = chainer.ChainList(
                *[GraphLinear(in_size=hidden_dim_super, out_size=self.gate_dim)
                  for _ in range(num_layer)]
            )

            self.gate_dim_super = hidden_dim_super
            self.H_super = chainer.ChainList(
                *[links.Linear(in_size=hidden_dim,
                               out_size=self.gate_dim_super)
                  for _ in range(num_layer)]
            )
            self.G_super = chainer.ChainList(
                *[links.Linear(in_size=hidden_dim_super,
                               out_size=self.gate_dim_super)
                  for _ in range(num_layer)]
            )

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
        #
        # Transmitter unit: inter-module message passing
        #
        # non linear update of the super node
        g_new = functions.relu(self.F_super[step](g))

        # original --> super transmission

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

        # g_trans: super --> original transmission

        # for local updates
        g_trans = self.F_super[step](g)
        # intermediate_h_super.shape == (mb, self.hidden_dim)
        g_trans = functions.tanh(g_trans)
        # intermediate_h_super.shape == (mb, 1, self.hidden_dim)
        g_trans = functions.expand_dims(g_trans, 1)
        # intermediate_h_super.shape == (mb, atom, self.hidden_dim)
        g_trans = functions.broadcast_to(g_trans, (mb, atom, self.hidden_dim))

        #
        # Warp Gate unit
        #
        z_local = self.H_local[step](h_new) + self.G_local[step](g_trans)
        z_local = functions.broadcast_to(z_local, (mb, atom, self.hidden_dim))
        if self.dropout_ratio > 0.0:
            z_local = functions.dropout(z_local, ratio=self.dropout_ratio)
        z_local = functions.sigmoid(z_local)
        #  new_h.shape == (mb, atom, ch)
        merged_h = (1.0-z_local) * h_new + z_local * g_trans

        z_super = self.H_super[step](h_trans) + self.G_super[step](g_new)
        z_super = functions.broadcast_to(z_super, (mb, self.hidden_dim_super))
        if self.dropout_ratio > 0.0:
            z_super = functions.dropout(z_super, ratio=self.dropout_ratio)
        z_super = functions.sigmoid(z_super)
        merged_g = (1.0-z_super) * h_trans + z_super * g_new
        # assert out_h_super.shape==(mb, self.hidden_dim_super)

        #
        # Self recurrent
        #
        out_h = functions.reshape(merged_h, (mb * atom, self.hidden_dim))
        out_h = self.GRU_local(out_h)
        out_h = functions.reshape(out_h, (mb, atom, self.hidden_dim))

        out_g = self.GRU_super(merged_g)

        return out_h, out_g
