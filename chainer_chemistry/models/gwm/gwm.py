import chainer
from chainer import functions
from chainer import links

from chainer_chemistry.links import GraphLinear


class WarpGateUnit(chainer.Chain):
    """WarpGateUnit

    It computes gated-sum mixing `merged` feature from normal node feature `h`
    and super node feature `g`,

    See Section "3.4 Warp Gate" of the paper.

    Args:
        output_type (str): supported type as below.
            graph:
            super:
        hidden_dim (int): hidden dim
        dropout_ratio (float): negative value indicates to not apply dropout.
        activation (callable):
    """
    def __init__(self, output_type='graph', hidden_dim=16,
                 dropout_ratio=-1, activation=functions.sigmoid):
        super(WarpGateUnit, self).__init__()
        if output_type == 'graph':
            LinearLink = GraphLinear
        elif output_type == 'super':
            LinearLink = links.Linear
        else:
            raise ValueError(
                'output_type = {} is unexpected. graph or super is supported.'
                .format(output_type))

        with self.init_scope():
            self.H = LinearLink(in_size=hidden_dim, out_size=hidden_dim)
            self.G = LinearLink(in_size=hidden_dim, out_size=hidden_dim)

        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio
        self.output_type = output_type
        self.activation = activation

    def __call__(self, h, g):
        # TODO(nakago): more efficient computation. Maybe we can calculate
        # self.G(g) as Linear layer followed by broadcast to each atom.
        z = self.H(h) + self.G(g)

        if self.dropout_ratio > 0.0:
            z = functions.dropout(z, ratio=self.dropout_ratio)
        z = self.activation(z)
        merged = (1 - z) * h + z * g
        return merged


class SuperNodeTransmitterUnit(chainer.Chain):
    """SuperNodeTransmitterUnit

    It calculates message from super node to normal node.

    Args:
        hidden_dim_super (int):
        hidden_dim (int): hiddem dim for
        dropout_ratio (float): negative value indicates to not apply dropout.
    """

    def __init__(self, hidden_dim_super=16, hidden_dim=16, dropout_ratio=-1):
        super(SuperNodeTransmitterUnit, self).__init__()
        with self.init_scope():
            self.F_super = links.Linear(in_size=hidden_dim_super,
                                        out_size=hidden_dim)
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.dropout_ratio = dropout_ratio

    def __call__(self, g, n_nodes):
        """main calculation

        Args:
            g: super node feature. shape (bs, hidden_dim_super)
            n_nodes (int): number of nodes

        Returns:
            g_trans: super --> original transmission
        """
        mb = len(g)
        # for local updates
        g_trans = self.F_super(g)
        # intermediate_h_super.shape == (mb, self.hidden_dim)
        g_trans = functions.tanh(g_trans)
        # intermediate_h_super.shape == (mb, 1, self.hidden_dim)
        g_trans = functions.expand_dims(g_trans, 1)
        # intermediate_h_super.shape == (mb, atom, self.hidden_dim)
        g_trans = functions.broadcast_to(g_trans,
                                         (mb, n_nodes, self.hidden_dim))
        return g_trans


class GraphTransmitterUnit(chainer.Chain):
    """GraphTransmitterUnit

    It calculates message from normal node to super node.

    Args:
        hidden_dim_super (int):
        hidden_dim (int):
        n_heads (int):
        dropout_ratio (float):
        activation (callable):
    """
    def __init__(self, hidden_dim_super=16, hidden_dim=16, n_heads=8,
                 dropout_ratio=-1, activation=functions.tanh):
        super(GraphTransmitterUnit, self).__init__()
        hdim_n = hidden_dim * n_heads
        with self.init_scope():
            self.V_super = GraphLinear(hidden_dim, hdim_n)
            self.W_super = links.Linear(hdim_n, hidden_dim_super)
            self.B = GraphLinear(hidden_dim, n_heads * hidden_dim_super)
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.dropout_ratio = dropout_ratio
        self.n_heads = n_heads
        self.activation = activation

    def __call__(self, h, g, step=0):
        mb, atom, ch = h.shape

        h_j = self.V_super(h)
        h_j = functions.reshape(h_j, (mb, atom, self.n_heads, ch))
        # h_j (mb, atom, self.n_heads, ch)
        h_j = functions.transpose(h_j, (0, 2, 1, 3))

        # expand h_super
        # g_extend.shape (mb, 1, self.hidden_dim_super)
        g_extend = functions.expand_dims(g, 1)
        # g_extend.shape == (mb, self.n_heads, self.hidden_dim_super)
        g_extend = functions.broadcast_to(g_extend, (mb, self.n_heads,
                                                     self.hidden_dim_super))
        # g_extend.shape == (mb, self.n_heads, 1, self.hidden_dim_super)
        g_extend = functions.expand_dims(g_extend, 2)

        # update for attention-message B h_i
        # h (mb, atom, ch)
        # Bh_i.shape == (mb, atom, self.n_heads * self.hidden_dim_super)
        Bh_i = self.B(h)
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
        # compress heads
        h_trans = self.W_super(attention_sum)
        # intermediate_h.shape == (mb, self.hidden_dim_super)
        h_trans = self.activation(h_trans)
        return h_trans


class GWM(chainer.Chain):
    """Graph Warping Module (GWM)

    Module for a single layer update.

    See: Ishiguro, Maeda, and Koyama. "Graph Warp Module: an Auxiliary Module
        for Boosting the Power of Graph NeuralNetworks", arXiv, 2019.

    Args:
        hidden_dim (int): dimension of hidden vectors
            associated to each atom (local node)
        hidden_dim_super (int); dimension of super-node hidden vector
        n_layers (int): number of layers
        n_heads (int): number of heads
        dropout_ratio (float): dropout ratio.
            Negative value indicates to not apply dropout.
        tying_flag (bool): enable if you want to share params across layers.
        activation (callable):
        wgu_activation (callable):
        gtu_activation (callable):
    """
    def __init__(self, hidden_dim=16, hidden_dim_super=16, n_layers=4,
                 n_heads=8, dropout_ratio=-1,
                 tying_flag=False, activation=functions.relu,
                 wgu_activation=functions.sigmoid,
                 gtu_activation=functions.tanh):
        super(GWM, self).__init__()

        n_use_layers = 1 if tying_flag else n_layers

        with self.init_scope():
            self.update_super = chainer.ChainList(
                *[links.Linear(in_size=hidden_dim_super,
                               out_size=hidden_dim_super)
                  for _ in range(n_use_layers)]
            )

            # for Transmitter unit
            self.super_transmitter = chainer.ChainList(
                *[SuperNodeTransmitterUnit(
                    hidden_dim=hidden_dim, hidden_dim_super=hidden_dim_super,
                    dropout_ratio=dropout_ratio) for _ in range(n_use_layers)])
            self.graph_transmitter = chainer.ChainList(
                *[GraphTransmitterUnit(
                    hidden_dim=hidden_dim, hidden_dim_super=hidden_dim_super,
                    n_heads=n_heads, dropout_ratio=dropout_ratio,
                    activation=gtu_activation) for _ in range(n_use_layers)])

            # for Warp Gate unit
            self.wgu_local = chainer.ChainList(
                *[WarpGateUnit(
                    output_type='graph', hidden_dim=hidden_dim,
                    dropout_ratio=dropout_ratio, activation=wgu_activation)
                    for _ in range(n_use_layers)])
            self.wgu_super = chainer.ChainList(
                *[WarpGateUnit(
                    output_type='super', hidden_dim=hidden_dim_super,
                    dropout_ratio=dropout_ratio, activation=wgu_activation)
                    for _ in range(n_use_layers)])

            # Weight tying: not layer-wise but recurrent through layers
            self.GRU_local = links.GRU(in_size=hidden_dim, out_size=hidden_dim)
            self.GRU_super = links.GRU(in_size=hidden_dim_super,
                                       out_size=hidden_dim_super)
        # end init_scope-with
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.tying_flag = tying_flag
        self.activation = activation
        self.wgu_activation = wgu_activation

    def __call__(self, h, h_new, g, step=0):
        """main calculation

        Note: Do not forget to reset GRU for each batch.

        Args:
            h: Minibatch by num_nodes by hidden_dim numpy array.
                current local node hidden states as input of the vanilla GNN
            h_new: Minibatch by num_nodes by hidden_dim numpy array.
                updated local node hidden states as output from the vanilla GNN
            g: Minibatch by bond_types by num_nodes by num_nodes 1/0
                array. Adjacency matrices over several bond types
            step: Minibatch by hidden_dim_super numpy array.
                current super node hiddden state

        Returns: Updated h and g
        """
        # (minibatch, atom, ch)
        mb, n_nodes, ch = h.shape
        # non linear update of the super node
        g_new = self.activation(self.update_super[step](g))

        # Transmitter unit: inter-module message passing
        # original --> super transmission
        h_trans = self.graph_transmitter[step](h, g)
        # g_trans: super --> original transmission
        g_trans = self.super_transmitter[step](g, n_nodes)

        # Warp Gate unit
        merged_h = self.wgu_local[step](h_new, g_trans)
        merged_g = self.wgu_super[step](h_trans, g_new)

        # Self recurrent
        out_h = functions.reshape(merged_h, (mb * n_nodes, self.hidden_dim))
        out_h = self.GRU_local(out_h)
        out_h = functions.reshape(out_h, (mb, n_nodes, self.hidden_dim))

        out_g = self.GRU_super(merged_g)

        return out_h, out_g

    def reset_state(self):
        self.GRU_local.reset_state()
        self.GRU_super.reset_state()
