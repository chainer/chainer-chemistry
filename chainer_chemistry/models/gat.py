import chainer
import chainer.backends.cuda as cuda
from chainer import functions
from chainer import Variable
# from chainer import links

# import chainer_chemistry
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import GraphLinear


class GraphAttentionNetworks(chainer.Chain):
    """Graph Attention Networks (GAT)

    See: Veličković, Petar, et al. (2017).\
        Graph Attention Networks.\
        `arXiv:1701.10903 <https://arxiv.org/abs/1710.10903>`_

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        weight_tying (bool): enable weight_tying or not

    """
    NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16, heads=2, negative_slope=0.2,
                 n_layers=4, n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 weight_tying=True):
        super(GraphAttentionNetworks, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        # n_message_layer = 1 if weight_tying else n_layers
        n_message_layer = n_layers
        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)
            # self.weight = GraphLinear(hidden_dim, heads * hidden_dim)
            # self.att_weight = GraphLinear(hidden_dim * 2, 1)
            self.message_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim,
                              self.NUM_EDGE_TYPE * heads * hidden_dim)
                  for _ in range(n_message_layer)]
            )
            self.attenstion_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim * 2, 1)
                  for _ in range(n_message_layer)]
            )
            # self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
            # Readout
            self.i_layers = chainer.ChainList(
                *[GraphLinear(2 * hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )
            self.j_layers = chainer.ChainList(
                *[GraphLinear(hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )
        self.out_dim = out_dim
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying
        self.negative_slope = negative_slope

    def update(self, h, adj, step=0):
        xp = self.xp
        # (minibatch, atom, channel)
        mb, atom, ch = h.shape
        # (minibatch, atom, EDGE_TYPE * heads * out_dim)
        h = self.message_layers[step](h)
        # (minibatch, atom, EDGE_TYPE, heads, out_dim)
        h = functions.reshape(h, (mb, atom, self.NUM_EDGE_TYPE, self.heads,
                                  self.hidden_dim))

        # concat all pairs of atom
        # (minibatch, 1, atom, heads, out_dim)
        h_i = functions.reshape(h, (mb, 1, atom, self.NUM_EDGE_TYPE,
                                    self.heads, self.hidden_dim))
        # (minibatch, atom, atom, heads, out_dim)
        h_i = functions.broadcast_to(h_i, (mb, atom, atom, self.NUM_EDGE_TYPE,
                                           self.heads, self.hidden_dim))

        # (minibatch, atom, 1, EDGE_TYPE, heads, out_dim)
        h_j = functions.reshape(h, (mb, atom, 1, self.NUM_EDGE_TYPE,
                                    self.heads, self.hidden_dim))
        # (minibatch, atom, atom, EDGE_TYPE, heads, out_dim)
        h_j = functions.broadcast_to(h_j, (mb, atom, atom, self.NUM_EDGE_TYPE,
                                           self.heads, self.hidden_dim))

        # (minibatch, atom, atom, EDGE_TYPE, heads, out_dim * 2)
        e = functions.concat([h_i, h_j], axis=5)

        # (minibatch, EDGE_TYPE, heads, atom, atom, out_dim * 2)
        e = functions.transpose(e, (0, 3, 4, 1, 2, 5))
        # (minibatch * EDGE_TYPE * heads, atom * atom, out_dim * 2)
        e = functions.reshape(e, (mb * self.NUM_EDGE_TYPE * self.heads,
                                  atom * atom, self.hidden_dim * 2))
        # (minibatch * EDGE_TYPE * heads, atom * atom, 1)
        e = self.attenstion_layers[step](e)

        # (minibatch, EDGE_TYPE, heads, atom, atom)
        e = functions.reshape(e, (mb, self.NUM_EDGE_TYPE, self.heads, atom,
                                  atom))
        e = functions.leaky_relu(e)

        # (minibatch, EDGE_TYPE, atom, atom)
        cond = adj.array.astype(xp.bool)
        # (minibatch, EDGE_TYPE, 1, atom, atom)
        cond = xp.reshape(cond, (mb, self.NUM_EDGE_TYPE, 1, atom, atom))
        # (minibatch, EDGE_TYPE, heads, atom, atom)
        cond = xp.broadcast_to(cond, e.array.shape)
        # TODO(mottodora): find better way to ignore non connected
        e = functions.where(cond, e,
                            xp.broadcast_to(xp.array(-10000), e.array.shape)
                            .astype(xp.float32))
        # (minibatch, heads, atom, atom)
        alpha = functions.softmax(e, axis=4)

        # before: (minibatch, atom, EDGE_TYPE, heads, out_dim)
        # after: (minibatch, EDGE_TYPE, heads, atom, out_dim)
        h = functions.transpose(h, (0, 2, 3, 1, 4))
        # (minibatch, EDGE_TYPE, heads, atom, out_dim)
        h_new = functions.matmul(alpha, h)
        h_new = functions.sum(h_new, axis=1)
        # (minibatch, EDGE_TYPE, atom, out_dim)
        h_new = functions.mean(h_new, axis=1)
        return h_new

    def readout(self, h, h0, step=0, is_real_node=None):
        # --- Readout part ---
        index = step if self.concat_hidden else 0
        # h, h0: (minibatch, atom, ch)
        g = functions.sigmoid(
            self.i_layers[index](functions.concat((h, h0), axis=2))) \
            * self.j_layers[index](h)
        if is_real_node is not None:
            mask = self.xp.broadcast_to(
                is_real_node[:, :, None], g.shape)
            g = functions.where(mask, g, self.xp.zeros(
                g.shape, dtype=self.xp.float32))
        g = functions.sum(g, axis=1)  # sum along atom's axis
        return g

    def __call__(self, atom_array, adj):
        """Forward propagation

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix with edge-type
                information

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        # x: minibatch, atom, channel
        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array

        device_id = cuda.get_device_from_array(h.array).id
        h0 = functions.copy(h, device_id)

        if isinstance(adj, Variable):
            w_adj = adj.data
        else:
            w_adj = adj
        is_real_node = self.xp.sum(w_adj, axis=(1, 2)) > 0
        w_adj = Variable(w_adj, requires_grad=False)

        for step in range(self.n_layers):
            h = self.update(h, w_adj, step)

        g = self.readout(h, h0, is_real_node)
        return g
