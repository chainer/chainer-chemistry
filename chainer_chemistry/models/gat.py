import numpy
import chainer
# from chainer import cuda
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
    # NUM_EDGE_TYPE = 4

    def __init__(self, out_dim, hidden_dim=16, heads=2, negative_slope=0.2,
                 n_layers=4, n_atom_types=MAX_ATOMIC_NUM, concat_hidden=False,
                 weight_tying=True):
        super(GraphAttentionNetworks, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        # n_message_layer = 1 if weight_tying else n_layers
        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)
            self.weight = GraphLinear(hidden_dim, heads * hidden_dim)
            self.att_weight = GraphLinear(hidden_dim * 2, 1)
            # self.message_layers = chainer.ChainList(
            #     *[GraphLinear(hidden_dim, self.NUM_EDGE_TYPE * hidden_dim)
            #       for _ in range(n_message_layer)]
            # )
            # self.update_layer = links.GRU(2 * hidden_dim, hidden_dim)
            # Readout
            self.i_layers = chainer.ChainList(
                *[GraphLinear(3 * hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )
            self.j_layers = chainer.ChainList(
                *[GraphLinear(2 * hidden_dim, out_dim)
                    for _ in range(n_readout_layer)]
            )
        self.out_dim = out_dim
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying
        self.negative_slope = negative_slope

    def readout(self, h, h0, step=0):
        # --- Readout part ---
        index = step if self.concat_hidden else 0
        # h, h0: (minibatch, atom, ch)
        g = functions.sigmoid(
            self.i_layers[index](functions.concat((h, h0), axis=2))) \
            * self.j_layers[index](h)
        print(g.shape)
        g = functions.sum(g, axis=1)  # sum along atom's axis
        print(g.shape)
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
            x = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            x = atom_array

        # (minibatch, atom, channel)
        x = self.embed(atom_array)
        z0 = functions.copy(x, -1)
        # (minibatch, atom, channel)
        mb, atom, ch = x.shape
        # (minibatch, atom, heads * out_dim)
        test = self.weight(x)

        # concat all pairs of atom
        # (minibatch, 1, atom, heads * out_dim)
        x = functions.expand_dims(test, axis=1)
        # (minibatch, atom, atom, heads * out_dim)
        x = functions.broadcast_to(x, (mb, atom, atom,
                                       self.heads * self.hidden_dim))
        y = functions.copy(x, -1)
        y = functions.transpose(y, (0, 2, 1, 3))
        # (minibatch, atom, atom, heads * out_dim)
        x = functions.broadcast_to(x, (mb, atom, atom,
                                       self.heads * self.hidden_dim))
        # (minibatch, atom, atom, heads * out_dim * 2)
        z = functions.concat([x, y], axis=3)

        # (minibatch * heads, atom, atom, out_dim * 2)
        z = functions.reshape(z, (mb * self.heads, atom * atom,
                                  self.hidden_dim * 2))
        # (minibatch * heads, atom, atom, 1)
        z = self.att_weight(z)
        # (minibatch * heads, atom, atom)
        z = functions.reshape(z, (mb * self.heads, atom, atom))
        z = functions.leaky_relu(z)
        z = functions.reshape(z, (self.heads, mb, atom, atom))

        if isinstance(adj, Variable):
            w_adj = adj.data
        else:
            w_adj = adj
        w_adj = Variable(w_adj, requires_grad=False)
        cond = w_adj.array.astype(numpy.bool)
        cond = numpy.broadcast_to(cond, z.array.shape)
        z = functions.where(cond, z,
                            numpy.broadcast_to(numpy.array(-10000),
                                               z.array.shape)
                            .astype(numpy.float32))
        z = functions.softmax(z)
        # (minibatch, atom, atom)
        z = functions.mean(z, axis=0)
        # (minibatch, atom, out_dim)
        z = functions.matmul(z, test)
        z = self.readout(z, z0)
        return z
