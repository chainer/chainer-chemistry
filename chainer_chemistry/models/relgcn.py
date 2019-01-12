import chainer
from chainer import cuda
from chainer import functions

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.connection.graph_linear import GraphLinear
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.relgcn_update import RelGCNUpdate


def rescale_adj(adj):
    """Normalize adjacency matrix
    It ensures that activations are on a similar scale irrespective of
    the number of neighbors

    Args:
        adj (:class:`chainer.Variable`, or :class:`numpy.ndarray` \
        or :class:`cupy.ndarray`):
            adjacency matrix

    Returns:
        :class:`chainer.Variable`: normalized adjacency matrix

    """
    xp = cuda.get_array_module(adj)
    num_neighbors = functions.sum(adj, axis=(1, 2))
    base = xp.ones(num_neighbors.shape, dtype=xp.float32)
    cond = num_neighbors.data != 0
    num_neighbors_inv = 1 / functions.where(cond, num_neighbors, base)
    return adj * functions.broadcast_to(
        num_neighbors_inv[:, None, None, :], adj.shape)


class RelGCN(chainer.Chain):

    """Relational GCN (RelGCN)

    See: Michael Schlichtkrull+, \
        Modeling Relational Data with Graph Convolutional Networks. \
        March 2017. \
        `arXiv:1703.06103 <https://arxiv.org/abs/1703.06103>`

    Args:
        out_channels (int): dimension of output feature vector
        num_edge_type (int): number of types of edge
        ch_list (list): channels of each update layer
        n_atom_types (int): number of types of atoms
        input_type (str): type of input vector
        scale_adj (bool): If ``True``, then this network normalizes
            adjacency matrix
    """

    def __init__(self, out_channels=64, num_edge_type=4, ch_list=None,
                 n_atom_types=MAX_ATOMIC_NUM, input_type='int',
                 scale_adj=False):

        super(RelGCN, self).__init__()
        if ch_list is None:
            ch_list = [16, 128, 64]
        with self.init_scope():
            if input_type == 'int':
                self.embed = EmbedAtomID(out_size=ch_list[0],
                                         in_size=n_atom_types)
            elif input_type == 'float':
                self.embed = GraphLinear(None, ch_list[0])
            else:
                raise ValueError("[ERROR] Unexpected value input_type={}"
                                 .format(input_type))
            self.rgcn_convs = chainer.ChainList(*[
                RelGCNUpdate(ch_list[i], ch_list[i+1], num_edge_type)
                for i in range(len(ch_list)-1)])
            self.rgcn_readout = GGNNReadout(
                out_dim=out_channels, hidden_dim=ch_list[-1],
                nobias=True, activation=functions.tanh)
        # self.num_relations = num_edge_type
        self.input_type = input_type
        self.scale_adj = scale_adj

    def __call__(self, x, adj):
        """

        Args:
            x: (batchsize, num_nodes, in_channels)
            adj: (batchsize, num_edge_type, num_nodes, num_nodes)

        Returns: (batchsize, out_channels)

        """
        if x.dtype == self.xp.int32:
            assert self.input_type == 'int'
        else:
            assert self.input_type == 'float'
        h = self.embed(x)  # (minibatch, max_num_atoms)
        if self.scale_adj:
            adj = rescale_adj(adj)
        for rgcn_conv in self.rgcn_convs:
            h = functions.tanh(rgcn_conv(h, adj))
        h = self.rgcn_readout(h)
        return h
