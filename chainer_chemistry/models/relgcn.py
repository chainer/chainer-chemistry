import chainer
from chainer import functions, cuda  # NOQA

from chainer.links import Linear
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.scatter_ggnn_readout import ScatterGGNNReadout  # NOQA
from chainer_chemistry.links import EmbedAtomID, GraphLinear  # NOQA
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.relgcn_update import RelGCNUpdate, RelGCNSparseUpdate  # NOQA


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
        out_dim (int): dimension of output feature vector
        hidden_channels (None or int or list):
            dimension of feature vector for each node
        n_update_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        n_edge_types (int): number of edge type.
            Defaults to 4 for single, double, triple and aromatic bond.
        scale_adj (bool): If ``True``, then this network normalizes
            adjacency matrix
    """

    def __init__(self, out_dim=64, hidden_channels=None, n_update_layers=None,
                 n_atom_types=MAX_ATOMIC_NUM, n_edge_types=4, input_type='int',
                 scale_adj=False):
        super(RelGCN, self).__init__()
        if hidden_channels is None:
            hidden_channels = [16, 128, 64]
        elif isinstance(hidden_channels, int):
            if not isinstance(n_update_layers, int):
                raise ValueError(
                    'Must specify n_update_layers when hidden_channels is int')
            hidden_channels = [hidden_channels] * n_update_layers
        with self.init_scope():
            if input_type == 'int':
                self.embed = EmbedAtomID(out_size=hidden_channels[0],
                                         in_size=n_atom_types)
            elif input_type == 'float':
                self.embed = GraphLinear(None, hidden_channels[0])
            else:
                raise ValueError("[ERROR] Unexpected value input_type={}"
                                 .format(input_type))
            self.rgcn_convs = chainer.ChainList(*[
                RelGCNUpdate(hidden_channels[i], hidden_channels[i + 1],
                             n_edge_types)
                for i in range(len(hidden_channels) - 1)])
            self.rgcn_readout = GGNNReadout(
                out_dim=out_dim, in_channels=hidden_channels[-1],
                nobias=True, activation=functions.tanh)
        # self.num_relations = num_edge_type
        self.input_type = input_type
        self.scale_adj = scale_adj

    def __call__(self, x, adj):
        """main calculation

        Args:
            x: (batchsize, num_nodes, in_channels)
            adj: (batchsize, num_edge_type, num_nodes, num_nodes)

        Returns: (batchsize, hidden_channels)
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


class RelGCNSparse(chainer.Chain):

    """Relational GCN (RelGCN) Sparse Pattern

    See: Michael Schlichtkrull+, \
        Modeling Relational Data with Graph Convolutional Networks. \
        March 2017. \
        `arXiv:1703.06103 <https://arxiv.org/abs/1703.06103>`

    Args:
        out_dim (int): dimension of output feature vector
        hidden_channels (None or int or list):
            dimension of feature vector for each node
        n_update_layers (int): number of layers
        n_atom_types (int): number of types of atoms
        n_edge_types (int): number of edge type.
            Defaults to 4 for single, double, triple and aromatic bond.
        scale_adj (bool): If ``True``, then this network normalizes
            adjacency matrix
    """

    def __init__(self, out_dim=64, hidden_channels=None, n_update_layers=None,
                 n_atom_types=MAX_ATOMIC_NUM, n_edge_types=4, input_type='int',
                 scale_adj=False):
        super(RelGCNSparse, self).__init__()
        if hidden_channels is None:
            hidden_channels = [16, 128, 64]
        elif isinstance(hidden_channels, int):
            if not isinstance(n_update_layers, int):
                raise ValueError(
                    'Must specify n_update_layers when hidden_channels is int')
            hidden_channels = [hidden_channels] * n_update_layers
        with self.init_scope():
            if input_type == 'int':
                self.embed = EmbedAtomID(out_size=hidden_channels[0],
                                         in_size=n_atom_types)
            elif input_type == 'float':
                self.embed = Linear(None, hidden_channels[0])
            else:
                raise ValueError("[ERROR] Unexpected value input_type={}"
                                 .format(input_type))
            self.rgcn_convs = chainer.ChainList(*[
                RelGCNSparseUpdate(hidden_channels[i], hidden_channels[i + 1],
                                   n_edge_types)
                for i in range(len(hidden_channels) - 1)])
            self.rgcn_readout = ScatterGGNNReadout(
                out_dim=out_dim, in_channels=hidden_channels[-1],
                nobias=True, activation=functions.tanh)
        # self.num_relations = num_edge_type
        self.input_type = input_type
        self.scale_adj = scale_adj

    def __call__(self, sparse_batch):
        """main calculation

        Args:
            x: (batchsize, num_nodes, in_channels)
            adj: (batchsize, num_edge_type, num_nodes, num_nodes)

        Returns: (batchsize, hidden_channels)
        """
        if sparse_batch.x.dtype == self.xp.int32:
            assert self.input_type == 'int'
        else:
            assert self.input_type == 'float'
        h = self.embed(sparse_batch.x)  # (minibatch, max_num_atoms)
        if self.scale_adj:
            raise NotImplementedError
        for rgcn_conv in self.rgcn_convs:
            h = functions.tanh(rgcn_conv(
                h, sparse_batch.edge_index, sparse_batch.edge_attr))
        h = self.rgcn_readout(h, sparse_batch.batch)
        return h
