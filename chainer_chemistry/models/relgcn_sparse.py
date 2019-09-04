import chainer
from chainer import functions
from chainer.links import Linear
from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links.readout.scatter_ggnn_readout import ScatterGGNNReadout  # NOQA
from chainer_chemistry.links.connection.embed_atom_id import EmbedAtomID
from chainer_chemistry.links.update.relgcn_sparse_layer import RelGCNSparseLayer  # NOQA


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
                RelGCNSparseLayer(hidden_channels[i], hidden_channels[i + 1],
                                  n_edge_types)
                for i in range(len(hidden_channels) - 1)])
            self.rgcn_readout = ScatterGGNNReadout(
                out_dim=out_dim, in_channels=hidden_channels[-1],
                nobias=True, activation=functions.tanh)
        # self.num_relations = num_edge_type
        self.input_type = input_type
        self.scale_adj = scale_adj

    def __call__(self, graph):
        """
        Args:
            x: (batchsize, num_nodes, in_channels)
            adj: (batchsize, num_edge_type, num_nodes, num_nodes)

        Returns: (batchsize, hidden_channels)
        """
        if graph.x.dtype == self.xp.int32:
            assert self.input_type == 'int'
        else:
            assert self.input_type == 'float'
        h = self.embed(graph.x)  # (minibatch, max_num_atoms)
        if self.scale_adj:
            raise NotImplementedError
        for rgcn_conv in self.rgcn_convs:
            h = functions.tanh(rgcn_conv(h, graph))
        h = self.rgcn_readout(h, graph.batch)
        return h
