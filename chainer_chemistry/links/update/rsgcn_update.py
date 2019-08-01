import chainer

import chainer_chemistry
from chainer_chemistry.links.connection.graph_linear import GraphLinear


class RSGCNUpdate(chainer.Chain):
    """RSGCN submodule for message and update part.

    Args:
        in_channels (int or None): input channel dimension
        out_channels (int): output channel dimension
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(RSGCNUpdate, self).__init__()
        with self.init_scope():
            self.graph_linear = GraphLinear(
                in_channels, out_channels, nobias=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, h, adj, **kwargs):
        # --- Message part ---
        h = chainer_chemistry.functions.matmul(adj, h)
        # --- Update part ---
        h = self.graph_linear(h)
        return h
