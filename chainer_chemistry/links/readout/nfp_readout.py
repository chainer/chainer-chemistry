import chainer
from chainer import functions

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class NFPReadout(chainer.Chain):
    """NFP submodule for readout part.

    Args:
        out_dim (int): output dimension of feature vector associated
            to each graph
        in_channels (int or None): dimension of feature vector associated to
            each node
    """

    def __init__(self, out_dim, in_channels):
        super(NFPReadout, self).__init__()
        with self.init_scope():
            self.output_weight = GraphLinear(in_channels, out_dim)
        self.in_channels = in_channels
        self.out_dim = out_dim

    def __call__(self, h, is_real_node=None, **kwargs):
        # h: (minibatch, node, ch)
        # is_real_node: (minibatch, node)

        # ---Readout part ---
        i = self.output_weight(h)
        i = functions.softmax(i, axis=2)  # softmax along channel axis
        if is_real_node is not None:
            # mask virtual node feature to be 0
            mask = self.xp.broadcast_to(
                is_real_node[:, :, None], i.shape)
            i = i * mask
        i = functions.sum(i, axis=1)  # sum along atom's axis
        return i
