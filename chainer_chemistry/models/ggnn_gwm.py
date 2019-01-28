import chainer
from chainer import cuda
from chainer import functions
from chainer import links

from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.ggnn_update import GGNNUpdate
from chainer_chemistry.models import gwm
from chainer_chemistry.models.gwm import GWM


class GGNN_GWM(chainer.Chain):
    """Gated Graph Neural Networks (GGNN) with Graph Warp Module

    See: Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2015).\
        Gated graph sequence neural networks. \
        `arXiv:1511.05493 <https://arxiv.org/abs/1511.05493>`_

    See: Ishiguro, Maeda, and Koyama. "Graph Warp Module: an Auxiliary Module for Boosting the Power of Graph Neural Networks", arXiv, 2019.

    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
       hiden_dim_super(default=16); dimension of super-node hidden vector 
       n_layers (int): number of layers
        n_heads (default=8): numbef of heads
        n_atom_types (int): number of types of atoms
        n_super_feature (default: tuned according to ggnn_gwm_preprocessor); number of super-node observation attributes
        dropout_ratio (float): if >0.0, perform dropout
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        weight_tying (bool): enable weight_tying or not
        activation (~chainer.Function or ~chainer.FunctionNode):
            activate function
        num_edge_type (int): number of edge type.
            Defaults to 4 for single, double, triple and aromatic bond.
    """

    def __init__(self, out_dim, hidden_dim=16, hidden_dim_super=16, n_layers=4,
                 n_heads=8,
                 n_atom_types=MAX_ATOMIC_NUM,
                 n_super_feature=4 + 2 + 4 + MAX_ATOMIC_NUM*2,
                 dropout_ratio=0.5,
                 concat_hidden=False,
                 weight_tying=True, activation=functions.identity,
                 num_edge_type=4):
        super(GGNN_GWM, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        n_message_layer = 1 if weight_tying else n_layers
        with self.init_scope():
            # Update
            self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)
            self.update_layers = chainer.ChainList(*[GGNNUpdate(
                hidden_dim=hidden_dim, num_edge_type=num_edge_type)
                for _ in range(n_message_layer)])
            # GWM
            self.embed_super = links.Linear(in_size=n_super_feature, out_size=hidden_dim_super)
            self.gwm = GWM(hidden_dim=hidden_dim, hidden_dim_super=hidden_dim_super,
                 n_layers=n_message_layer, n_heads=n_heads,
                 dropout_ratio=dropout_ratio,
                 tying_flag=weight_tying,
                 scaler_mgr_flag=False,
                 gpu=-1)            
            # Readout
            self.readout_layers = chainer.ChainList(*[GGNNReadout(
                out_dim=out_dim, hidden_dim=hidden_dim,
                activation=activation, activation_agg=activation)
                for _ in range(n_readout_layer)])
            self.linear_for_concat_super = links.Linear(in_size=None, out_size=out_dim)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_super = hidden_dim_super
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.num_edge_type = num_edge_type
        self.activation = activation
        self.concat_hidden = concat_hidden
        self.weight_tying = weight_tying

    def __call__(self, atom_array, adj, super_node, is_real_node=None):
        """Forward propagation

        Args:
            atom_array (numpy.ndarray): minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj (numpy.ndarray): minibatch of adjancency matrix with edge-type
                information
            super_node (numpy.ndarray): 1D rray, the super-node observation.
            is_real_node (numpy.ndarray): 2-dim array (minibatch, num_nodes).
                1 for real node, 0 for virtual node.
                If `None`, all node is considered as real node.

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        # reset state
        self.reset_state()
        if atom_array.dtype == self.xp.int32:
            h = self.embed(atom_array)  # (minibatch, max_num_atoms)
        else:
            h = atom_array
        h0 = functions.copy(h, cuda.get_device_from_array(h.data).id)

        self.gwm.GRU_local.reset_state()
        self.gwm.GRU_super.reset_state()

        # ebmbed super node
        h_s = self.embed_super(super_node)

        g_list = []
        for step in range(self.n_layers):
            message_layer_index = 0 if self.weight_tying else step
            h2 = self.update_layers[message_layer_index](h, adj)

            h, h_s = self.gwm(h, h2, h_s, message_layer_index)
            if self.concat_hidden:
                g = self.readout_layers[step](h, h0, is_real_node)
                g_list.append(g)

        if self.concat_hidden:
            return functions.concat(g_list, axis=1)
        else:
            g = self.readout_layers[0](h, h0, is_real_node)
            g2 = functions.concat( (g, h_s), axis=1 )
            out_g = functions.relu(self.linear_for_concat_super(g2))
            return out_g

    def reset_state(self):
        [update_layer.reset_state() for update_layer in self.update_layers]
