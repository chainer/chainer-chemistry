import chainer
from chainer import functions
from chainer import links

from chainer_chemistry.links.connection.graph_linear import GraphLinear


class GNNFiLMUpdate(chainer.Chain):
    """GNNFiLM submodule for update part.

    Args:
        hidden_dim (int): dimension of feature vector associated to
            each atom
        num_edge_type (int): number of types of edge
    """

    
    def __init__(self, hidden_dim=16, num_edge_type=5, activation=functions.relu):
        super(GNNFiLMUpdate, self).__init__()
        self.num_edge_type = num_edge_type
        self.activation = activation
        with self.init_scope():
            self.W_linear = GraphLinear(in_size=None, out_size = self.num_edge_type * hidden_dim, nobias=True) # W_l in eq. (6)
            self.W_g = GraphLinear(in_size=None, out_size = self.num_edge_type * hidden_dim * 2, nobias=True) # g in eq. (6)
            self.norm_layer = links.LayerNormalization() # l in eq. (6)
        
            
    def forward(self, h, adj):
        # --- Message part ---
        
        xp = self.xp
        mb, atom, ch = h.shape
        adj = xp.broadcast_to(adj[:,:,:,:, xp.newaxis], (*adj.shape, ch))
        messages = functions.reshape(self.W_linear(h), (mb, atom, ch, self.num_edge_type))
        messages = functions.transpose(messages, (3, 0, 1, 2))
        film_weights = functions.reshape(self.W_g(h),
            (mb, atom, 2 * ch, self.num_edge_type))
        film_weights = functions.transpose(film_weights, (3, 0, 1, 2))
        gamma = film_weights[:,:,:, :ch] # (num_edge_type, minibatch, atom, out_ch)
        beta = film_weights[:,:,:, ch:] # (num_edge_type, minibatch, atom, out_ch)
        
        # --- Update part ---

        messages = functions.expand_dims(gamma,axis=3) * functions.expand_dims(messages,axis=2) + functions.expand_dims(beta,axis=3)
        messages = self.activation(messages)
        messages = functions.transpose(messages, (1, 0, 2, 3, 4)) # (minibatch, num_edge_type, atom, atom, out_ch)
        messages = adj * messages
        messages = functions.sum(messages, axis=3) # sum across atoms
        messages = functions.sum(messages, axis=1) # sum across num_edge_type
        messages = functions.reshape(messages, (mb * atom, ch))
        messages = self.norm_layer(messages)
        messages = functions.reshape(messages, (mb, atom, ch))
        return messages