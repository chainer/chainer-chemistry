import chainer

from chainer import initializers
from chainer import link
from chainer import variable

from chainer_chemistry import functions

class GraphConvolution(link.Link):

    def __init__(self, in_size, out_size, initialW=None):
        super(GraphConvolution, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        
        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            self.W.initialize((self.in_size, self.out_size))

    def __call__(self, x, adj=None):
        return functions.graph_convolution(x, adj, self.W)
