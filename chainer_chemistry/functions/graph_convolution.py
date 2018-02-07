import chainer
from chainer import functions
from chainer import function_node
from chainer import utils

class GraphConvolution(function_node.FunctionNode):

    def __init__(self, adj):
        if isinstance(adj, chainer.Variable):
            adj = adj.data
        self.adj = adj
        
    def forward(self, inputs):
        self.retain_inputs((0, 1))
        x, w = inputs
        if self.adj is None:
            h = x
        else:
            h = functions.math.matmul._matmul(self.adj, x, False, False, False)
        self.h = h
        h_axes = ((0, 1), (2,))
        w_axes = ((0,), (1,))
        y_axes = ((0, 1), (2,))
        y = functions.math.tensordot._tensordot(h, w,
                                                h_axes, w_axes, y_axes)
        return y,

    def backward(self, indexes, grad_outputs):
        x, w = self.get_retained_inputs()
        gy, = grad_outputs
        x_data = x.data
        w_data = w.data
        gy_data = gy.data
        h_axes = ((0, 1), (2,))
        w_axes = ((0,), (1,))
        y_axes = ((0, 1), (2,))
        gw_data = functions.math.tensordot._tensordot(self.h, gy_data,
                                            (h_axes[1], h_axes[0]), y_axes, w_axes)
        self.h = None
        gh_data = functions.math.tensordot._tensordot(gy_data, w_data,
                                            y_axes, (w_axes[1], w_axes[0]), h_axes)
        if self.adj is None:
            gx_data = gh_data
        else:
            gx_data = functions.math.matmul._matmul(self.adj, gh_data, True, False, False)
            self.adj = None
        gx = chainer.Variable(gx_data)
        gw = chainer.Variable(gw_data)
        return gx, gw

def graph_convolution(x, adj, w):
    #
    # adj: (B,  V,  V)
    # x:   (B,  V, CI)
    # w:      (CI, CO)
    #
    # h = functions.matmul(adj, x)
    # y = functions.tensordot(h, w, axes=1)
    y = GraphConvolution(adj).apply((x, w))[0]
    return y
