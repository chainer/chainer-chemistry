import chainer

from chainer_chemistry.models import MLP, NFP, GGNN, SchNet, WeaveNet


class GraphConvPredictor(chainer.Chain):

    def __init__(self, graph_conv, mlp):
        """

        Args:
            graph_conv: graph convolution network to obtain molecule feature 
                        representation
            mlp: multi layer perceptron, used as final connected layer
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            self.mlp = mlp

    def __call__(self, atoms, adjs):
        x = self.graph_conv(atoms, adjs)
        x = self.mlp(x)
        return x

        # def _predict(self, atoms, adjs):
        #     with chainer.no_backprop_mode(), chainer.using_config('train', False):
        #         x = self.__call__(atoms, adjs)
        #         return F.sigmoid(x)
        #
        # def predict(self, *args, batchsize=32, device=-1):
        #     if device >= 0:
        #         chainer.cuda.get_device_from_id(device).use()
        #         self.to_gpu()  # Copy the model to the GPU
        #
        #     # TODO: Not test yet, check behavior
        #     data = args[0]
        #     y_list = []
        #     for i in range(0, len(data), batchsize):
        #         atoms, adjs = concat_mols(data[i:i + batchsize], device=device)[:2]
        #         y = self._predict(atoms, adjs)
        #         y_list.append(cuda.to_cpu(y.data))
        #     y_array = numpy.concatenate(y_list, axis=0)
        #     return y_array


def model_constructor(method, class_num, n_unit, conv_layers):
    if method == 'nfp':
        print('Train NFP model...')
        model = GraphConvPredictor(NFP(out_dim=n_unit, hidden_dim=n_unit,
                                       n_layers=conv_layers),
                                   MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'ggnn':
        print('Train GGNN model...')
        model = GraphConvPredictor(GGNN(out_dim=n_unit, hidden_dim=n_unit,
                                        n_layers=conv_layers),
                                   MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'schnet':
        print('Train SchNet model...')
        model = SchNet(out_dim=class_num)
    elif method == 'weavenet':
        print('Train WeaveNet model...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers
        model = GraphConvPredictor(
            WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                     n_sub_layer=n_sub_layer, n_atom=n_atom),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    else:
        raise ValueError('[ERROR] Invalid method {}'.format(method))
    return model
