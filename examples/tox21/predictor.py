import chainer
from chainer import functions as F

from chainer_chemistry.models import GGNN
from chainer_chemistry.models import MLP
from chainer_chemistry.models import NFP
from chainer_chemistry.models import RSGCN
from chainer_chemistry.models import SchNet
from chainer_chemistry.models import WeaveNet
from chainer_chemistry.models import RelGCN
from chainer_chemistry.models import RelGAT


def build_predictor(method, n_unit, conv_layers, class_num):
    if method == 'nfp':
        print('Use NFP predictor...')
        predictor = GraphConvPredictor(
            NFP(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'ggnn':
        print('Use GGNN predictor...')
        predictor = GraphConvPredictor(
            GGNN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'schnet':
        print('Use SchNet predictor...')
        # MLP layer is not necessary for SchNet
        predictor = GraphConvPredictor(
            SchNet(out_dim=class_num, hidden_dim=n_unit, n_layers=conv_layers,
                   readout_hidden_dim=n_unit), None)
    elif method == 'weavenet':
        print('Use WeaveNet predictor...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers
        predictor = GraphConvPredictor(
            WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                     n_sub_layer=n_sub_layer, n_atom=n_atom),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'rsgcn':
        print('Use RSGCN predictor...')
        predictor = GraphConvPredictor(
            RSGCN(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    elif method == 'relgcn':
        print('Use Relational GCN predictor...')
        num_edge_type = 4
        predictor = GraphConvPredictor(
            RelGCN(out_channels=class_num, num_edge_type=num_edge_type,
                   scale_adj=True))
    elif method == 'relgat':
        print('Use GAT predictor...')
        predictor = GraphConvPredictor(
            RelGAT(out_dim=n_unit, hidden_dim=n_unit, n_layers=conv_layers),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    else:
        raise ValueError('[ERROR] Invalid predictor: method={}'.format(method))
    return predictor


class GraphConvPredictor(chainer.Chain):
    """Wrapper class that combines a graph convolution and MLP."""

    def __init__(self, graph_conv, mlp=None):
        """Constructor

        Args:
            graph_conv: graph convolution network to obtain molecule feature
                        representation
            mlp: multi layer perceptron, used as final connected layer.
                It can be `None` if no operation is necessary after
                `graph_conv` calculation.
        """

        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp

    def __call__(self, atoms, adjs, is_real_node=None):
        if is_real_node is None:
            x = self.graph_conv(atoms, adjs)
        else:
            x = self.graph_conv(atoms, adjs, is_real_node)
        if self.mlp:
            x = self.mlp(x)
        return x

    def predict(self, atoms, adjs):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = self.__call__(atoms, adjs)
            return F.sigmoid(x)
