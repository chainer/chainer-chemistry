import chainer
from chainer import cuda
from chainer import functions as F
from chainer import iterators as I
import numpy

from chainerchem.dataset.converters import concat_mols
from chainerchem.models import GGNN
from chainerchem.models import MLP
from chainerchem.models import NFP
from chainerchem.models import SchNet


def build_predictor(method, n_unit, conv_layers, class_num):
    if method == 'nfp':
        print('Use NFP predictor...')
        predictorl = GraphConvPredictor(NFP(n_unit, n_unit, conv_layers),
                                        MLP(n_unit, class_num))
    elif method == 'ggnn':
        print('Use GGNN predictor...')
        predictorl = GraphConvPredictor(GGNN(n_unit, n_unit, conv_layers),
                                        MLP(n_unit, class_num))
    elif method == 'schnet':
        print('Use SchNet predictor...')
        predictorl = SchNet(n_unit, class_num, conv_layers, n_unit)
    else:
        print('[ERROR] Invalid predictorl: method={}'.format(method))
        raise ValueError
    return predictorl


class GraphConvPredictor(chainer.Chain):
    """Wrapper class that combines a graph convolution and MLP."""

    def __init__(self, graph_conv, mlp):
        """Constructor

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

    def predict(self, atoms, adjs):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = self.__call__(atoms, adjs)
            return F.sigmoid(x)


class InferenceLoop(object):
    """Wrapper for predictors that offers inference loop."""

    def __init__(self, predictor):
        self.predictor = predictor

    def customized_inference(self, iterator, converter, device):
        """Predict with given predictor to given dataset

        This function iteratively gets a set of sample from ``iterator``,
        convert it to a minibatch with ``converter``, move it to
        a device specified by ``device`` ID, and feed it to ``predictor``
        to get predictions. All predictions are concatenaed along the first
        axis and then returned as a single :class:`numpy.ndarray`.

        Note that this function does not assume a tuple of minibatches
        that converter outputs does not have a minibatch of labels,
        and directly feeds the tuple to the predictor,
        whereas :class:`chainer.links.Classifier` extracts all but the last
        minibatch as feature vectors and treat the last one as labels

        Args:
            predictor: A predictor.
            iterator: An iterator that runs over the dataset.
            converter: A converter for creating minibatches.
            device: A device to which minibatches are transferred.

        Returns:
            numpy.ndarray: Prediction results.

        """
        iterator.reset()
        ret = []
        for batch in iterator:
            x = converter(batch, device=device)
            y = self.predictor.predict(*x)
            ret.append(cuda.to_cpu(y.data))
        return numpy.concatenate(ret, axis=0)

    def inference(self, X):
        """Predict with given predictor to given dataset

        We simplify the API of this method for easy-use.
        This fixes a size of minibatch size, a converter for creating
        minibatches, and a device to which minibatches are transferred.

        For customized prediction, use ``GraphConvPredictor.predict_``
        method instead.

        Args:
            X: test dataset

        Returns:
            numpy.ndarray: Prediction results

        """

        batchsize = 128
        data_iter = I.SerialIterator(X, batchsize, repeat=False, shuffle=False)

        def converter(batch, device):
            return concat_mols(batch, device)[:-1]

        return self.customized_inference(data_iter,
                                         converter=converter,
                                         device=-1)
