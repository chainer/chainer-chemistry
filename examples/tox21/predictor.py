import chainer
from chainer import cuda
from chainer import functions as F
from chainer import iterators as I
import numpy as np

from chainer_chemistry.dataset.converters import concat_mols
from chainer_chemistry.models import GGNN
from chainer_chemistry.models import MLP
from chainer_chemistry.models import NFP
from chainer_chemistry.models import SchNet
from chainer_chemistry.models import WeaveNet


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
        predictor = SchNet(out_dim=class_num, hidden_dim=n_unit,
                           n_layers=conv_layers, readout_hidden_dim=n_unit)
    elif method == 'weavenet':
        print('Use WeaveNet predictor...')
        n_atom = 20
        n_sub_layer = 1
        weave_channels = [50] * conv_layers
        predictor = GraphConvPredictor(
            WeaveNet(weave_channels=weave_channels, hidden_dim=n_unit,
                     n_sub_layer=n_sub_layer, n_atom=n_atom),
            MLP(out_dim=class_num, hidden_dim=n_unit))
    else:
        raise ValueError('[ERROR] Invalid predictor: method={}'.format(method))
    return predictor


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
        return np.concatenate(ret, axis=0)

    def inference(self, X):
        """Predict with given predictor to given dataset

        We simplify the API of this method for easy-use and fixes
        several configurations. Specifically, we fix a size of
        minibatch size, a converter for creating minibatches.
        Also, if the predictor ``InferenceLoop`` holds is located in
        host memory (judged by the ``xp`` attribute), all computations are
        done in CPU. Otherwise, i.e. it is in GPU memory,
        minibatches are transferred to the
        `current device <https://docs-cupy.chainer.org/en/\
        stable/tutorial/basic.html?device#current-device>`_
        in the sense of CuPy. For customized prediction, use
        ``GraphConvPredictor.customized_inference`` method instead.

        Args:
            X: test dataset

        Returns:
            numpy.ndarray: Prediction results

        """

        batchsize = 128
        data_iter = I.SerialIterator(X, batchsize, repeat=False, shuffle=False)

        if self.predictor.xp is np:
            device_id = -1
        else:
            device_id = cuda.cupy.cuda.get_device_id()

        def converter(batch, device):
            return concat_mols(batch, device)[:-1]

        return self.customized_inference(data_iter,
                                         converter=converter,
                                         device=device_id)
