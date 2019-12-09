from typing import Optional  # NOQA

import chainer
import numpy  # NOQA


class GraphConvPredictor(chainer.Chain):
    """Wrapper class that combines a graph convolution and MLP."""

    def __init__(
            self,
            graph_conv,  # type: chainer.Link
            mlp=None,  # type: Optional[chainer.Link]
            label_scaler=None,  # type: Optional[chainer.Link]
            postprocess_fn=None  # type: Optional[chainer.FunctionNode]
    ):
        # type: (...) -> None
        """Initialize the graph convolution predictor.

        Args:
            graph_conv (chainer.Chain): The graph convolution network
                required to obtain molecule feature representation.
            mlp (chainer.Chain or None): Multi layer perceptron;
                used as the final fully connected layer. Set it to
                `None` if no operation is necessary after the
                `graph_conv` calculation.
            label_scaler (chainer.Link or None): scaler link
            postprocess_fn (chainer.FunctionNode or None):
                postprocess function for prediction.
        """
        super(GraphConvPredictor, self).__init__()
        with self.init_scope():
            self.graph_conv = graph_conv
            if isinstance(mlp, chainer.Link):
                self.mlp = mlp
            if isinstance(label_scaler, chainer.Link):
                self.label_scaler = label_scaler
        if not isinstance(mlp, chainer.Link):
            self.mlp = mlp
        if not isinstance(label_scaler, chainer.Link):
            self.label_scaler = label_scaler
        self.postprocess_fn = postprocess_fn or chainer.functions.identity

    def __call__(self, *args, **kwargs):
        x = self.graph_conv(*args, **kwargs)
        if self.mlp:
            x = self.mlp(x)
        if self.label_scaler is not None:
            x = self.label_scaler.inverse_transform(x)
        return x

    def predict(self, atoms, adjs):
        # type: (numpy.ndarray, numpy.ndarray) -> chainer.Variable
        # TODO(nakago): support super_node & is_real_node args.
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            x = self.__call__(atoms, adjs)
            return self.postprocess_fn(x)
