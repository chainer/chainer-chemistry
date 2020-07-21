import numpy

import chainer

from chainer_chemistry.links.scaler.base import BaseScaler, to_array  # NOQA


def _sigmoid_derivative(x):
    h = chainer.functions.sigmoid(x)
    return chainer.grad([h], [x], enable_double_backprop=True)[0]


def format_x(x):
    """x may be array or Variable"""
    # currently, only consider the case x is 2-dim, (batchsize, feature)
    if x.ndim == 1:
        # Deal with as 1 feature with several samples.
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(
            "Unexpected value x.shape={}, only x.ndim=2 is supported."
            .format(x.shape))
    return x


class FlowScaler(BaseScaler):
    """Flow Scaler.

    Flow Scaler is a Scaler that scale data into the normal distribution.
    This scaler uses a technique named "flow". By using this technique,
    parametrized bijective function is learned to scale data that distributes
    arbitrary continuous distribution into specified continuous distribution.
    In this scaler, multi-layer perceptron whose weight is restricted into
    positive range is used as parametrized bijective function.

    Args:
        hidden_num(int): number of units in hidden layer of multi-layer
            perceptron.
    """

    def __init__(self, hidden_num=20):
        super(FlowScaler, self).__init__()

        self.hidden_num = hidden_num

        self.mean = None
        self.register_persistent('mean')
        self.std = None
        self.register_persistent('std')
        self.eps = numpy.float32(1e-6)
        W_initializer = chainer.initializers.Normal(0.1)
        with self.init_scope():
            self.W1_ = chainer.Parameter(W_initializer)
            self.b1 = chainer.Parameter(0)
            self.W2_ = chainer.Parameter(W_initializer)
            self.b2 = chainer.Parameter(0)

    def _initialize_params(self, in_size):
        self.W1_.initialize((self.hidden_num, in_size, 1, 1, 1, 1))
        self.b1.initialize((self.hidden_num, in_size, 1))
        self.W2_.initialize((1, in_size, 1, self.hidden_num, 1, 1))
        self.b2.initialize((1, in_size, 1))

    @property
    def W1(self):
        return chainer.functions.softplus(self.W1_)

    @property
    def W2(self):
        return chainer.functions.softplus(self.W2_)

    def _forward(self, x):
        x = chainer.functions.expand_dims(x, axis=1)
        x = chainer.functions.expand_dims(x, axis=3)
        h = chainer.functions.local_convolution_2d(x, self.W1, self.b1)
        h = chainer.functions.sigmoid(h)
        h = chainer.functions.local_convolution_2d(h, self.W2, self.b2)
        h = h[:, 0, :, 0]
        return h

    def _derivative(self, x):
        x = chainer.functions.expand_dims(x, axis=1)
        x = chainer.functions.expand_dims(x, axis=3)
        h = chainer.functions.local_convolution_2d(x, self.W1, self.b1)
        h = _sigmoid_derivative(h)
        h = h * chainer.functions.expand_dims(self.W1[:, :, 0, 0, 0], axis=0)
        h = chainer.functions.local_convolution_2d(h, self.W2)
        h = h[:, 0, :, 0]
        return h

    def _loss(self, x):
        # loss = -log(p(f(x))) - log|f'(x)|
        x_nan = self.xp.isnan(x)
        x_not_nan = self.xp.logical_not(x_nan)
        x = self.xp.nan_to_num(x)
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x.astype(numpy.float32))
        y = self._forward(x)
        gy = self._derivative(x)
        # gy, = chainer.grad([y], [x], enable_double_backprop=True)
        std_gaussian = chainer.distributions.Normal(
            self.xp.zeros(shape=x.shape, dtype=numpy.float32),
            self.xp.ones(shape=x.shape, dtype=numpy.float32))
        loss = -std_gaussian.log_prob(y)
        loss -= chainer.functions.log(abs(gy) + self.eps)
        loss = chainer.functions.sum(loss[x_not_nan]) / x_not_nan.sum()
        chainer.reporter.report({'loss': loss}, self)
        return loss

    def fit(self, x, batch_size=100, iteration=3000):
        """Fitting parameter.

        Args:
            x(:class:`~chainer.Variable` or :ref:`ndarray`): data for learning.
            batch_size(int): size of batch used for learning multi-layer
                perceptron.
            iteration(int): number of iteration.

        Returns:
            self (FlowScaler): this instance.
        """
        if isinstance(x, chainer.Variable):
            x = x.array

        x = format_x(x)

        self._initialize_params(x.shape[1])

        xp = self.xp
        if xp is numpy:
            self.mean = xp.nanmean(x, axis=0)
            self.std = xp.nanstd(x, axis=0)
        else:
            if int(xp.sum(xp.isnan(x))) > 0:
                raise NotImplementedError(
                    "FlowScaling with nan value on GPU is not supported.")
            # cupy.nanmean, cupy.nanstd is not implemented yet.
            self.mean = xp.mean(x, axis=0)
            self.std = xp.std(x, axis=0)

        x = (x - self.mean) / (self.std + self.eps)

        optimizer = chainer.optimizers.Adam(0.3)
        optimizer.setup(self)

        train = chainer.datasets.TupleDataset(x)
        train_iter = chainer.iterators.SerialIterator(train, batch_size)

        updater = chainer.training.updaters.StandardUpdater(
            train_iter, optimizer, loss_func=self._loss)

        trainer = chainer.training.Trainer(
            updater, (iteration, 'iteration'))
        trainer.extend(chainer.training.extensions.LogReport(
            trigger=(100, 'iteration')))
        trainer.extend(chainer.training.extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'elapsed_time']))

        trainer.run()

        return self

    def transform(self, x, batch_size=100):
        """Transform.

        Args:
            x(:class:`~chainer.Variable` or :ref:`ndarray`): data.
            batch_size(int): size of batch used for learning multi-layer
                perceptron.

        Returns:
            scaled_x(:class:`~chainer.Variable` or :ref:`ndarray`):
                transformed data.
        """
        if self.mean is None:
            raise AttributeError('[Error] mean is None, call fit beforehand!')

        x_ = format_x(x)
        x_ = (x_ - self.mean) / (self.std + self.eps)

        y = []
        for i in range((len(x) - 1) // batch_size + 1):
            y.append(self._forward(
                x_[i*batch_size: (i+1)*batch_size]))

        y = chainer.functions.concat(y, axis=0)

        if x.ndim == 1:
            y = y[:, 0]

        if isinstance(x_, chainer.Variable):
            return y
        else:
            return y.data
