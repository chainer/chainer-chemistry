import copy
from logging import getLogger

import numpy

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer.training.extensions import Evaluator


def _get_1d_numpy_array(v):
    """Convert array or Variable to 1d numpy array

    Args:
        v (numpy.ndarray or cupy.ndarray or chainer.Variable): array to be
            converted to 1d numpy array

    Returns (numpy.ndarray): Raveled 1d numpy array

    """
    if isinstance(v, chainer.Variable):
        v = v.data
    return cuda.to_cpu(v).ravel()


class BatchEvaluator(Evaluator):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, metrics_fun=None,
                 name=None, logger=None):
        super(BatchEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.name = name
        self.logger = logger or getLogger()

        if callable(metrics_fun):
            # TODO(mottodora): use better name or infer
            self.metrics_fun = {"evaluation": metrics_fun}
        elif isinstance(metrics_fun, dict):
            self.metrics_fun = metrics_fun
        else:
            raise TypeError('Unexpected type metrics_fun must be Callable or '
                            'dict.')

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = []
        t_total = []
        for batch in it:
            in_arrays = self.converter(batch, self.device)
            with chainer.no_backprop_mode(), chainer.using_config('train',
                                                                  False):
                y = eval_func(*in_arrays[:-1])
            t = in_arrays[-1]
            y_data = _get_1d_numpy_array(y)
            t_data = _get_1d_numpy_array(t)
            y_total.append(y_data)
            t_total.append(t_data)

        y_total = numpy.concatenate(y_total).ravel()
        t_total = numpy.concatenate(t_total).ravel()
        # metrics_value = self.metrics_fun(y_total, t_total)
        metrics = {key: metric_fun(y_total, t_total) for key, metric_fun in
                   self.metrics_fun.items()}

        observation = {}
        with reporter.report_scope(observation):
            reporter.report(metrics, self._targets['main'])
        return observation
