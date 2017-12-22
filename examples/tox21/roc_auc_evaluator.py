import numpy
import six

import chainer
from chainer import cuda, reporter
from chainer.dataset import convert
from chainer.training.extensions import Evaluator
from sklearn import metrics


def get_1d_numpy_array(v):
    if isinstance(v, chainer.Variable):
        v = v.data
    return cuda.to_cpu(v).ravel()

import chainerex.utils as cl


class ROCAUCEvaluator(Evaluator):

    def __init__(self, iterator, target, predictor=None,
                 converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, name=None):
        super(ROCAUCEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        if name is not None:
            self.name = name or 'validation'
        self.predictor = predictor

    def evaluate(self):
        tm = cl.TimeMeasure.get_instance()
        tm.start()
        iterator = self._iterators['main']
        predictor = self.predictor or self._targets['main']
        xp = predictor.xp

        iterator.reset()

        y_total = []
        t_total = []
        for batch in iterator:
            in_arrays = self.converter(batch, self.device)
            with chainer.no_backprop_mode(), chainer.using_config('train',
                                                                  False):
                y = predictor(*in_arrays[:-1])
            t = in_arrays[-1]
            y_data = get_1d_numpy_array(y)
            t_data = get_1d_numpy_array(t)
            y_data = y_data[t_data != -1]
            t_data = t_data[t_data != -1]
            y_total.append(y_data)
            t_total.append(t_data)

        t_total = numpy.concatenate(t_total).ravel()
        y_total = numpy.concatenate(y_total).ravel()
        roc_auc = metrics.roc_auc_score(t_total, y_total)
        # tm.update('eval roc_auc')
        # reporter.report({'roc_auc': roc_auc}, self._targets['main'])
        return {'{}/roc_auc'.format(self.name): roc_auc}
