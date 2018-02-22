import numpy

import chainer
from chainer import cuda
from chainer import reporter
from chainer.dataset import convert
from chainer.training.extensions import Evaluator
from sklearn import metrics


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


class ROCAUCEvaluator(Evaluator):

    """Evaluator which calculates ROC AUC score

    Args:
        iterator: Dataset iterator for the dataset to calculate ROC AUC score.
            It can also be a dictionary of iterators. If this is just an 
            iterator, the iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays.
            :func:`~chainer.dataset.concat_examples` is used by default.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        name (str): name of this extension. When `name` is None, 
            `default_name='validation'` which is defined in super class 
            `Evaluator` is used as extension name. This name affects to the
            reported key name.

    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.

    """

    def __init__(self, iterator, target, predictor=None,
                 converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, name=None):
        super(ROCAUCEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.name = name
        self.predictor = predictor

    def evaluate(self):
        iterator = self._iterators['main']
        predictor = self.predictor or self._targets['main']

        iterator.reset()

        y_total = []
        t_total = []
        for batch in iterator:
            in_arrays = self.converter(batch, self.device)
            with chainer.no_backprop_mode(), chainer.using_config('train',
                                                                  False):
                y = predictor(*in_arrays[:-1])
            t = in_arrays[-1]
            y_data = _get_1d_numpy_array(y)
            t_data = _get_1d_numpy_array(t)
            y_data = y_data[t_data != -1]
            t_data = t_data[t_data != -1]
            y_total.append(y_data)
            t_total.append(t_data)

        t_total = numpy.concatenate(t_total).ravel()
        y_total = numpy.concatenate(y_total).ravel()
        roc_auc = metrics.roc_auc_score(t_total, y_total)

        observation = {}
        with reporter.report_scope(observation):
            reporter.report({'roc_auc': roc_auc}, self._targets['main'])
        return observation
