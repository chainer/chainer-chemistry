import copy

import numpy

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
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


def _to_list(a):
    """convert value `a` to list

    Args:
        a: value to be convert to `list`

    Returns (list):

    """
    if isinstance(a, (int, float)):
        return [a, ]
    else:
        # expected to be list or some iterable class
        return a


class ROCAUCEvaluator(Evaluator):

    """Evaluator which calculates ROC AUC score

    Note that this Evaluator is only applicable to binary classification task.

    Args:
        iterator: Dataset iterator for the dataset to calculate ROC AUC score.
            It can also be a dictionary of iterators. If this is just an
            iterator, the iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays and true label.
            :func:`~chainer.dataset.concat_examples` is used by default.
            It is expected to return input arrays of the form
            `[x_0, ..., x_n, t]`, where `x_0, ..., x_n` are the inputs to
            the evaluation function and `t` is the true label.
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
        pos_labels (int or list): labels of the positive class, other classes
            are considered as negative.
        ignore_labels (int or list or None): labels to be ignored.
            `None` is used to not ignore all labels.

    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
        pos_labels (list): labels of the positive class
        ignore_labels (list): labels to be ignored.

    """

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, name=None,
                 pos_labels=1, ignore_labels=None):
        super(ROCAUCEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.name = name
        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)

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

        # --- ignore labels if specified ---
        if self.ignore_labels:
            valid_ind = numpy.in1d(t_total, self.ignore_labels, invert=True)
            y_total = y_total[valid_ind]
            t_total = t_total[valid_ind]

        # --- set positive labels to 1, negative labels to 0 ---
        pos_indices = numpy.in1d(t_total, self.pos_labels)
        t_total = numpy.where(pos_indices, 1, 0)
        roc_auc = metrics.roc_auc_score(t_total, y_total)

        observation = {}
        with reporter.report_scope(observation):
            reporter.report({'roc_auc': roc_auc}, self._targets['main'])
        return observation
