import numpy

from chainer.dataset import convert
from sklearn import metrics

from chainer_chemistry.training.extensions.batch_evaluator import BatchEvaluator  # NOQA


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


class PRCAUCEvaluator(BatchEvaluator):

    """Evaluator which calculates PRC AUC score

    Note that this Evaluator is only applicable to binary classification task.

    Args:
        iterator: Dataset iterator for the dataset to calculate PRC AUC score.
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
        raise_value_error (bool): If `False`, `ValueError` caused by
            `roc_auc_score` calculation is suppressed and ignored with a
            warning message.
        logger:

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
                 pos_labels=1, ignore_labels=None, raise_value_error=True,
                 logger=None):
        metrics_fun = {'prc_auc': self.prc_auc_score}
        super(PRCAUCEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func, metrics_fun=metrics_fun,
            name=name, logger=logger)

        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)
        self.raise_value_error = raise_value_error

    def prc_auc_score(self, y_total, t_total):
        # --- ignore labels if specified ---
        if self.ignore_labels:
            valid_ind = numpy.in1d(t_total, self.ignore_labels, invert=True)
            y_total = y_total[valid_ind]
            t_total = t_total[valid_ind]

        # --- set positive labels to 1, negative labels to 0 ---
        pos_indices = numpy.in1d(t_total, self.pos_labels)
        t_total = numpy.where(pos_indices, 1, 0)

        if len(numpy.unique(t_total)) != 2:
            if self.raise_value_error:
                raise ValueError("Only one class present in y_true. PRC AUC "
                                 "score is not defined in that case.")
            else:
                return numpy.nan

        precision, recall, _ = metrics.precision_recall_curve(t_total, y_total)
        prc_auc = metrics.auc(recall, precision)
        return prc_auc
