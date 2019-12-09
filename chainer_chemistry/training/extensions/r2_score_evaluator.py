from chainer.backends import cuda
from chainer.dataset import convert

from chainer_chemistry.training.extensions.batch_evaluator import BatchEvaluator  # NOQA


class R2ScoreEvaluator(BatchEvaluator):

    """Evaluator with calculates R^2 (coefficient of determination)

    regression score.

    Args:
        iterator: Dataset iterator for the dataset to calculate
            R^2(coefficient of determination) regression score.
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
        sample_weight: This argument is for compatibility with
            scikit-learn's implementation of r2_score. Current
            implementation admits None only.
        multioutput (str): If 'uniform_average', this function returns an
            average of R^2 score of multiple output. If 'raw_average', this
            function return a set of R^2 score of multiple output.

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
                 pos_label=1, ignore_labels=None, raise_value_error=True,
                 logger=None, sample_weight=None,
                 multioutput='uniform_average', ignore_nan=False):
        metrics_fun = {'r2_score': self.r2_score}
        super(R2ScoreEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func, metrics_fun=metrics_fun,
            name=name, logger=logger)

        self.pos_label = pos_label
        self.ignore_labels = ignore_labels
        self.raise_value_error = raise_value_error
        self.sample_weight = sample_weight
        self.multioutput = multioutput
        self.ignore_nan = ignore_nan

    def r2_score(self, pred, true, sample_weight=None,
                 multioutput='uniform_average', ignore_nan=False):

        if self.sample_weight is not None:
            raise NotImplementedError()
        if self.multioutput not in ['uniform_average', 'raw_values']:
            raise ValueError('invalid multioutput argument')

        xp = cuda.get_array_module(pred)
        diff = pred - true
        dev = true - xp.mean(true, axis=0)
        if self.ignore_nan:
            diff[xp.isnan(diff)] = 0.
            dev[xp.isnan(dev)] = 0.
        SS_res = xp.asarray(xp.sum(diff ** 2, axis=0))
        SS_tot = xp.asarray(xp.sum(dev ** 2, axis=0))
        SS_tot_iszero = SS_tot == 0
        SS_tot[SS_tot_iszero] = 1  # Assign dummy value to avoid zero-division
        ret = xp.where(
            SS_tot_iszero, 0.0, 1 - SS_res / SS_tot).astype(pred.dtype)
        if self.multioutput == 'uniform_average':
            return xp.asarray(ret.mean())
        elif self.multioutput == 'raw_values':
            return ret
