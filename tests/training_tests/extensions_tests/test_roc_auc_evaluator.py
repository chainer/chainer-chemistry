"""
ROCAUCEvaluator uses `sklearn.metrics.roc_auc_score` internally.
Refer: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.\
roc_auc_score.html
"""
import numpy
import pytest

import chainer
from chainer.iterators import SerialIterator

from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset  # NOQA
from chainer_chemistry.training.extensions.roc_auc_evaluator import ROCAUCEvaluator  # NOQA


@pytest.fixture
def data0():
    # `t` is correct label, `y` is dummy predict value by predictor
    t = numpy.array([0, 0, 1, 1], dtype=numpy.int32)[:, None]
    y = numpy.array([0.1, 0.4, 0.35, 0.8], dtype=numpy.float32)[:, None]
    return y, t


@pytest.fixture
def data1():
    # `t` is correct label, `y` is dummy predict value by predictor
    t = numpy.array([0, 1, -1, 0, 2, -1], dtype=numpy.int32)[:, None]
    y = numpy.array([0.1, 0.35, 0.2, 0.4, 0.8, 0.35],
                    dtype=numpy.float32)[:, None]
    return y, t


class DummyPredictor(chainer.Chain):

    def __call__(self, y):
        # it receives `y` and return `y` directly
        return y


def test_roc_auc_evaluator(data0, data1):
    _test_roc_auc_evaluator_default_args(data0)
    _test_roc_auc_evaluator_with_labels(data1)


def _test_roc_auc_evaluator_default_args(data0):

    predictor = DummyPredictor()
    dataset = NumpyTupleDataset(*data0)

    iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)
    evaluator = ROCAUCEvaluator(
        iterator, predictor, name='train',
        pos_labels=1, ignore_labels=None
    )
    repo = chainer.Reporter()
    repo.add_observer('target', predictor)
    with repo:
        observation = evaluator.evaluate()

    expected_roc_auc = 0.75
    # print('observation ', observation)
    assert observation['target/roc_auc'] == expected_roc_auc

    # --- test __call__ ---
    result = evaluator()
    # print('result ', result)
    assert result['train/main/roc_auc'] == expected_roc_auc


def _test_roc_auc_evaluator_with_labels(data1):
    """test `pos_labels` and `ignore_labels` behavior"""

    predictor = DummyPredictor()
    dataset = NumpyTupleDataset(*data1)

    iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)
    evaluator = ROCAUCEvaluator(
        iterator, predictor, name='val',
        pos_labels=[1, 2], ignore_labels=-1,
    )

    # --- test evaluate ---
    repo = chainer.Reporter()
    repo.add_observer('target', predictor)
    with repo:
        observation = evaluator.evaluate()

    expected_roc_auc = 0.75
    # print('observation ', observation)
    assert observation['target/roc_auc'] == expected_roc_auc

    # --- test __call__ ---
    result = evaluator()
    # print('result ', result)
    assert result['val/main/roc_auc'] == expected_roc_auc


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
