import numpy
import pytest

import chainer
from chainer import cuda
from chainer.iterators import SerialIterator

from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.training.extensions.r2_score_evaluator import R2ScoreEvaluator  # NOQA


@pytest.fixture
def inputs():
    numpy.random.seed(0)
    x0 = numpy.random.uniform(-1, 1, (4, 3)).astype('f')
    # Add sufficient margin to prevent computational error
    diff = numpy.random.uniform(-1, 1, (4, 3)).astype('f')
    diff[abs(diff) < 0.01] = 0.5
    x1 = x0 + diff
    x2 = numpy.asarray([[0.3, numpy.nan, 0.2],
                        [numpy.nan, 0.1, 0.5],
                        [0.9, 0.7, numpy.nan],
                        [0.2, -0.3, 0.4]]).astype('f')
    return x0, x1, x2


def r2_score(pred, true, sample_weight=None, multioutput="uniform_average",
             ignore_nan=False):
    diff = pred - true
    dev = true - numpy.mean(true, axis=0)
    if ignore_nan:
        diff[numpy.isnan(diff)] = 0.
        dev[numpy.isnan(dev)] = 0.
    SS_res = numpy.asarray(
        numpy.sum(diff ** 2, axis=0))
    SS_tot = numpy.asarray(
        numpy.sum(dev ** 2, axis=0))

    if multioutput == 'uniform_average':
        if numpy.any(SS_tot == 0):
            return 0.0
        else:
            return (1 - SS_res / SS_tot).mean()
    elif multioutput == 'raw_values':
        if numpy.any(SS_tot == 0):
            # Assign dummy value to avoid zero-division
            SS_tot_iszero = SS_tot == 0
            SS_tot[SS_tot_iszero] = 1

            return numpy.where(SS_tot_iszero, 0.0, 1 - SS_res / SS_tot)
        else:
            return 1 - SS_res / SS_tot


class DummyPredictor(chainer.Chain):

    def __call__(self, y):
        # it receives `y` and return `y` directly
        return y


def test_r2_score_evaluator(inputs):
    _test_r2_score_evaluator(inputs)
    _test_r2_score_evaluator_ignore_nan(inputs)
    _test_r2_score_evaluator_ignore_nan_with_nonnan_value(inputs)
    _test_r2_score_evaluator_raw_values(inputs)


@pytest.mark.gpu
def test_r2_score_evaluator_gpu(inputs):
    x0, x1, x2 = inputs
    _test_r2_score_evaluator((cuda.to_gpu(x0), cuda.to_gpu(x1), None))
    _test_r2_score_evaluator_ignore_nan(
        (cuda.to_gpu(x0), None, cuda.to_gpu(x2)))
    _test_r2_score_evaluator_ignore_nan_with_nonnan_value(
        (cuda.to_gpu(x0), cuda.to_gpu(x1), None))
    _test_r2_score_evaluator_raw_values(
        (cuda.to_gpu(x0), cuda.to_gpu(x1), None))


def _test_r2_score_evaluator(inputs):
    predictor = DummyPredictor()
    x0, x1, _ = inputs
    dataset = NumpyTupleDataset(x0, x1)

    iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)
    evaluator = R2ScoreEvaluator(iterator, predictor, name='train')
    repo = chainer.Reporter()
    repo.add_observer('target', predictor)
    with repo:
        observation = evaluator.evaluate()

    expected = r2_score(x0, x1)
    pytest.approx(observation['target/r2_score'][0], expected)

    # --- test __call__ ---
    result = evaluator()
    pytest.approx(result['train/main/r2_score'][0], expected)


def _test_r2_score_evaluator_ignore_nan(inputs):
    predictor = DummyPredictor()
    x0, _, x2 = inputs
    dataset = NumpyTupleDataset(x0, x2)

    iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)
    evaluator = R2ScoreEvaluator(
        iterator, predictor, name='train', ignore_nan=True)
    repo = chainer.Reporter()
    repo.add_observer('target', predictor)
    with repo:
        observation = evaluator.evaluate()

    expected = r2_score(x0, x2, ignore_nan=True)
    pytest.approx(observation['target/r2_score'][0], expected)

    # --- test __call__ ---
    result = evaluator()
    pytest.approx(result['train/main/r2_score'][0], expected)


def _test_r2_score_evaluator_ignore_nan_with_nonnan_value(inputs):
    predictor = DummyPredictor()
    x0, x1, _ = inputs
    dataset = NumpyTupleDataset(x0, x1)

    iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)
    evaluator = R2ScoreEvaluator(
        iterator, predictor, name='train', ignore_nan=True)
    repo = chainer.Reporter()
    repo.add_observer('target', predictor)
    with repo:
        observation = evaluator.evaluate()

    expected = r2_score(x0, x1, ignore_nan=True)
    pytest.approx(observation['target/r2_score'][0], expected)

    # --- test __call__ ---
    result = evaluator()
    pytest.approx(result['train/main/r2_score'][0], expected)


def _test_r2_score_evaluator_raw_values(inputs):
    predictor = DummyPredictor()
    x0, x1, _ = inputs
    dataset = NumpyTupleDataset(x0, x1)

    iterator = SerialIterator(dataset, 2, repeat=False, shuffle=False)
    evaluator = R2ScoreEvaluator(
        iterator, predictor, name='train', multioutput='raw_values')
    repo = chainer.Reporter()
    repo.add_observer('target', predictor)
    with repo:
        observation = evaluator.evaluate()

    expected = r2_score(x0, x1, multioutput='raw_values')
    pytest.approx(observation['target/r2_score'][0], expected)

    # --- test __call__ ---
    result = evaluator()
    pytest.approx(result['train/main/r2_score'][0], expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
