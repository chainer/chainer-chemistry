import mock
import numpy
import pytest

import chainer
from chainer import cuda
from chainer import functions
from chainer import links
from chainer import reporter

from chainer_chemistry.models.prediction import Classifier


# testing.parameterize takes a list of dictionaries.
# Currently, we cannot set a function to the value of the dictionaries.
# As a workaround, we wrap the function and invoke it in __call__ method.
# See issue #1337 for detail.
class AccuracyWithIgnoreLabel(object):

    def __call__(self, y, t):
        return functions.accuracy(y, t, ignore_label=1)


class DummyPredictor(chainer.Chain):
    def __call__(self, x):
        return x


@pytest.mark.parametrize(
    'metrics_fun', [AccuracyWithIgnoreLabel(), None,
                    {'user_key': AccuracyWithIgnoreLabel()}])
@pytest.mark.parametrize('compute_metrics', [True, False])
class TestClassifier(object):

    def setup_method(self, method):
        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=5).astype(numpy.int32)
        self.y = numpy.random.uniform(-1, 1, (5, 7)).astype(numpy.float32)

    def check_call(
            self, gpu, label_key, args, kwargs, model_args, model_kwargs,
            metrics_fun, compute_metrics):
        init_kwargs = {'label_key': label_key}
        if metrics_fun is not None:
            init_kwargs['metrics_fun'] = metrics_fun
        link = Classifier(chainer.Link(), **init_kwargs)

        if gpu:
            xp = cuda.cupy
            link.to_gpu()
        else:
            xp = numpy

        link.compute_metrics = compute_metrics

        y = chainer.Variable(self.y)
        link.predictor = mock.MagicMock(return_value=y)

        loss = link(*args, **kwargs)
        link.predictor.assert_called_with(*model_args, **model_kwargs)

        assert hasattr(link, 'y')
        assert link.y is not None

        assert hasattr(link, 'loss')
        xp.testing.assert_allclose(link.loss.data, loss.data)

        assert hasattr(link, 'metrics')
        if compute_metrics:
            assert link.metrics is not None
        else:
            assert link.metrics is None

    def test_call_cpu(self, metrics_fun, compute_metrics):
        self.check_call(
            False, -1, (self.x, self.t), {}, (self.x,), {},
            metrics_fun, compute_metrics)

    def test_call_three_args_cpu(self, metrics_fun, compute_metrics):
        self.check_call(
            False, -1, (self.x, self.x, self.t), {}, (self.x, self.x), {},
            metrics_fun, compute_metrics)

    def test_call_positive_cpu(self, metrics_fun, compute_metrics):
        self.check_call(
            False, 2, (self.x, self.x, self.t), {}, (self.x, self.x), {},
            metrics_fun, compute_metrics)

    def test_call_kwargs_cpu(self, metrics_fun, compute_metrics):
        self.check_call(
            False, 't', (self.x,), {'t': self.t}, (self.x,), {},
            metrics_fun, compute_metrics)

    def test_call_no_arg_cpu(self, metrics_fun, compute_metrics):
        self.check_call(
            False, 0, (self.t,), {}, (), {},
            metrics_fun, compute_metrics)

    @pytest.mark.gpu
    def test_call_gpu(self, metrics_fun, compute_metrics):
        self.to_gpu()
        self.check_call(
            True, -1, (self.x, self.t), {}, (self.x,), {},
            metrics_fun, compute_metrics)

    @pytest.mark.gpu
    def test_call_three_args_gpu(self, metrics_fun, compute_metrics):
        self.to_gpu()
        self.check_call(
            True, -1, (self.x, self.x, self.t), {}, (self.x, self.x), {},
            metrics_fun, compute_metrics)

    @pytest.mark.gpu
    def test_call_positive_gpu(self, metrics_fun, compute_metrics):
        self.to_gpu()
        self.check_call(
            True, 2, (self.x, self.x, self.t), {}, (self.x, self.x), {},
            metrics_fun, compute_metrics)

    @pytest.mark.gpu
    def test_call_kwargs_gpu(self, metrics_fun, compute_metrics):
        self.to_gpu()
        self.check_call(
            True, 't', (self.x,), {'t': self.t}, (self.x,), {},
            metrics_fun, compute_metrics)

    @pytest.mark.gpu
    def test_call_no_arg_gpu(self, metrics_fun, compute_metrics):
        self.to_gpu()
        self.check_call(
            True, 0, (self.t,), {}, (), {}, metrics_fun, compute_metrics)

    def to_gpu(self):
        self.x = cuda.to_gpu(self.x)
        self.t = cuda.to_gpu(self.t)
        self.y = cuda.to_gpu(self.y)

    def test_report_key(self, metrics_fun, compute_metrics):
        repo = chainer.Reporter()

        link = Classifier(predictor=DummyPredictor(),
                          metrics_fun=metrics_fun)
        link.compute_metrics = compute_metrics
        repo.add_observer('target', link)
        with repo:
            observation = {}
            with reporter.report_scope(observation):
                link(self.x, self.t)

        # print('observation ', observation)
        actual_keys = set(observation.keys())
        if compute_metrics:
            if metrics_fun is None:
                assert set(['target/loss']) == actual_keys
            elif isinstance(metrics_fun, dict):
                assert set(['target/loss', 'target/user_key']) == actual_keys
            elif callable(metrics_fun):
                assert set(['target/loss', 'target/accuracy']) == actual_keys
            else:
                raise TypeError()
        else:
            assert set(['target/loss']) == actual_keys


class TestInvalidArgument(object):

    @classmethod
    def setup_class(cls):
        cls.link = Classifier(links.Linear(10, 3))
        cls.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)

    def check_invalid_argument(self):
        x = chainer.Variable(self.link.xp.asarray(self.x))
        with pytest.raises(TypeError):
            # link.__call__ raises TypeError as the number of arguments
            # is illegal
            self.link(x)

    def test_invalid_argument_cpu(self):
        self.check_invalid_argument()

    @pytest.mark.gpu
    def test_invalid_argument_gpu(self):
        self.link.to_gpu()
        self.check_invalid_argument()


class TestInvalidLabelKey(object):

    @classmethod
    def setup_class(cls):
        cls.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)

    def test_invalid_label_key_type(self):
        with pytest.raises(TypeError):
            Classifier(links.Linear(10, 3), label_key=None)

    def check_invalid_key(self, gpu, label_key):
        link = Classifier(links.Linear(10, 3), label_key=label_key)
        if gpu:
            link.to_gpu()
        x = chainer.Variable(link.xp.asarray(self.x))
        with pytest.raises(ValueError):
            link(x)

    def test_invalid_index_cpu(self):
        self.check_invalid_key(False, 1)

    @pytest.mark.gpu
    def test_invalid_argument_gpu(self):
        self.check_invalid_key(True, 1)

    def test_invalid_index_too_small_cpu(self):
        self.check_invalid_key(False, -2)

    @pytest.mark.gpu
    def test_invalid_index_too_small_gpu(self):
        self.check_invalid_key(True, -2)

    def test_invalid_str_key_cpu(self):
        self.check_invalid_key(False, 't')

    @pytest.mark.gpu
    def test_invalid_str_key_gpu(self):
        self.check_invalid_key(True, 't')


class TestClassifierPrediction(object):

    @classmethod
    def setup_class(cls):
        cls.predictor = DummyPredictor()
        cls.x = numpy.array([[0., 1.], [-1., -2.], [4., 0.]],
                            dtype=numpy.float32)
        cls.t = numpy.array([1, 0, 0], dtype=numpy.int32)

    def test_predict_cpu(self):
        clf = Classifier(self.predictor)
        actual_t = clf.predict(self.x)
        assert actual_t.shape == (3,)
        assert actual_t.dtype == numpy.int32
        assert numpy.alltrue(actual_t == self.t)

    @pytest.mark.gpu
    def test_predict_gpu(self):
        clf = Classifier(self.predictor, device=0)
        actual_t = clf.predict(self.x)
        assert numpy.alltrue(actual_t == self.t)

    def check_predict_proba(self, device):
        clf = Classifier(self.predictor, device=device)
        actual_y = clf.predict_proba(self.x)
        assert actual_y.shape == (3, 2)
        assert actual_y.dtype == numpy.float32
        assert numpy.alltrue(0 <= actual_y)
        assert numpy.alltrue(actual_y <= 1.)

        actual_t = numpy.argmax(actual_y, axis=1)
        assert numpy.alltrue(actual_t == self.t)

    def test_predict_proba_cpu(self):
        self.check_predict_proba(-1)

    @pytest.mark.gpu
    def test_predict_proba_gpu(self):
        self.check_predict_proba(0)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
