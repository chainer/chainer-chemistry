import unittest

import mock
import numpy
import pytest

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import links

from chainer_chemistry.models.prediction import Classifier


# testing.parameterize takes a list of dictionaries.
# Currently, we cannot set a function to the value of the dictionaries.
# As a workaround, we wrap the function and invoke it in __call__ method.
# See issue #1337 for detail.
class AccuracyWithIgnoreLabel(object):

    def __call__(self, y, t):
        return functions.accuracy(y, t, ignore_label=1)


@pytest.mark.parametrize('accfun', [AccuracyWithIgnoreLabel(), None,
                                    {'user_key': AccuracyWithIgnoreLabel()}])
@pytest.mark.parametrize('compute_accuracy', [True, False])
class TestClassifier(object):

    @classmethod
    def setup_class(cls):
        print('setup class...')
        cls.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        cls.t = numpy.random.randint(3, size=5).astype(numpy.int32)
        cls.y = numpy.random.uniform(-1, 1, (5, 7)).astype(numpy.float32)
        # cls.accfun = accfun
        # cls.compute_accuracy = compute_accuracy
        cls.accfun = None
        cls.compute_accuracy = False

    def check_call(
            self, gpu, label_key, args, kwargs, model_args, model_kwargs,
            accfun, compute_accuracy):
        init_kwargs = {'label_key': label_key}
        if accfun is not None:
            init_kwargs['accfun'] = accfun
        link = Classifier(chainer.Link(), **init_kwargs)

        if gpu:
            xp = cuda.cupy
            link.to_gpu()
        else:
            xp = numpy

        link.compute_accuracy = compute_accuracy

        y = chainer.Variable(self.y)
        link.predictor = mock.MagicMock(return_value=y)

        loss = link(*args, **kwargs)
        link.predictor.assert_called_with(*model_args, **model_kwargs)

        assert hasattr(link, 'y')
        assert link.y is not None

        assert hasattr(link, 'loss')
        xp.testing.assert_allclose(link.loss.data, loss.data)

        assert hasattr(link, 'accuracy')
        if compute_accuracy:
            assert link.metrics is not None
        else:
            assert link.metrics is None

    def test_call_cpu(self, accfun, compute_accuracy):
        self.check_call(
            False, -1, (self.x, self.t), {}, (self.x,), {},
            accfun, compute_accuracy)


    def test_call_three_args_cpu(self, accfun, compute_accuracy):
        self.check_call(
            False, -1, (self.x, self.x, self.t), {}, (self.x, self.x), {},
            accfun, compute_accuracy)

    def test_call_positive_cpu(self, accfun, compute_accuracy):
        self.check_call(
            False, 2, (self.x, self.x, self.t), {}, (self.x, self.x), {},
            accfun, compute_accuracy)

    def test_call_kwargs_cpu(self, accfun, compute_accuracy):
        self.check_call(
            False, 't', (self.x,), {'t': self.t}, (self.x,), {},
            accfun, compute_accuracy)

    def test_call_no_arg_cpu(self, accfun, compute_accuracy):
        self.check_call(
            False, 0, (self.t,), {}, (), {},
            accfun, compute_accuracy)

    @pytest.mark.gpu
    def test_call_gpu(self, accfun, compute_accuracy):
        self.to_gpu()
        self.check_call(
            True, -1, (self.x, self.t), {}, (self.x,), {},
            accfun, compute_accuracy)

    @pytest.mark.gpu
    def test_call_three_args_gpu(self, accfun, compute_accuracy):
        self.to_gpu()
        self.check_call(
            True, -1, (self.x, self.x, self.t), {}, (self.x, self.x), {},
            accfun, compute_accuracy)

    @pytest.mark.gpu
    def test_call_positive_gpu(self, accfun, compute_accuracy):
        self.to_gpu()
        self.check_call(
            True, 2, (self.x, self.x, self.t), {}, (self.x, self.x), {},
            accfun, compute_accuracy)

    @pytest.mark.gpu
    def test_call_kwargs_gpu(self, accfun, compute_accuracy):
        self.to_gpu()
        self.check_call(
            True, 't', (self.x,), {'t': self.t}, (self.x,), {},
            accfun, compute_accuracy)

    @pytest.mark.gpu
    def test_call_no_arg_gpu(self, accfun, compute_accuracy):
        self.to_gpu()
        self.check_call(
            True, 0, (self.t,), {}, (), {}, accfun, compute_accuracy)

    def to_gpu(self):
        self.x = cuda.to_gpu(self.x)
        self.t = cuda.to_gpu(self.t)
        self.y = cuda.to_gpu(self.y)


class TestInvalidArgument(unittest.TestCase):

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


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
