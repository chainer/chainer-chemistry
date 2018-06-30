import os
import tempfile

import numpy
import pytest
import six

from chainer_chemistry.datasets import NumpyTupleDataset


@pytest.fixture
def data():
    a = numpy.array([1, 2])
    b = numpy.array([4, 5])
    c = numpy.array([[6, 7, 8], [8, 9, 10]])
    return a, b, c


@pytest.fixture
def long_data():
    a = numpy.array([1, 2, 3, 4])
    b = numpy.array([4, 5, 6, 7])
    c = numpy.array([[6, 7, 8], [8, 9, 10], [11, 12, 13], [14, 15, 16]])
    return a, b, c


class TestNumpyTupleDataset(object):

    def test_len(self, data):
        dataset = NumpyTupleDataset(*data)
        assert len(dataset) == 2

    @pytest.mark.parametrize('index', [0, 1])
    def test_get_item_integer_index(self, data, index):
        dataset = NumpyTupleDataset(*data)
        actual = dataset[index]

        assert len(actual) == len(data)
        for a, d in six.moves.zip(actual, data):
            numpy.testing.assert_array_equal(a, d[index])

    @pytest.mark.parametrize('index', [slice(0, 2, None)])
    def test_get_item_slice_index(self, data, index):
        dataset = NumpyTupleDataset(*data)
        actual = dataset[index]

        batches = [d[index] for d in data]
        length = len(batches[0])
        expect = [tuple([batch[i] for batch in batches])
                  for i in six.moves.range(length)]

        assert len(actual) == len(expect)
        for tuple_a, tuple_e in six.moves.zip(actual, expect):
            assert len(tuple_a) == len(tuple_e)
            for a, e in six.moves.zip(tuple_a, tuple_e):
                numpy.testing.assert_array_equal(a, e)

    @pytest.mark.parametrize('index', [numpy.asarray([2, 0]),
                                       numpy.asarray([1])])
    def test_get_item_ndarray_index(self, long_data, index):
        dataset = NumpyTupleDataset(*long_data)
        actual = dataset[index]

        batches = [d[index] for d in long_data]
        length = len(batches[0])
        expect = [tuple([batch[i] for batch in batches])
                  for i in six.moves.range(length)]

        assert len(actual) == len(expect)
        for tuple_a, tuple_e in six.moves.zip(actual, expect):
            assert len(tuple_a) == len(tuple_e)
            for a, e in six.moves.zip(tuple_a, tuple_e):
                numpy.testing.assert_array_equal(a, e)

    @pytest.mark.parametrize('index', [[2, 0], [1]])
    def test_get_item_list_index(self, long_data, index):
        dataset = NumpyTupleDataset(*long_data)
        actual = dataset[index]

        batches = [d[index] for d in long_data]
        length = len(batches[0])
        expect = [tuple([batch[i] for batch in batches])
                  for i in six.moves.range(length)]

        assert len(actual) == len(expect)
        for tuple_a, tuple_e in six.moves.zip(actual, expect):
            assert len(tuple_a) == len(tuple_e)
            for a, e in six.moves.zip(tuple_a, tuple_e):
                numpy.testing.assert_array_equal(a, e)

    def test_invalid_datasets(self):
        a = numpy.array([1, 2])
        b = numpy.array([1, 2, 3])
        with pytest.raises(ValueError):
            NumpyTupleDataset(a, b)

    def test_save_load(self, data):
        tmp_cache_path = os.path.join(tempfile.mkdtemp(), 'tmp.npz')
        dataset = NumpyTupleDataset(*data)
        NumpyTupleDataset.save(tmp_cache_path, dataset)
        assert os.path.exists(tmp_cache_path)
        load_dataset = NumpyTupleDataset.load(tmp_cache_path)
        os.remove(tmp_cache_path)

        assert len(dataset._datasets) == len(load_dataset._datasets)
        for a, d in six.moves.zip(dataset._datasets, load_dataset._datasets):
            numpy.testing.assert_array_equal(a, d)

    def test_get_datasets(self, data):
        dataset = NumpyTupleDataset(*data)
        datasets = dataset.get_datasets()
        assert len(datasets) == len(data)
        for i in range(len(datasets)):
            numpy.testing.assert_array_equal(datasets[i], data[i])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
