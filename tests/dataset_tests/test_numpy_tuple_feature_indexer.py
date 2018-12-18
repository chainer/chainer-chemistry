import numpy
import pytest


from chainer_chemistry.dataset.indexers.numpy_tuple_dataset_feature_indexer import NumpyTupleDatasetFeatureIndexer  # NOQA
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


@pytest.fixture
def data():
    a = numpy.array([1, 2])
    b = numpy.array([4, 5])
    c = numpy.array([[6, 7, 8], [8, 9, 10]])
    return a, b, c


@pytest.fixture
def indexer(data):
    dataset = NumpyTupleDataset(*data)
    indexer = NumpyTupleDatasetFeatureIndexer(dataset)
    return indexer


class TestNumpyTupleDatasetFeatureIndexer(object):

    def test_feature_length(self, indexer):
        assert indexer.features_length() == 3

    @pytest.mark.parametrize('slice_index', [
        0, 1, slice(0, 2, None), slice(0, 0, None)])
    @pytest.mark.parametrize('j', [0, 1])
    def test_extract_feature_by_slice(self, indexer, data, slice_index, j):
        numpy.testing.assert_array_equal(
            indexer.extract_feature_by_slice(slice_index, j),
            data[j][slice_index])
        # indexer's __getitem__ should call `extract_feature_by_slice` method,
        # result should be same with above.
        numpy.testing.assert_array_equal(
            indexer[slice_index, j],
            data[j][slice_index])

    @pytest.mark.parametrize('ndarray_index', [
        numpy.asarray([0, 1]), numpy.asarray([1]),
        numpy.asarray([], dtype=numpy.int32)])
    @pytest.mark.parametrize('j', [0, 1])
    def test_extract_feature_by_ndarray(self, indexer, data, ndarray_index, j):
        numpy.testing.assert_array_equal(
            indexer.extract_feature_by_slice(ndarray_index, j),
            data[j][ndarray_index])
        # indexer's __getitem__ should call `extract_feature_by_slice` method,
        # result should be same with above.
        numpy.testing.assert_array_equal(
            indexer[ndarray_index, j],
            data[j][ndarray_index])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
