import numpy
import pytest

from chainer import serializer

from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

from chainer_chemistry.iterators.balanced_serial_iterator import BalancedSerialIterator  # NOQA
from chainer_chemistry.iterators.balanced_serial_iterator import IndexIterator  # NOQA


class DummySerializer(serializer.Serializer):

    def __init__(self, target):
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        self.target[key] = value
        return self.target[key]


def test_index_iterator():
    _test_index_iterator_no_shuffle()
    _test_index_iterator_with_shuffle()


def _test_index_iterator_no_shuffle():
    index_list = [1, 3, 5, 10]
    ii = IndexIterator(index_list, shuffle=False, num=2)

    indices1 = ii.get_next_indices(3)
    indices2 = ii.get_next_indices(6)
    indices3 = ii.__next__()
    # print('shuffle=False, indices', indices1, indices2, indices3)

    assert isinstance(indices1, numpy.ndarray)
    assert len(indices1) == 3
    assert isinstance(indices2, numpy.ndarray)
    assert len(indices2) == 6
    assert isinstance(indices3, numpy.ndarray)
    assert len(indices3) == 2
    assert indices1[0] == index_list[0]
    assert indices1[1] == index_list[1]
    assert indices1[2] == index_list[2]
    assert indices2[0] == index_list[3]
    assert indices2[1] == index_list[0]
    assert indices2[2] == index_list[1]
    assert indices2[3] == index_list[2]
    assert indices2[4] == index_list[3]
    assert indices2[5] == index_list[0]
    assert indices3[0] == index_list[1]
    assert indices3[1] == index_list[2]

    target = dict()
    ii.serialize(DummySerializer(target))
    assert isinstance(ii.current_index_list, numpy.ndarray)
    assert len(ii.current_index_list) == len(index_list)
    assert numpy.array_equal(ii.current_index_list, numpy.asarray(index_list))
    assert ii.current_pos == (3 + 6) % len(index_list) + 2


def _test_index_iterator_with_shuffle():
    index_list = [1, 3, 5, 10]
    ii = IndexIterator(index_list, shuffle=True, num=2)

    indices1 = ii.get_next_indices(3)
    indices2 = ii.get_next_indices(6)
    indices3 = ii.__next__()
    # print('shuffle=True, indices', indices1, indices2, indices3)

    assert isinstance(indices1, numpy.ndarray)
    assert len(indices1) == 3
    assert isinstance(indices2, numpy.ndarray)
    assert len(indices2) == 6
    assert isinstance(indices3, numpy.ndarray)
    assert len(indices3) == 2
    for indices in [indices1, indices2, indices3]:
        for index in indices:
            assert index in index_list

    target = dict()
    ii.serialize(DummySerializer(target))
    for index in ii.current_index_list:
        assert index in index_list
    assert ii.current_pos == (3 + 6) % len(index_list) + 2


def test_balanced_serial_iterator():
    _test_balanced_serial_iterator_no_batch_balancing()
    _test_balanced_serial_iterator_with_batch_balancing()


def _test_balanced_serial_iterator_no_batch_balancing():
    x = numpy.arange(8)
    t = numpy.asarray([0, 0, -1, 1, 1, 2, -1, 1])
    iterator = BalancedSerialIterator(NumpyTupleDataset(x, t), batch_size=9,
                                      labels=t, ignore_labels=-1,
                                      batch_balancing=False)
    # In this case, we have 3 examples of label=1.
    # When BalancedSerialIterator runs, all label examples are sampled 3 times
    # in one epoch.
    # Therefore, number of data is "augmented" as 9
    # 3 (number of label types) * 3 (number of maximum examples in one label)
    expect_N_augmented = 9
    assert iterator.N_augmented == expect_N_augmented
    # iterator.show_label_stats()  # we can show label stats

    batch = iterator.next()
    # print('batch', batch)
    assert len(batch) == 9
    labels_batch = numpy.array([example[-1] for example in batch])
    # print('labels_batch_labels', labels_batch)
    assert numpy.sum(labels_batch == 0) == 3
    assert numpy.sum(labels_batch == 1) == 3
    assert numpy.sum(labels_batch == 2) == 3

    # This does not work!
    # target = dict()
    # iterator.serialize(DummySerializer(target))


def _test_balanced_serial_iterator_with_batch_balancing():
    x = numpy.arange(8)
    t = numpy.asarray([0, 0, -1, 1, 1, 2, -1, 1])
    iterator = BalancedSerialIterator(NumpyTupleDataset(x, t), batch_size=3,
                                      labels=t, ignore_labels=-1,
                                      batch_balancing=True)
    expect_N_augmented = 9
    assert iterator.N_augmented == expect_N_augmented
    batch1 = iterator.next()
    batch2 = iterator.next()
    batch3 = iterator.next()
    for batch in [batch1, batch2, batch3]:
        # print('batch', batch)
        assert len(batch) == 3
        labels_batch = numpy.array([example[-1] for example in batch])
        # print('labels_batch_labels', labels_batch)
        assert numpy.sum(labels_batch == 0) == 1
        assert numpy.sum(labels_batch == 1) == 1
        assert numpy.sum(labels_batch == 2) == 1

    # This does not work!
    # target = dict()
    # iterator.serialize(DummySerializer(target))


if __name__ == '__main__':
    pytest.main([__file__, '-s', '-v'])
