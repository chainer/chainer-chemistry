import numpy
import pytest

from chainer import serializer

from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.iterators.balanced_serial_iterator import BalancedSerialIterator  # NOQA


class DummySerializer(serializer.Serializer):

    def __init__(self, target):
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        target_child = dict()
        self.target[key] = target_child
        return DummySerializer(target_child)

    def __call__(self, key, value):
        self.target[key] = value
        return self.target[key]


class DummyDeserializer(serializer.Deserializer):

    def __init__(self, target):
        super(DummyDeserializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        target_child = self.target[key]
        return DummyDeserializer(target_child)

    def __call__(self, key, value):
        if value is None:
            value = self.target[key]
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, self.target[key])
        else:
            value = type(value)(numpy.asarray(self.target[key]))
        return value


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

    assert len(batch) == 9
    labels_batch = numpy.array([example[-1] for example in batch])

    assert numpy.sum(labels_batch == 0) == 3
    assert numpy.sum(labels_batch == 1) == 3
    assert numpy.sum(labels_batch == 2) == 3


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
        assert len(batch) == 3
        labels_batch = numpy.array([example[-1] for example in batch])
        assert numpy.sum(labels_batch == 0) == 1
        assert numpy.sum(labels_batch == 1) == 1
        assert numpy.sum(labels_batch == 2) == 1


def test_balanced_serial_iterator_serialization():
    _test_balanced_serial_iterator_serialization_no_batch_balancing()
    _test_balanced_serial_iterator_serialization_with_batch_balancing()


def _test_balanced_serial_iterator_serialization_no_batch_balancing():
    x = numpy.arange(8)
    t = numpy.asarray([0, 0, -1, 1, 1, 2, -1, 1])
    iterator = BalancedSerialIterator(NumpyTupleDataset(x, t), batch_size=9,
                                      labels=t, ignore_labels=-1,
                                      batch_balancing=False)
    batch = iterator.next()

    assert iterator.current_position == 0
    assert iterator.epoch == 1
    assert iterator.is_new_epoch

    target = dict()
    iterator.serialize(DummySerializer(target))
    current_index_list_orig = dict()
    current_pos_orig = dict()
    for label, index_iterator in iterator.labels_iterator_dict.items():
        ii_label = 'index_iterator_{}'.format(label)
        current_index_list_orig[ii_label] = index_iterator.current_index_list
        current_pos_orig[ii_label] = index_iterator.current_pos

    iterator = BalancedSerialIterator(NumpyTupleDataset(x, t), batch_size=9,
                                      labels=t, ignore_labels=-1,
                                      batch_balancing=False)
    iterator.serialize(DummyDeserializer(target))
    assert iterator.current_position == 0
    assert iterator.epoch == 1
    assert iterator.is_new_epoch
    for label, index_iterator in iterator.labels_iterator_dict.items():
        ii_label = 'index_iterator_{}'.format(label)
        assert numpy.array_equal(index_iterator.current_index_list,
                                 current_index_list_orig[ii_label])
        assert index_iterator.current_pos == current_pos_orig[ii_label]


def _test_balanced_serial_iterator_serialization_with_batch_balancing():
    x = numpy.arange(8)
    t = numpy.asarray([0, 0, -1, 1, 1, 2, -1, 1])
    iterator = BalancedSerialIterator(NumpyTupleDataset(x, t), batch_size=3,
                                      labels=t, ignore_labels=-1,
                                      batch_balancing=True)
    batch1 = iterator.next()
    batch2 = iterator.next()
    batch3 = iterator.next()

    assert iterator.current_position == 0
    assert iterator.epoch == 1
    assert iterator.is_new_epoch

    target = dict()
    iterator.serialize(DummySerializer(target))
    current_index_list_orig = dict()
    current_pos_orig = dict()
    for label, index_iterator in iterator.labels_iterator_dict.items():
        ii_label = 'index_iterator_{}'.format(label)
        current_index_list_orig[ii_label] = index_iterator.current_index_list
        current_pos_orig[ii_label] = index_iterator.current_pos

    iterator = BalancedSerialIterator(NumpyTupleDataset(x, t), batch_size=3,
                                      labels=t, ignore_labels=-1,
                                      batch_balancing=True)
    iterator.serialize(DummyDeserializer(target))
    assert iterator.current_position == 0
    assert iterator.epoch == 1
    assert iterator.is_new_epoch
    for label, index_iterator in iterator.labels_iterator_dict.items():
        ii_label = 'index_iterator_{}'.format(label)
        assert numpy.array_equal(index_iterator.current_index_list,
                                 current_index_list_orig[ii_label])
        assert index_iterator.current_pos == current_pos_orig[ii_label]


if __name__ == '__main__':
    pytest.main([__file__, '-s', '-v'])
