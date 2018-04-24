from __future__ import division

from logging import getLogger

from chainer.dataset import iterator
import numpy

from chainer_chemistry.iterators.index_iterator import IndexIterator


class BalancedSerialIterator(iterator.Iterator):

    """Dataset iterator that serially reads the examples with balancing label.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each minibatch.
        labels (list or numpy.ndarray): 1d array which specifies label feature
            of `dataset`. Its size must be same as the length of `dataset`.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch.
            Otherwise, the order is permanently same as that of `dataset`.
        batch_balancing (bool):  If ``True``, examples are sampled in the way
            that each label examples are roughly evenly sampled in each
            minibatch. Otherwise, the iterator only guarantees that total
            numbers of examples are same among label features.
        ignore_labels (int or list or None): Labels to be ignored.
            If not ``None``, the example whose label is in `ignore_labels`
            are not sampled by this iterator.

    """

    def __init__(self, dataset, batch_size, labels, repeat=True, shuffle=True,
                 batch_balancing=False, ignore_labels=None,
                 logger=getLogger(__name__)):
        assert len(dataset) == len(labels)
        labels = numpy.asarray(labels)
        if len(dataset) != labels.size:
            raise ValueError('dataset length {} and labels size {} must be '
                             'same!'.format(len(dataset), labels.size))
        labels = numpy.ravel(labels)
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = labels
        self.logger = logger

        if ignore_labels is None:
            ignore_labels = []
        elif isinstance(ignore_labels, int):
            ignore_labels = [ignore_labels, ]
        self.ignore_labels = list(ignore_labels)
        self._repeat = repeat
        self._shuffle = shuffle
        self._batch_balancing = batch_balancing

        self.labels_iterator_dict = {}

        max_label_count = -1
        include_label_count = 0
        for label in numpy.unique(labels):
            label_index = numpy.argwhere(labels == label).ravel()
            label_count = len(label_index)
            ii = IndexIterator(label_index, shuffle=shuffle)
            self.labels_iterator_dict[label] = ii
            if label in self.ignore_labels:
                continue
            if max_label_count < label_count:
                max_label_count = label_count
            include_label_count += 1

        self.max_label_count = max_label_count
        self.N_augmented = max_label_count * include_label_count
        self.reset()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = self.N_augmented

        batch = [self.dataset[index] for index in self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                self._update_order()
                if rest > 0:
                    batch.extend([self.dataset[index]
                                  for index in self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self.N_augmented

    @property
    def previous_epoch_detail(self):
        # This iterator saves ``-1`` as _previous_epoch_detail instead of
        # ``None`` because some serializers do not support ``None``.
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('order', self._order)
        self._previous_epoch_detail = serializer(
            'previous_epoch_detail', self._previous_epoch_detail)

        for label, index_iterator in self.labels_iterator_dict.items():
            self.labels_iterator_dict[label].serialize(
                serializer['index_iterator_{}'.format(label)])

    def _update_order(self):
        indices_list = []
        for label, index_iterator in self.labels_iterator_dict.items():
            if label in self.ignore_labels:
                # Not include index of ignore_labels
                continue
            indices_list.append(index_iterator.get_next_indices(
                self.max_label_count))

        if self._batch_balancing:
            # `indices_list` contains same number of indices of each label.
            # we can `transpose` and `ravel` it to get each label's index in
            # sequence, which guarantees that label in each batch is balanced.
            indices = numpy.array(indices_list).transpose().ravel()
            self._order = indices
        else:
            indices = numpy.array(indices_list).ravel()
            self._order = numpy.random.permutation(indices)

    def reset(self):
        self._update_order()
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

    def show_label_stats(self):
        self.logger.warning('   label    count     rate     status')
        total = 0
        for label, index_iterator in self.labels_iterator_dict.items():
            count = len(index_iterator.index_list)
            total += count

        for label, index_iterator in self.labels_iterator_dict.items():
            count = len(index_iterator.index_list)
            rate = count / len(self.dataset)
            status = 'ignored' if label in self.ignore_labels else 'included'
            self.logger.warning('{:>8} {:>8} {:>8.4f} {:>10}'
                                .format(label, count, rate, status))
