from __future__ import division

from logging import getLogger

import numpy

from chainer.dataset import iterator


class IndexIterator(iterator.Iterator):
    """Index iterator

    IndexIterator is used internally in `BalancedSerialIterator`, as each 
    label's index iterator 

        Args:
            index_list (list): list of int which represents indices.
            shuffle (bool): shuffle flag. If True, indices specified by
                `index_list` will be randomly shuffled.
            num (int): number of indices to be extracted when `___next___` is
                called.

    """

    def __init__(self, index_list, shuffle=True, num=0):
        """
        
        """
        self.index_list = numpy.asarray(index_list)
        assert self.index_list.ndim == 1
        self.index_length = len(index_list)
        self.current_index_list = None
        self.current_pos = 0
        self.shuffle = shuffle
        self.num = num

        self.update_current_index_list()

    def update_current_index_list(self):
        if self.shuffle:
            self.current_index_list = numpy.random.permutation(self.index_list)
        else:
            self.current_index_list = self.index_list

    def __next__(self):
        return self.get_next_indices(self.num)

    def get_next_indices(self, num):
        """get next indices

        Args:
            num (int): number for indices to extract.

        Returns (numpy.ndarray): 1d array of indices

        .. admonition:: Example

           >>> ii = IndexIterator([1, 3, 5, 10], shuffle=True)
           >>> print(ii.get_next_indices(5))
           [ 5  1 10  3 10]
           >>> print(ii.get_next_indices(5))
           [ 3  1  5 10  1]

        """

        indices = []
        if self.current_pos + num < self.index_length:
            indices.append(self.current_index_list[
                           self.current_pos: self.current_pos + num])
            self.current_pos += num
        else:
            indices.append(self.current_index_list[self.current_pos:])
            num -= (self.index_length - self.current_pos)
            q, r = divmod(num, self.index_length)
            indices.append(numpy.tile(self.index_list, q))
            self.update_current_index_list()
            indices.append(self.current_index_list[:r])
            self.current_pos = r

        return numpy.concatenate(indices).ravel()

    def serialize(self, serializer):
        self.current_index_list = serializer('current_index_list',
                                             self.current_index_list)
        self.current_pos = serializer('current_pos', self.current_pos)


class BalancedSerialIterator(iterator.Iterator):

    """Dataset iterator that serially reads the examples with balancing label.

    This is a implementation of :class:`~chainer.dataset.Iterator`
    that visits each example in

    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.

    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes.

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
        # if labels.ndim != 1:
        #     raise ValueError('labels must be 1 dim, but got {} dim array'
        #                      .format(labels.ndim))
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
            # --- below only for included labels ---
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
                # if self._order is not None:
                #     numpy.random.shuffle(self._order)
                if rest > 0:
                    # if self._order is None:
                    #     batch.extend(self.dataset[:rest])
                    # else:
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
