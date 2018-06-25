import numpy
from chainer_chemistry.dataset.splitters.base_splitter import BaseSplitter


class RandomSplitter(BaseSplitter):
    def _split(self, dataset, **kwargs):
        frac_train = kwargs.get('frac_train', 0.8)
        frac_valid = kwargs.get('frac_valid', 0.1)
        frac_test = kwargs.get('frac_test', 0.1)
        seed = kwargs.get('seed', None)
        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test,
                                          1.)

        if seed is not None:
            perm = numpy.random.RandomState(seed).permutation(len(dataset))
        else:
            perm = numpy.random.permutation(len(dataset))
        train_data_size = int(len(dataset) * frac_train)
        valid_data_size = int(len(dataset) * frac_valid)
        return (perm[:train_data_size],
                perm[train_data_size:train_data_size + valid_data_size],
                perm[train_data_size + valid_data_size:])
