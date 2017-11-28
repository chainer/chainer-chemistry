import os
import six

import numpy

from chainerchem.dataset.indexer import BaseFeatureIndexer


class NumpyTupleDatasetFeatureIndexer(BaseFeatureIndexer):
    """FeatureIndexer for NumpyTupleDataset"""

    def __init__(self, dataset):
        super(NumpyTupleDatasetFeatureIndexer, self).__init__(dataset)
        self.datasets = dataset._datasets

    def features_length(self):
        return len(self.datasets)

    def extract_feature_by_slice(self, slice_index, j):
        return self.datasets[j][slice_index]

    def extract_feature(self, i, j):
        return self.datasets[j][i]
