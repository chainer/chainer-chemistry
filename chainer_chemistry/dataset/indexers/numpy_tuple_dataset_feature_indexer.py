from chainer_chemistry.dataset.indexer import BaseFeatureIndexer


class NumpyTupleDatasetFeatureIndexer(BaseFeatureIndexer):
    """FeatureIndexer for NumpyTupleDataset

    Args:
        dataset (NumpyTupleDataset): dataset instance

    """

    def __init__(self, dataset):
        super(NumpyTupleDatasetFeatureIndexer, self).__init__(dataset)
        self.datasets = dataset.get_datasets()

    def features_length(self):
        return len(self.datasets)

    def extract_feature_by_slice(self, slice_index, j):
        return self.datasets[j][slice_index]

    def extract_feature(self, i, j):
        return self.datasets[j][i]
