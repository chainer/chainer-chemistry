class BaseSplitter(object):
    def k_fold_split(self, dataset, k):
        raise NotImplementedError

    def train_valid_test_split(self, dataset, frac_train=.8, frac_valid=.1,
                               frac_test=.1, seed=None, return_index=True):
        train_inds, valid_inds, test_inds = self.split(dataset,
                                                       frac_train=frac_train,
                                                       frac_valid=frac_valid,
                                                       frac_test=frac_test)
        if return_index:
            return train_inds, valid_inds, test_inds
        else:
            return dataset[train_inds], dataset[valid_inds], dataset[test_inds]
