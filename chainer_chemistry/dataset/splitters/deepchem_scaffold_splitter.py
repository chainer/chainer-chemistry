from collections import defaultdict

import numpy
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from chainer_chemistry.dataset.splitters.base_splitter import BaseSplitter


def generate_scaffold(smiles, include_chirality=False):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold\
        .MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


class DeepChemScaffoldSplitter(BaseSplitter):
    """Class for doing data splits by chemical scaffold.

    Referred Deepchem for the implementation,  https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    """
    def _split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1,
               **kwargs):
        print("Using DeepChem Scaffold")
        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test,
                                          1.)
        seed = kwargs.get('seed', None)
        smiles_list = kwargs.get('smiles_list')
        include_chirality = kwargs.get('include_chirality')
        if len(dataset) != len(smiles_list):
            raise ValueError("The lengths of dataset and smiles_list are "
                             "different")

        rng = numpy.random.RandomState(seed)

        scaffolds = {}

        data_len = len(dataset)
        for ind, smiles in enumerate(smiles_list):
            scaffold = generate_scaffold(smiles, include_chirality)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [ scaffold_set for (scaffold, scaffold_set) in sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True) ]
        train_cutoff = frac_train * len(dataset)
        valid_cutoff = (frac_train + frac_valid) * len(dataset)
        train_inds, valid_inds, test_inds = [], [], []

        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set

        return numpy.array(train_inds), numpy.array(valid_inds),\
            numpy.array(test_inds),\


    def train_valid_test_split(self, dataset, smiles_list, frac_train=0.8,
                               frac_valid=0.1, frac_test=0.1, converter=None,
                               return_index=True, seed=None,
                               include_chirality=False, **kwargs):
        """Split dataset into train, valid and test set.

        Split indices are generated by splitting based on the scaffold of small
        molecules.

        Args:
            dataset(NumpyTupleDataset, numpy.ndarray):
                Dataset.
            smiles_list(list):
                SMILES list corresponding to datset.
            seed (int):
                Random seed.
            frac_train(float):
                Fraction of dataset put into training data.
            frac_valid(float):
                Fraction of dataset put into validation data.
            converter(callable):
            return_index(bool):
                If `True`, this function returns only indices. If `False`, this
                function returns splitted dataset.

        Returns:
            SplittedDataset(tuple): splitted dataset or indices

        """
        return super(DeepChemScaffoldSplitter, self)\
            .train_valid_test_split(dataset, frac_train, frac_valid, frac_test,
                                    converter, return_index, seed=seed,
                                    smiles_list=smiles_list,
                                    include_chirality=include_chirality,
                                    **kwargs)

    def train_valid_split(self, dataset, smiles_list, frac_train=0.9,
                          frac_valid=0.1, converter=None, return_index=True,
                          seed=None, include_chirality=False, **kwargs):
        """Split dataset into train and valid set.

        Split indices are generated by splitting based on the scaffold of small
        molecules.

        Args:
            dataset(NumpyTupleDataset, numpy.ndarray):
                Dataset.
            smiles_list(list):
                SMILES list corresponding to datset.
            seed (int):
                Random seed.
            frac_train(float):
                Fraction of dataset put into training data.
            frac_valid(float):
                Fraction of dataset put into validation data.
            converter(callable):
            return_index(bool):
                If `True`, this function returns only indices. If `False`, this
                function returns splitted dataset.

        Returns:
            SplittedDataset(tuple): splitted dataset or indices

        """
        return super(DeepChemScaffoldSplitter, self)\
            .train_valid_split(dataset, frac_train, frac_valid, converter,
                               return_index, seed=seed,
                               smiles_list=smiles_list,
                               include_chirality=include_chirality, **kwargs)