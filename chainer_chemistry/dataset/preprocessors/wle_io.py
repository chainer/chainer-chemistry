import numpy as np


from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset


DEBUG = False


def create_datasets(atom_arrays, adj_arrays, teach_signals, wle_arrays=None):
    """
    Expand the atomic_num_arrays with the expanded labels,
    then return valid datasets (tuple of NumpyTupleDataset)

    Args:
        atom_arrays: 3-tuple of list of lists.
                        atom_arrays[i][j][k] is the id of an atom
                        i: train/val/test
                        j: index of a sample (i.e. molcule)
                        k: index of an atom
        adj_arrays: list of list of numpy.array, all mol's adjacnecy tensors
        teach_signals: list of list of numpy.array,
                          all teacher (supervision) signals
        wle_arrays: None (for WLE) or 3-tuple of list of lists (for CWLE and GWLE).

    Returns: 3 tuple of valid datasets (train/vel/test) in NumpyTuppleDataset

    """

    output_datasets = []

    # ToDo: try another indexing: e.g. orignal node label + extneions
    assert len(atom_arrays) == len(adj_arrays) == len(teach_signals)
    if wle_arrays is not None:
        assert len(atom_arrays) == len(wle_arrays)
    for i in range(len(atom_arrays)):
        # We have swaped the axes 0 and 1 for adj-arrays. re-swap
        set_adj_arrays = np.array(adj_arrays[i])
        for m in range(len(set_adj_arrays)):
            set_adj_arrays[m] = np.swapaxes(set_adj_arrays[m], 0, 1)

        if wle_arrays is None:
            dataset = NumpyTupleDataset(np.array(atom_arrays[i]),
                                        set_adj_arrays,
                                        np.array(teach_signals[i]))
        else:
            dataset = NumpyTupleDataset(np.array(atom_arrays[i]),
                                        set_adj_arrays,
                                        np.array(wle_arrays[i]),
                                        np.array(teach_signals[i]))
        output_datasets.append(dataset)
    # end expanded-for

    return output_datasets


def load_dataset_elements(datasets):
    """
    Load all dataset tuples: atom array, adj. array, and teacher signals.

    Args:
        datasets: tuple of NumpyTupleDataset

    Returns:
        - tuple of lists of atom arrays, adj.arrays, and teacher signals.

    """

    if DEBUG:
        print('type(datasets)', type(datasets))  # tuple

    atom_arrays = []  # 3 by num_mols by N by F
    adj_arrays = []  # 3 by num_mols by N by N, or 3 by N by R by N by N by N
    teach_signals = []  # 3 by num_mols by N by (data-dependent)
    for dataset in datasets:

        if DEBUG:
            print('type(dataset)', type(dataset))  # NumpyTupleDataset

        set_atom_arrays = []  # Mol by N
        set_adj_arrays = []  # Mol by N by N or N by R by N by N
        set_teach_signals = []  # Mol by (data-dependent)

        for mol_data in dataset:

            atom_array = mol_data[0]
            adj_array = mol_data[1]
            teach_signal = mol_data[2]

            if DEBUG:
                print("type(mol_data)=", type(mol_data))  # tuple
                print("type(atom_arrray)=", type(atom_array))  # ndarray
                print("type(adj_arrray)=", type(adj_array))  # ndarray
                print("type(teach_signal)=", type(teach_signal))  # ndarray

            set_atom_arrays.append(atom_array)

            # for 3-D tensor, we swap axis here
            set_adj_arrays.append(adj_array.swapaxes(0, 1))
            set_teach_signals.append(teach_signal)
        # end dataset-for

        atom_arrays.append(set_atom_arrays)
        adj_arrays.append(set_adj_arrays)
        teach_signals.append(set_teach_signals)
    # end datasets-for
    return atom_arrays, adj_arrays, teach_signals