import collections

import numpy as np

from chainer_chemistry.dataset.preprocessors import wle_util


def update_atom_arrays(atom_arrays, adj_arrays, cutoff, with_focus_atom=True):
    expanded_atom_lists, labels_frequencies = list_all_expanded_labels(
            atom_arrays, adj_arrays, with_focus_atom)
    if cutoff > 0:
        expanded_atom_lists, labels_frequencies = shrink_expanded_labels(
                expanded_atom_lists, labels_frequencies, cutoff)
    expanded_labels = list(labels_frequencies.keys())
    atom_arrays = [wle_util.to_index(l, expanded_labels)
                        for l in expanded_atom_lists]
    return atom_arrays, labels_frequencies


def shrink_expanded_labels(expanded_atom_lists,
                           labels_frequencies,
                           cutoff):
    """
    Cut off the few-appearance expanded labels

    Args:
        expanded_atom_lists: tuple of list of expanded labels
        labels_frequencies: list of label apperacne frequencies
        cutoff: int, frequency cut of expanded labels

    Returns:
        - 3 (train/val/test) tuple of expanded atom arrays
          (all nodes are associated with string representations of expanded signals)
        - dictionary of frequencies all labels (key: label, value: frequency)
    """

    # atom_array values are expanded label "STRING", not numbers
    new_expanded_atom_lists = []
    new_labels_frequencies = collections.defaultdict(lambda: 0)

    # for each train/val/test, do
    for set_expanded_atom_list in expanded_atom_lists:

        # for each molecule sample, do
        new_set_expanded_atom_list = []
        for expanded_atom_list in set_expanded_atom_list:

            mol_expanded_atom_list = []
            # for each node i in the molecule,
            # get the neighbor's atom label (number index)
            for expanded_label in expanded_atom_list:
                freq = labels_frequencies[expanded_label]

                # check frequency here
                if freq > cutoff:
                    label = expanded_label
                else:
                    label = wle_util.get_focus_node_label(expanded_label)
                mol_expanded_atom_list.append(label)
                new_labels_frequencies[label] = new_labels_frequencies[label] + 1
                # end cutoff-ifelse

            # end i-for
            new_set_expanded_atom_list.append(mol_expanded_atom_list)
        # end zip(atom_arrays, adj_array)-for

        new_expanded_atom_lists.append(new_set_expanded_atom_list)
    # end zip(atom_arrays, adj_array)-for

    return new_expanded_atom_lists, dict(new_labels_frequencies)


def list_all_expanded_labels(atom_arrays, adj_arrays, with_focus_atom=True):
    """
    Exapnd all nodes into WLE representation. At the same time, return the list of all labels after expansion

    Args:
        atom_arrays: 3 (train/val/test) tuple of atom arrays
        adj_arrays: 3 (train/val/test) tuple of adj.arrays
        with_focus_atom: bool, if True, the expanded label starts from the original atom label ("C-ON-OFN")
                                   if False, the exnapndd label do not include the original atom albel("-CN-OFN")

    Returns:
        - 3 (train/val/test) tuple of expanded atom arrays
             (all nodes are associated with string representations of expanded signals)
        - list of all labels appeared in the expanded atom arrays.
        - dictionary of frequencies all labels (key: label, value: frequency)

    """

    expanded_atom_lists = []  # atom_array values are expanded label "STRING", not numbers
    labels_frequencies = collections.defaultdict(lambda: 0)

    # for each train/val/test, do
    for set_atom_arrays, set_adj_arrays in zip(atom_arrays, adj_arrays):
        # for each molecule sample, do
        set_expanded_atom_list = []
        for atom_array, adj_array in zip(set_atom_arrays, set_adj_arrays):
            N = len(atom_array)  # number of molecules
            # atom_array: N by F
            # adj_array: N by N or N by R by N

            # compress the relation axis
            adj_array = wle_util.compress_relation_axis(adj_array)
            assert adj_array.shape == (N, N)
            # find neighbors
            # array[0]: row index array[1]: column index
            neighbors = np.nonzero(adj_array)

            mol_expanded_atom_list = []
            # for each node i in the molecule,
            # get the neighbor's atom label (number index)
            for i in range(N):
                expanded_label = wle_util.get_neighbor_representation(
                    i, atom_array, neighbors, with_focus_atom)
                mol_expanded_atom_list.append(expanded_label)
                labels_frequencies[expanded_label] = labels_frequencies[expanded_label] + 1
            # end i-for
            set_expanded_atom_list.append(mol_expanded_atom_list)
        # end zip(atom_arrays, adj_array)-for

        expanded_atom_lists.append(set_expanded_atom_list)
    # end zip(atom_arrays, adj_array)-for

    # Convert to a normal dictionary because
    # we cannot pickle defaultdicts with lambdas.
    return expanded_atom_lists, dict(labels_frequencies)
