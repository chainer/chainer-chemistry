import numpy as np


DEBUG = False


def _index(atom, values):
    idx = values.index(atom)
    if DEBUG:
        print("idx=", idx)
        print("expanded_label=", atom)
    return idx


def to_index(mols, values):
    return np.array([np.array([_index(atom, values) for atom in mol],
                              dtype=np.int32)
                     for mol in mols])


def compress_relation_axis(adj_array):
    ndim = adj_array.ndim
    if ndim == 2:
        return adj_array
    elif ndim == 3:
        return np.sum(adj_array, axis=1, keepdims=False)
    else:
        raise ValueError(
                'ndim of adjacency matrix should be 2 or 3. '
                'Found ndim={}.'.format(ndim))


def _to_string(atom_label, neighbor_labels, with_focus_atom):
    expanded_label = ".".join(map(str, neighbor_labels))
    if with_focus_atom:
        expanded_label = str(atom_label) + "-" + expanded_label
    if DEBUG:
        print("expanded_label=" + expanded_label)
    return expanded_label


def get_neighbor_representation(idx, atom_array, neighbors, with_focus_atom):
    atom_label = atom_array[idx]
    neighbor = neighbors[1][np.where(neighbors[0] == idx)]
    if DEBUG:
        print(neighbor)
        print("len(neighbor_i)=", len(neighbor))
    neighbor_labels = np.sort([atom_array[x] for x in neighbor if x != idx])
    return _to_string(
        atom_label, neighbor_labels, with_focus_atom)


def get_focus_node_label(expanded_label):
    tokens = expanded_label.split('-')
    if len(tokens) != 2:
        raise ValueError(
            'Invalid label={}'.format(expanded_label))
    return tokens[0]
