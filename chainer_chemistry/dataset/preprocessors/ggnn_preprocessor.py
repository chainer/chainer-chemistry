import numpy

from chainer_chemistry.dataset.graph_dataset.base_graph_data import SparseGraphData  # NOQA
from chainer_chemistry.dataset.graph_dataset.base_graph_dataset import SparseGraphDataset  # NOQA
from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array, construct_discrete_edge_matrix  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms  # NOQA
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA


class GGNNPreprocessor(MolPreprocessor):
    """GGNN Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.

    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False,
                 kekulize=False):
        super(GGNNPreprocessor, self).__init__(
            add_Hs=add_Hs, kekulize=kekulize)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol): Molecule input

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)

        return atom_array, adj_array


class GGNNSparsePreprocessor(GGNNPreprocessor):
    """Sparse GGNN Preprocessor"""

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False,
                 kekulize=False):
        super(GGNNSparsePreprocessor, self).__init__(
            max_atoms=max_atoms, out_size=out_size, add_Hs=add_Hs,
            kekulize=kekulize)

    def construct_sparse_data(self, x, adj, y):
        """Construct `SparseGraphData` from `x`, `adj`, `y`

        Args:
            x (numpy.ndarray): input feature
            adj (numpy.ndarray): adjacency matrix
            y (numpy.ndarray): output label

        Returns:
            SparseGraphData: graph data object for sparse pattern
        """
        edge_index = [[], []]
        edge_attr = []
        label_num, n, _ = adj.shape
        for label in range(label_num):
            for i in range(n):
                for j in range(n):
                    if adj[label, i, j] != 0.:
                        edge_index[0].append(i)
                        edge_index[1].append(i)
                        edge_attr.append(label)
        return SparseGraphData(
            x=x,
            edge_index=numpy.array(edge_index, dtype=numpy.int),
            edge_attr=numpy.array(edge_attr, dtype=numpy.int),
            y=y
        )

    def create_dataset(self, *args, **kwargs):
        """Create `SparseGraphData` from list of `(x, adj, y)`

        Returns:
            SparseGraphDataset: graph dataset object for sparse pattern
        """
        # args: (atom_array, adj_array, label_array)
        data_list = [
            self.construct_sparse_data(x, adj, y) for (x, adj, y) in zip(*args)
        ]
        return SparseGraphDataset(data_list)
