import numpy

from chainerchem.dataset.preprocessors.common import construct_atomic_numbers
from chainerchem.dataset.preprocessors.common import type_check_num_atoms
from chainerchem.dataset.preprocessors.mol_preprocessor import MolFeatureExtractFailure  # NOQA
from chainerchem.dataset.preprocessors.mol_preprocessor import MolPreprocessor


def construct_discrete_edge_matrix(mol, zero_padding=False,
                                   num_max_atoms=-1):
    """construct discrete edge matrix

    Args:
        mol (Chem.Mol):
        zero_padding (bool):
        num_max_atoms (int):

    Returns (numpy.ndarray):

    """

    if mol is None:
        raise MolFeatureExtractFailure('mol is None')
    N = mol.GetNumAtoms()
    # adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    # size = adj.shape[0]
    size = num_max_atoms if zero_padding else N
    adjs = numpy.zeros((4, size, size), dtype=numpy.float32)
    for i in range(N):
        for j in range(N):
            bond = mol.GetBondBetweenAtoms(i, j)  # type: Chem.Bond
            if bond is not None:
                bond_type = str(bond.GetBondType())
                if bond_type == 'SINGLE':
                    adjs[0, i, j] = 1.0
                elif bond_type == 'DOUBLE':
                    adjs[1, i, j] = 1.0
                elif bond_type == 'TRIPLE':
                    adjs[2, i, j] = 1.0
                elif bond_type == 'AROMATIC':
                    adjs[3, i, j] = 1.0
                else:
                    raise ValueError("[ERROR] Unknown bond type {}"
                                     .format(bond_type))
    return adjs


class GGNNPreprocessor(MolPreprocessor):
    """GGNN Preprocessor

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
        number
        of atoms is more than this value, this data is simply ignored.
        Setting negative value indicates no limit for max atoms.
        zero_padding (bool): True

    """

    def __init__(self, max_atoms=-1, zero_padding=False):
        super(GGNNPreprocessor, self).__init__()
        if zero_padding and max_atoms <= 0:
            raise ValueError('max_atoms must be set to positive value when '
                             'zero_padding is True')

        self.max_atoms = max_atoms
        self.zero_padding = zero_padding

    def get_input_features(self, mol):
        """get descriptor

        Args:
            mol (Mol):

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_numbers(mol, self.max_atoms)
        adj_array = construct_discrete_edge_matrix(
            mol, self.zero_padding, num_max_atoms=self.max_atoms)
        return atom_array, adj_array
