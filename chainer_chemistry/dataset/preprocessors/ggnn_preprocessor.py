import numpy

from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor


def construct_discrete_edge_matrix(mol, out_size=-1):
    """construct discrete edge matrix

    Args:
        mol (Chem.Mol):
        out_size (int):

    Returns (numpy.ndarray):

    """

    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise MolFeatureExtractionError('out_size {} is smaller than number '
                                        'of atoms in mol {}'
                                        .format(out_size, N))

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
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.

    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False):
        super(GGNNPreprocessor, self).__init__(add_Hs=add_Hs)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be equal to or larget than '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol):

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)
        return atom_array, adj_array
