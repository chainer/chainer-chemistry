import numpy
from rdkit import Chem

from chainerchem.dataset.preprocessors.common import construct_atomic_number_array
from chainerchem.dataset.preprocessors.common import type_check_num_atoms
from chainerchem.dataset.preprocessors.mol_preprocessor import MolPreprocessor


MAX_DISTANCE = 7
# TODO(Nakago): Review default size
DEFAULT_NUM_MAX_ATOMS = 20  # 40


def construct_bond(mol, i, j):
    bond_feature = numpy.zeros((4, ), dtype=numpy.float32)
    k = mol.GetBondBetweenAtoms(i, j)
    if k is not None:
        bond_type = str(k.GetBondType())
        if bond_type == 'SINGLE':
            bond_feature[0] = 1.0
        elif bond_type == 'DOUBLE':
            bond_feature[1] = 1.0
        elif bond_type == 'TRIPLE':
            bond_feature[2] = 1.0
        elif bond_type == 'AROMATIC':
            bond_feature[3] = 1.0
        else:
            raise ValueError("Unknown bond type {}".format(bond_type))
    return bond_feature


def construct_distance(distance_matrix, i, j):
    distance = min(MAX_DISTANCE, int(distance_matrix[i][j]))
    distance_feature = numpy.zeros((MAX_DISTANCE, ), dtype=numpy.float32)
    distance_feature[:distance] = 1.0
    return distance_feature


def construct_ring_feature(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    n_atom = mol.GetNumAtoms()
    # rinfo = mol.GetRingInfo()
    sssr = Chem.GetSymmSSSR(mol)
    ring_feature = numpy.zeros((num_max_atoms ** 2, 1,), dtype=numpy.float32)
    for ring in sssr:
        ring = list(ring)
        n_atom_in_ring = len(ring)
        for i in range(n_atom_in_ring):
            for j in range(n_atom_in_ring):
                a0 = ring[i]
                a1 = ring[j]
                ring_feature[a0 * n_atom + a1] = 1
    return ring_feature


def construct_pair_feature(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    n_atom = mol.GetNumAtoms()
    distance_matrix = Chem.GetDistanceMatrix(mol)
    distance_feature = numpy.zeros((num_max_atoms ** 2, MAX_DISTANCE,),
                                   dtype=numpy.float32)
    for i in range(n_atom):
        for j in range(n_atom):
            distance_feature[i * n_atom + j] = construct_distance(
                distance_matrix, i, j)
    bond_feature = numpy.zeros((num_max_atoms ** 2, 4,), dtype=numpy.float32)
    for i in range(n_atom):
        for j in range(n_atom):
            bond_feature[i * n_atom + j] = construct_bond(mol, i, j)
    ring_feature = construct_ring_feature(mol, num_max_atoms=num_max_atoms)
    feature = numpy.hstack((distance_feature, bond_feature, ring_feature))
    return feature


class WeaveNetPreprocessor(MolPreprocessor):

    """WeaveNetPreprocessor

     WeaveNet must have fixed-size atom list for now, zero_padding option
     is always set to True.

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
    """

    def __init__(self, max_atoms=DEFAULT_NUM_MAX_ATOMS):
        super(WeaveNetPreprocessor, self).__init__(add_Hs=True)
        zero_padding = True
        if zero_padding and max_atoms <= 0:
            raise ValueError('max_atoms must be set to positive value when '
                             'zero_padding is True')

        self.max_atoms = max_atoms
        self.zero_padding = zero_padding

    def get_input_features(self, mol):
        """get input features for WeaveNet

        WeaveNetPreprocessor automatically add `H` to `mol`

        Args:
            mol (Mol):

        """
        type_check_num_atoms(mol, self.max_atoms)
        # TODO(Nakago): support original paper feature extraction
        # currently only embed id is supported.
        atom_array = construct_atomic_number_array(mol, self.max_atoms)
        pair_feature = construct_pair_feature(mol,
                                              num_max_atoms=self.max_atoms)
        return atom_array, pair_feature
