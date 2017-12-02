import os

import numpy
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures

from chainerchem.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainerchem.dataset.preprocessors.common import type_check_num_atoms
from chainerchem.dataset.preprocessors.mol_preprocessor import MolPreprocessor


ATOM = ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br', 'F', 'P', 'I']
MAX_DISTANCE = 2  # 7
DEFAULT_NUM_MAX_ATOMS = 20  # 60  # paper


# --- Atom feature extraction ---
def construct_atom_type_vector(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    n_atom_type = len(ATOM)
    n_atom = mol.GetNumAtoms()
    atom_type_vector = numpy.zeros((num_max_atoms, n_atom_type),
                                   dtype=numpy.float32)
    for i in range(n_atom):
        a = mol.GetAtomWithIdx(i)
        atom_idx = ATOM.index(a.GetSymbol())
        atom_type_vector[i, atom_idx] = 1.0
    return atom_type_vector


def construct_formal_charge_vec(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    n_atom = mol.GetNumAtoms()
    formal_charge_vec = numpy.zeros((num_max_atoms, 1), dtype=numpy.float32)
    for i in range(n_atom):
        a = mol.GetAtomWithIdx(i)
        formal_charge_vec[i, 0] = a.GetFormalCharge()
    return formal_charge_vec


def construct_hybridization_vec(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    n_atom = mol.GetNumAtoms()
    hybridization_vec = numpy.zeros((num_max_atoms, 3), dtype=numpy.float32)
    for i in range(n_atom):
        a = mol.GetAtomWithIdx(i)
        if a.GetHybridization() is None:
            continue
        hybridization_type = str(a.GetHybridization())
        if hybridization_type == 'SP1':
            hybridization_vec[i, 0] = 1.0
        elif hybridization_type == 'SP2':
            hybridization_vec[i, 1] = 1.0
        elif hybridization_type == 'SP3':
            hybridization_vec[i, 2] = 1.0
    return hybridization_vec


def construct_partial_charge_vec(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    AllChem.ComputeGasteigerCharges(mol)
    n = mol.GetNumAtoms()
    partial_charge_vec = numpy.zeros((num_max_atoms, 1), dtype=numpy.float32)
    for i in range(n):
        a = mol.GetAtomWithIdx(i)
        partial_charge_vec[i, 0] = a.GetProp("_GasteigerCharge")
    return partial_charge_vec


def construct_atom_ring_vec(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    nAtom = mol.GetNumAtoms()
    rinfo = mol.GetRingInfo()
    sssr = Chem.GetSymmSSSR(mol)
    ring_feature = numpy.zeros((num_max_atoms, 6,), dtype=numpy.float32)
    for ring in sssr:
        ring = list(ring)
        for i in range(nAtom):
            if i in ring:
                ring_size = len(ring)
                if ring_size >= 3 and ring_size <= 8:
                    ring_feature[i, ring_size - 3] = 1.0
    return ring_feature


def construct_hydrogen_bonding(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)
    feats = factory.GetFeaturesForMol(mol)
    hydrogen_bonding_vec = numpy.zeros((num_max_atoms, 2), dtype=numpy.float32)
    for f in feats:
        if f.GetFamily() == 'Donor':
            idx = f.GetAtomIds()[0]
            hydrogen_bonding_vec[idx, 0] = 1.0
        if f.GetFamily() == 'Acceptor':
            idx = f.GetAtomIds()[0]
            hydrogen_bonding_vec[idx, 1] = 1.0
    return hydrogen_bonding_vec


def make_num_hydrogens(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    n_hydrogen = numpy.zeros((num_max_atoms, 1), dtype=numpy.float32)
    n_atom = mol.GetNumAtoms()
    for i in range(n_atom):
        n = 0
        for j in range(n_atom):
            if i == j:
                continue
            a = mol.GetAtomWithIdx(j)
            if a.GetSymbol() != 'H':
                continue
            k = mol.GetBondBetweenAtoms(i, j)
            if k is not None:
                n += 1
        n_hydrogen[i, 0] = n
    return n_hydrogen


def construct_aromaticity_vec(mol, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    n_atom = mol.GetNumAtoms()
    aromaticity_vec = numpy.zeros((num_max_atoms, 1), dtype=numpy.float32)
    aromatix_atoms = mol.GetAromaticAtoms()
    for a in aromatix_atoms:
        aromaticity_vec[a.GetIdx()] = 1.0

    return aromaticity_vec


def construct_atom_feature(mol, add_Hs, num_max_atoms=DEFAULT_NUM_MAX_ATOMS):
    atom_type_vector = construct_atom_type_vector(mol, num_max_atoms)
    # TODO(nakago): Chilarity
    formal_charge_vec = construct_formal_charge_vec(mol)
    partial_charge_vec = construct_partial_charge_vec(mol)
    atom_ring_vec = construct_atom_ring_vec(mol)
    hybridization_vec = construct_hybridization_vec(mol)
    hydrogen_bonding = construct_hydrogen_bonding(mol)
    aromaticity_vec = construct_aromaticity_vec(mol)
    if add_Hs:
        num_hydrogens_vec = make_num_hydrogens(mol)
        feature = numpy.hstack((atom_type_vector, formal_charge_vec,
                                partial_charge_vec, atom_ring_vec,
                                hybridization_vec, hydrogen_bonding,
                                aromaticity_vec, num_hydrogens_vec))
    else:
        feature = numpy.hstack((atom_type_vector, formal_charge_vec,
                                partial_charge_vec, atom_ring_vec,
                                hybridization_vec, hydrogen_bonding,
                                aromaticity_vec))
    return feature


# --- Pair feature extraction ---
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
        add_Hs (bool): If True, implicit Hs are added.
        use_fixed_atom_feature (bool):
            If True, atom feature is extracted used in original paper.
            If it is False, atomic number is used instead.
    """

    def __init__(self, max_atoms=DEFAULT_NUM_MAX_ATOMS, add_Hs=True,
                 use_fixed_atom_feature=False):
        super(WeaveNetPreprocessor, self).__init__(add_Hs=add_Hs)
        zero_padding = True
        if zero_padding and max_atoms <= 0:
            raise ValueError('max_atoms must be set to positive value when '
                             'zero_padding is True')

        self.max_atoms = max_atoms
        self.add_Hs = add_Hs
        self.zero_padding = zero_padding
        self.use_fixed_atom_feature = use_fixed_atom_feature

    def get_input_features(self, mol):
        """get input features for WeaveNet

        WeaveNetPreprocessor automatically add `H` to `mol`

        Args:
            mol (Mol):

        """
        type_check_num_atoms(mol, self.max_atoms)
        if self.use_fixed_atom_feature:
            # original paper feature extraction
            atom_array = construct_atom_feature(mol, self.add_Hs,
                                                self.max_atoms)
        else:
            # embed id of atomic numbers
            atom_array = construct_atomic_number_array(mol, self.max_atoms)
        pair_feature = construct_pair_feature(mol,
                                              num_max_atoms=self.max_atoms)
        return atom_array, pair_feature
