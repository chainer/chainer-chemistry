import os

import numpy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

from chainer_chemistry.config import WEAVE_DEFAULT_NUM_MAX_ATOMS
from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common \
    import MolFeatureExtractionError
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor


ATOM = ['H', 'C', 'N', 'O', 'S', 'Cl', 'Br', 'F', 'P', 'I']
MAX_DISTANCE = 2  # 7


# --- Atom feature extraction ---
def construct_atom_type_vec(mol, num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS,
                            atom_list=None, include_unknown_atom=False):
    atom_list = atom_list or ATOM
    if include_unknown_atom:
        # all atom not in `atom_list` as considered as "unknown atom"
        # and its index is `len(atom_list)`
        n_atom_type = len(atom_list) + 1
    else:
        n_atom_type = len(atom_list)
    n_atom = mol.GetNumAtoms()
    atom_type_vec = numpy.zeros((num_max_atoms, n_atom_type),
                                dtype=numpy.float32)
    for i in range(n_atom):
        a = mol.GetAtomWithIdx(i)
        try:
            atom_idx = atom_list.index(a.GetSymbol())
        except ValueError as e:
            if include_unknown_atom:
                atom_idx = len(atom_list)
            else:
                raise MolFeatureExtractionError(e)
        atom_type_vec[i, atom_idx] = 1.0
    return atom_type_vec


def construct_formal_charge_vec(mol,
                                num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS):
    n_atom = mol.GetNumAtoms()
    formal_charge_vec = numpy.zeros((num_max_atoms, 1), dtype=numpy.float32)
    for i in range(n_atom):
        a = mol.GetAtomWithIdx(i)
        formal_charge_vec[i, 0] = a.GetFormalCharge()
    return formal_charge_vec


def construct_hybridization_vec(mol,
                                num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS):
    # TODO(Oono)
    # Can we enhance preprocessing speed by making factory once
    # prior to calling this function many times?
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


def construct_partial_charge_vec(
        mol, num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS):
    AllChem.ComputeGasteigerCharges(mol)
    n = mol.GetNumAtoms()
    partial_charge_vec = numpy.zeros((num_max_atoms, 1), dtype=numpy.float32)
    for i in range(n):
        a = mol.GetAtomWithIdx(i)
        partial_charge_vec[i, 0] = a.GetProp("_GasteigerCharge")
    return partial_charge_vec


def construct_atom_ring_vec(mol, num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS):
    nAtom = mol.GetNumAtoms()
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


def construct_hydrogen_bonding(mol, num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
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


def construct_num_hydrogens_vec(mol,
                                num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS):
    n_hydrogen_vec = numpy.zeros((num_max_atoms, 1), dtype=numpy.float32)
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
        n_hydrogen_vec[i, 0] = n
    return n_hydrogen_vec


def construct_aromaticity_vec(mol, num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS):
    aromaticity_vec = numpy.zeros((num_max_atoms, 1), dtype=numpy.float32)
    aromatix_atoms = mol.GetAromaticAtoms()
    for a in aromatix_atoms:
        aromaticity_vec[a.GetIdx()] = 1.0

    return aromaticity_vec


def construct_atom_feature(mol, add_Hs,
                           num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS,
                           atom_list=None, include_unknown_atom=False):
    """construct atom feature

    Args:
        mol (Mol): mol instance
        add_Hs (bool): if the `mol` instance was added Hs, set True.
        num_max_atoms (int): number of max atoms
        atom_list (list): list of atoms to extract feature. If None, default
            `ATOM` is used as `atom_list`
        include_unknown_atom (bool): If False, when the `mol` includes atom
            which is not in `atom_list`, it will raise
            `MolFeatureExtractionError`.
            If True, even the atom is not in `atom_list`, `atom_type` is set
            as "unknown" atom.

    Returns (numpy.ndarray): 2 dimensional array. First axis size is
        `num_max_atoms`, representing each atom index.
        Second axis for feature.

    """
    atom_type_vec = construct_atom_type_vec(
        mol, num_max_atoms, atom_list=atom_list,
        include_unknown_atom=include_unknown_atom)
    # TODO(nakago): Chilarity
    formal_charge_vec = construct_formal_charge_vec(mol)
    partial_charge_vec = construct_partial_charge_vec(mol)
    atom_ring_vec = construct_atom_ring_vec(mol)
    hybridization_vec = construct_hybridization_vec(mol)
    hydrogen_bonding = construct_hydrogen_bonding(mol)
    aromaticity_vec = construct_aromaticity_vec(mol)
    if add_Hs:
        num_hydrogens_vec = construct_num_hydrogens_vec(mol)
        feature = numpy.hstack((atom_type_vec, formal_charge_vec,
                                partial_charge_vec, atom_ring_vec,
                                hybridization_vec, hydrogen_bonding,
                                aromaticity_vec, num_hydrogens_vec))
    else:
        feature = numpy.hstack((atom_type_vec, formal_charge_vec,
                                partial_charge_vec, atom_ring_vec,
                                hybridization_vec, hydrogen_bonding,
                                aromaticity_vec))
    return feature


# --- Pair feature extraction ---
def construct_bond_vec(mol, i, j):
    bond_feature_vec = numpy.zeros((4, ), dtype=numpy.float32)
    k = mol.GetBondBetweenAtoms(i, j)
    if k is not None:
        bond_type = str(k.GetBondType())
        if bond_type == 'SINGLE':
            bond_feature_vec[0] = 1.0
        elif bond_type == 'DOUBLE':
            bond_feature_vec[1] = 1.0
        elif bond_type == 'TRIPLE':
            bond_feature_vec[2] = 1.0
        elif bond_type == 'AROMATIC':
            bond_feature_vec[3] = 1.0
        else:
            raise ValueError("Unknown bond type {}".format(bond_type))
    return bond_feature_vec


def construct_distance_vec(distance_matrix, i, j):
    distance = min(MAX_DISTANCE, int(distance_matrix[i][j]))
    distance_feature = numpy.zeros((MAX_DISTANCE, ), dtype=numpy.float32)
    distance_feature[:distance] = 1.0
    return distance_feature


def construct_ring_feature_vec(mol, num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS):
    n_atom = mol.GetNumAtoms()
    sssr = Chem.GetSymmSSSR(mol)
    ring_feature_vec = numpy.zeros(
        (num_max_atoms ** 2, 1,), dtype=numpy.float32)
    for ring in sssr:
        ring = list(ring)
        n_atom_in_ring = len(ring)
        for i in range(n_atom_in_ring):
            for j in range(n_atom_in_ring):
                a0 = ring[i]
                a1 = ring[j]
                ring_feature_vec[a0 * n_atom + a1] = 1
    return ring_feature_vec


def construct_pair_feature(mol, num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS):
    """construct pair feature

    Args:
        mol (Mol): mol instance
        num_max_atoms (int): number of max atoms

    Returns (numpy.ndarray): 2 dimensional array. First axis size is
        `num_max_atoms` ** 2, representing index of each atom pair.
        Second axis for feature.

    """
    n_atom = mol.GetNumAtoms()
    distance_matrix = Chem.GetDistanceMatrix(mol)
    distance_feature = numpy.zeros((num_max_atoms ** 2, MAX_DISTANCE,),
                                   dtype=numpy.float32)
    for i in range(n_atom):
        for j in range(n_atom):
            distance_feature[i * n_atom + j] = construct_distance_vec(
                distance_matrix, i, j)
    bond_feature = numpy.zeros((num_max_atoms ** 2, 4,), dtype=numpy.float32)
    for i in range(n_atom):
        for j in range(n_atom):
            bond_feature[i * n_atom + j] = construct_bond_vec(mol, i, j)
    ring_feature = construct_ring_feature_vec(mol, num_max_atoms=num_max_atoms)
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
        atom_list (list): list of atoms to extract feature. If None, default
            `ATOM` is used as `atom_list`
        include_unknown_atom (bool): If False, when the `mol` includes atom
            which is not in `atom_list`, it will raise
            `MolFeatureExtractionError`.
            If True, even the atom is not in `atom_list`, `atom_type` is set
            as "unknown" atom.
    """

    def __init__(self, max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS, add_Hs=True,
                 use_fixed_atom_feature=False, atom_list=None,
                 include_unknown_atom=False):
        super(WeaveNetPreprocessor, self).__init__(add_Hs=add_Hs)
        zero_padding = True
        if zero_padding and max_atoms <= 0:
            raise ValueError('max_atoms must be set to positive value when '
                             'zero_padding is True')

        self.max_atoms = max_atoms
        self.add_Hs = add_Hs
        self.zero_padding = zero_padding
        self.use_fixed_atom_feature = use_fixed_atom_feature
        self.atom_list = atom_list
        self.include_unknown_atom = include_unknown_atom

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
                                                self.max_atoms, self.atom_list,
                                                self.include_unknown_atom)
        else:
            # embed id of atomic numbers
            atom_array = construct_atomic_number_array(mol, self.max_atoms)
        pair_feature = construct_pair_feature(mol,
                                              num_max_atoms=self.max_atoms)
        return atom_array, pair_feature
