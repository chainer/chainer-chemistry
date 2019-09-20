import os
from logging import getLogger
import traceback

import numpy
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, Descriptors, rdmolops


from chainer_chemistry.dataset.preprocessors.common \
    import MolFeatureExtractionError
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor

from chainer_chemistry.dataset.utils import GaussianDistance


MAX_ATOM_ELEMENT = 94
ATOM = ['H', 'C', 'N', 'O', 'F']


# create singleton class
class ChemicalFeaturesFactory:
    _instance = None

    @classmethod
    def get_instance(self):
        if not self._instance:
            fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            self._instance = ChemicalFeatures.BuildFeatureFactory(fdefName)

        return self._instance


# --- atom feature extraction ---
def construct_atom_type_vec(mol, num_max_atoms, atom_list=None,
                            include_unknown_atom=False):
    atom_list = atom_list or ATOM
    if include_unknown_atom:
        # all atom not in `atom_list` as considered as "unknown atom"
        # and its index is `len(atom_list)`
        n_atom_type = len(atom_list) + 1
    else:
        n_atom_type = len(atom_list)

    atom_type_vec = numpy.zeros((num_max_atoms, n_atom_type),
                                dtype=numpy.float32)
    for i in range(num_max_atoms):
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


def construct_atom_chirality_vec(mol, num_max_atoms):
    chirality_vec = numpy.zeros((num_max_atoms, 2), dtype=numpy.float32)

    # chiral_cc: (atom_index, chirality) : (1, 'S')
    chiral_cc = Chem.FindMolChiralCenters(mol)
    for chiral_dict in chiral_cc:
        if chiral_dict[1] == 'R':
            chirality_vec[chiral_dict[0]] = [1, 0]
        if chiral_dict[1] == 'S':
            chirality_vec[chiral_dict[0]] = [0, 1]

    return chirality_vec


def construct_atom_ring_vec(mol, num_max_atoms):
    sssr = Chem.GetSymmSSSR(mol)
    ring_feature = numpy.zeros((num_max_atoms, 6,), dtype=numpy.float32)
    for ring in sssr:
        ring = list(ring)
        for i in range(num_max_atoms):
            if i in ring:
                ring_size = len(ring)
                if ring_size >= 3 and ring_size <= 8:
                    ring_feature[i, ring_size - 3] = 1.0
    return ring_feature


def construct_hybridization_vec(mol, num_max_atoms):
    hybridization_vec = numpy.zeros((num_max_atoms, 3), dtype=numpy.float32)
    for i in range(num_max_atoms):
        a = mol.GetAtomWithIdx(i)
        hybridization_type = a.GetHybridization()
        if hybridization_type is None:
            continue
        hybridization_type = str(hybridization_type)
        if hybridization_type == 'SP1':
            hybridization_vec[i, 0] = 1.0
        elif hybridization_type == 'SP2':
            hybridization_vec[i, 1] = 1.0
        elif hybridization_type == 'SP3':
            hybridization_vec[i, 2] = 1.0
    return hybridization_vec


def construct_hydrogen_bonding(mol, num_max_atoms):
    factory = ChemicalFeaturesFactory.get_instance()
    feats = factory.GetFeaturesForMol(mol)
    hydrogen_bonding_vec = numpy.zeros((num_max_atoms, 2), dtype=numpy.float32)
    for f in feats:
        atom_type = f.GetFamily()
        if atom_type == 'Donor':
            idx = f.GetAtomIds()[0]
            hydrogen_bonding_vec[idx, 0] = 1.0
        if atom_type == 'Acceptor':
            idx = f.GetAtomIds()[0]
            hydrogen_bonding_vec[idx, 1] = 1.0
    return hydrogen_bonding_vec


def construct_aromaticity_vec(mol, num_max_atoms):
    aromaticity_vec = numpy.zeros((num_max_atoms, 1), dtype=numpy.float32)
    aromatix_atoms = mol.GetAromaticAtoms()
    for a in aromatix_atoms:
        aromaticity_vec[a.GetIdx()] = 1.0

    return aromaticity_vec


def construct_atom_feature(mol, use_all_feature, atom_list=None,
                           include_unknown_atom=False):
    """construct atom feature

    Args:
        mol (Mol): mol instance
        use_all_feature (bool):
            If True, all atom features are extracted.
            If False, a part of atom features is extracted.
            You can confirm the detail in the paper.
        atom_list (list): list of atoms to extract feature. If None, default
            `ATOM` is used as `atom_list`
        include_unknown_atom (bool): If False, when the `mol` includes atom
            which is not in `atom_list`, it will raise
            `MolFeatureExtractionError`.
            If True, even the atom is not in `atom_list`, `atom_type` is set
            as "unknown" atom.

    Returns:
        atom_feature (numpy.ndarray): 2 dimensional array.
            First axis size is `num_max_atoms`, representing each atom index.
            Second axis size is each atom feature dimension.

    """
    num_max_atoms = mol.GetNumAtoms()
    atom_type_vec = construct_atom_type_vec(
        mol, num_max_atoms, atom_list=atom_list,
        include_unknown_atom=include_unknown_atom)
    atom_chirality_vec = construct_atom_chirality_vec(
        mol, num_max_atoms=num_max_atoms)
    atom_ring_vec = construct_atom_ring_vec(
        mol, num_max_atoms=num_max_atoms)
    hybridization_vec = construct_hybridization_vec(
        mol, num_max_atoms=num_max_atoms)
    hydrogen_bonding = construct_hydrogen_bonding(
        mol, num_max_atoms=num_max_atoms)
    aromaticity_vec = construct_aromaticity_vec(
        mol, num_max_atoms=num_max_atoms)

    if use_all_feature:
        feature = numpy.hstack((atom_type_vec, atom_chirality_vec,
                                atom_ring_vec, hybridization_vec,
                                hydrogen_bonding, aromaticity_vec))
    else:
        feature = construct_atom_type_vec(
            mol, num_max_atoms, atom_list=atom_list,
            include_unknown_atom=include_unknown_atom)

    return feature


# --- pair feature extraction ---
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


def get_is_in_ring(mol):
    """create a cache about whether the atom is in a ring or not

    Args:
        mol (Mol): mol instance

    Returns
        is_in_ring (dict): key is the atom idx, value is the set()
    """
    sssr = Chem.GetSymmSSSR(mol)
    is_in_ring = {}
    ring_idx = 0
    for ring in sssr:
        ring = list(ring)
        for i in ring:
            if i not in is_in_ring:
                is_in_ring[i] = set()
            is_in_ring[i].add(ring_idx)
        ring_idx += 1

    return is_in_ring


def construct_ring_feature_vec(is_in_ring, i, j):
    ring_feature_vec = numpy.zeros((1, ), dtype=numpy.float32)
    if i in is_in_ring and j in is_in_ring and is_in_ring[i] & is_in_ring[j]:
        ring_feature_vec[0] = 1.0

    return ring_feature_vec


def construct_expanded_distance_vec(coordinate_matrix, converter, i, j):
    # calculate the bond length
    distance = numpy.linalg.norm(coordinate_matrix[i] - coordinate_matrix[j])
    # convert from the bond length to vector
    expanded_distance_vec = converter.expand(distance)
    return expanded_distance_vec


def construct_pair_feature(mol, use_all_feature):
    """construct pair feature

    Args:
        mol (Mol): mol instance
        use_all_feature (bool):
            If True, all pair features are extracted.
            If False, a part of pair features is extracted.
            You can confirm the detail in the paper.

    Returns:
        features (numpy.ndarray): 2 dimensional array.
            First axis size is the number of the bond.
            Second axis size is each pair feature dimension.
        bond_idx (numpy.ndarray): 2 dimensional array.
            First axis size is the number of the bond.
            Second axis represents tuple(StartNodeIdx, EndNodeIdx).
    """
    converter = GaussianDistance()

    # prepare the data for extracting the pair feature
    bonds = mol.GetBonds()
    graph_distance_matrix = Chem.GetDistanceMatrix(mol)
    is_in_ring = get_is_in_ring(mol)
    confid = AllChem.EmbedMolecule(mol)
    try:
        coordinate_matrix = rdmolops.Get3DDistanceMatrix(
            mol, confId=confid)
    except ValueError as e:
        logger = getLogger(__name__)
        logger.info('construct_distance_matrix failed, type: {}, {}'
                    .format(type(e).__name__, e.args))
        logger.debug(traceback.format_exc())
        raise MolFeatureExtractionError

    feature = []
    bond_idx = []
    for bond in bonds:
        start_node = bond.GetBeginAtomIdx()
        end_node = bond.GetEndAtomIdx()

        # create pair feature
        distance_feature = numpy.array(
            graph_distance_matrix[start_node][end_node], dtype=numpy.float32)
        bond_feature = construct_bond_vec(mol, start_node, end_node)
        ring_feature = construct_ring_feature_vec(
            is_in_ring, start_node, end_node)

        bond_idx.append((start_node, end_node))
        if use_all_feature:
            expanded_distance_feature = \
                construct_expanded_distance_vec(
                    coordinate_matrix, converter, start_node, end_node)
            feature.append(numpy.hstack((bond_feature, ring_feature,
                                         distance_feature,
                                         expanded_distance_feature)))
        else:
            feature.append(expanded_distance_feature)

    bond_idx = numpy.array(bond_idx).T
    feature = numpy.array(feature)
    return feature, bond_idx


def construct_global_state_feature(mol):
    """construct global state feature

    Args:
        mol (Mol): mol instance

    Returns:
        feature (numpy.ndarray): 1 dimensional array
    """
    n_atom = mol.GetNumAtoms()
    ave_mol_wt = Descriptors.MolWt(mol) / n_atom
    ave_num_of_bonds = len(mol.GetBonds()) / n_atom
    feature = numpy.array([ave_mol_wt, ave_num_of_bonds], dtype=numpy.float32)
    return feature


class MEGNetPreprocessor(MolPreprocessor):
    """MEGNetPreprocessor

    Args:
    For Molecule
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        add_Hs (bool): If True, implicit Hs are added.
        use_all_feature (bool):
            If True, all atom and pair features is extracted.
            If it is False, a part of atom and pair features is extracted.
            You can confirm the detail in the paper.
        atom_list (list): list of atoms to extract feature. If None, default
            `ATOM` is used as `atom_list`
        include_unknown_atom (bool): If False, when the `mol` includes atom
            which is not in `atom_list`, it will raise
            `MolFeatureExtractionError`.
            If True, even the atom is not in `atom_list`, `atom_type` is set
            as "unknown" atom.
        kekulize (bool): If True, Kekulizes the molecule.

    For Crystal
        max_neighbors (int): Max number of atom considered as neighbors
        max_radius (float): Cutoff radius (angstrom)
        expand_dim (int): Dimension converting from distance to vector
    """

    def __init__(self, max_atoms=-1, add_Hs=True,
                 use_all_feature=False, atom_list=None,
                 include_unknown_atom=False, kekulize=False,
                 max_neighbors=12, max_radius=8, expand_dim=100):
        super(MEGNetPreprocessor, self).__init__(
            add_Hs=add_Hs, kekulize=kekulize)

        self.max_atoms = max_atoms
        self.add_Hs = add_Hs
        self.use_all_feature = use_all_feature
        self.atom_list = atom_list
        self.include_unknown_atom = include_unknown_atom
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.expand_dim = expand_dim
        self.gdf = GaussianDistance(centers=numpy.linspace(0, 5, expand_dim))

    def get_input_features(self, mol):
        """get input features from mol object

        Args:
           mol (Mol):

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_feature = construct_atom_feature(mol, self.use_all_feature,
                                              self.atom_list,
                                              self.include_unknown_atom)

        pair_feature, bond_idx = construct_pair_feature(mol,
                                                        self.use_all_feature)
        global_feature = construct_global_state_feature(mol)
        return atom_feature, pair_feature, global_feature, bond_idx

    def get_input_feature_from_crystal(self, structure):
        """get input features from structure object

        Args:
            structure (Structure):

        """
        atom_num = len(structure)
        atom_feature = numpy.zeros(
            (atom_num, MAX_ATOM_ELEMENT), dtype=numpy.float32)
        for i in range(atom_num):
            if structure[i].specie.number < MAX_ATOM_ELEMENT:
                atom_feature[i][structure[i].specie.number] = 1

        # get edge feature vector & bond idx
        neighbor_indexes = []
        neighbor_features = []
        all_neighbors = structure.get_all_neighbors(self.max_radius,
                                                    include_index=True)
        all_neighbors = [sorted(nbrs, key=lambda x: x[1])
                         for nbrs in all_neighbors]
        bond_num = len(all_neighbors)
        for i in range(bond_num):
            nbrs = all_neighbors[i]
            start_node_idx = i
            nbr_feature = numpy.zeros(
                self.max_neighbors, dtype=numpy.float32) + self.max_radius + 1.
            nbr_feature_idx = numpy.zeros((self.max_neighbors, 2),
                                          dtype=numpy.int32)
            nbr_feature_idx[:, 0] = start_node_idx
            nbr_feature_idx[:len(nbrs), 1] = list(
                map(lambda x: x[2], nbrs[:self.max_neighbors]))
            nbr_feature[:len(nbrs)] = list(map(lambda x: x[1],
                                               nbrs[:self.max_neighbors]))
            neighbor_indexes.append(nbr_feature_idx)
            neighbor_features.append(nbr_feature)

        bond_idx = numpy.array(neighbor_indexes).reshape(-1, 2).T
        pair_feature = numpy.array(neighbor_features)
        pair_feature = self.gdf.expand_from_distances(
            pair_feature).reshape(-1, self.expand_dim)
        global_feature = numpy.array([0, 0], dtype=numpy.float32)

        return atom_feature, pair_feature, global_feature, bond_idx
