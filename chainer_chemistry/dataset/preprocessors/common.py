"""Common preprocess method is gethered in this file"""
import numpy
from rdkit import Chem
from rdkit.Chem import rdmolops

from chainer_chemistry.config import MAX_ATOMIC_NUM


class MolFeatureExtractionError(Exception):
    pass


# --- Type check ---
def type_check_num_atoms(mol, num_max_atoms=-1):
    """Check number of atoms in `mol` does not exceed `num_max_atoms`

    If number of atoms in `mol` exceeds the number `num_max_atoms`, it will
    raise `MolFeatureExtractionError` exception.

    Args:
        mol (Mol):
        num_max_atoms (int): If negative value is set, not check number of
            atoms.

    """
    num_atoms = mol.GetNumAtoms()
    if num_max_atoms >= 0 and num_atoms > num_max_atoms:
        # Skip extracting feature. ignore this case.
        raise MolFeatureExtractionError(
            'Number of atoms in mol {} exceeds num_max_atoms {}'
            .format(num_atoms, num_max_atoms))


# --- Atom preprocessing ---
def construct_atomic_number_array(mol, out_size=-1):
    """Returns atomic numbers of atoms consisting a molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of returned array.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the tail of
            the array is padded with zeros.

    Returns:
        numpy.ndarray: an array consisting of atomic numbers
            of atoms in the molecule.
    """

    atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n_atom = len(atom_list)

    if out_size < 0:
        return numpy.array(atom_list, dtype=numpy.int32)
    elif out_size >= n_atom:
        # 'empty' padding for atom_list
        # 0 represents empty place for atom
        atom_array = numpy.zeros(out_size, dtype=numpy.int32)
        atom_array[:n_atom] = numpy.array(atom_list, dtype=numpy.int32)
        return atom_array
    else:
        raise ValueError('`out_size` (={}) must be negative or '
                         'larger than or equal to the number '
                         'of atoms in the input molecules (={})'
                         '.'.format(out_size, n_atom))


# --- Adjacency matrix preprocessing ---
def construct_adj_matrix(mol, out_size=-1, self_connection=True):
    """Returns the adjacent matrix of the given molecule.

    This function returns the adjacent matrix of the given molecule.
    Contrary to the specification of
    :func:`rdkit.Chem.rdmolops.GetAdjacencyMatrix`,
    The diagonal entries of the returned matrix are all-one.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of the returned matrix.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the adjacent
            matrix is expanded and zeros are padded to right
            columns and bottom rows.
        self_connection (bool): Add self connection or not.
            If True, diagonal element of adjacency matrix is filled with 1.

    Returns:
        adj_array (numpy.ndarray): The adjacent matrix of the input molecule.
            It is 2-dimensional array with shape (atoms1, atoms2), where
            atoms1 & atoms2 represent from and to of the edge respectively.
            If ``out_size`` is non-negative, the returned
            its size is equal to that value. Otherwise,
            it is equal to the number of atoms in the the molecule.
    """

    adj = rdmolops.GetAdjacencyMatrix(mol)
    s0, s1 = adj.shape
    if s0 != s1:
        raise ValueError('The adjacent matrix of the input molecule'
                         'has an invalid shape: ({}, {}). '
                         'It must be square.'.format(s0, s1))

    if self_connection:
        adj = adj + numpy.eye(s0)
    if out_size < 0:
        adj_array = adj.astype(numpy.float32)
    elif out_size >= s0:
        adj_array = numpy.zeros((out_size, out_size),
                                dtype=numpy.float32)
        adj_array[:s0, :s1] = adj
    else:
        raise ValueError(
            '`out_size` (={}) must be negative or larger than or equal to the '
            'number of atoms in the input molecules (={}).'
            .format(out_size, s0))
    return adj_array


def construct_discrete_edge_matrix(mol, out_size=-1,
                                   add_self_connection_channel=False):
    """Returns the edge-type dependent adjacency matrix of the given molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of the returned matrix.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the adjacent
            matrix is expanded and zeros are padded to right
            columns and bottom rows.
        add_self_connection_channel (bool): Add self connection or not.
            If True, adjacency matrix whose diagonal element filled with 1
            is added to last channel.

    Returns:
        adj_array (numpy.ndarray): The adjacent matrix of the input molecule.
            It is 3-dimensional array with shape (edge_type, atoms1, atoms2),
            where edge_type represents the bond type,
            atoms1 & atoms2 represent from and to of the edge respectively.
            If ``out_size`` is non-negative, its size is equal to that value.
            Otherwise, it is equal to the number of atoms in the the molecule.
    """
    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise ValueError(
            'out_size {} is smaller than number of atoms in mol {}'
            .format(out_size, N))
    if add_self_connection_channel:
        adjs = numpy.zeros((5, size, size), dtype=numpy.float32)
    else:
        adjs = numpy.zeros((4, size, size), dtype=numpy.float32)

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel[bond_type]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[ch, i, j] = 1.0
        adjs[ch, j, i] = 1.0
    if add_self_connection_channel:
        adjs[-1] = numpy.eye(N)
    return adjs


def mol_basic_info_feature(mol, atom_array, adj):
    n_atoms = mol.GetNumAtoms()
    if n_atoms != len(atom_array):
        raise ValueError("[ERROR] n_atoms {} != len(atom_array) {}"
                         .format(n_atoms, len(atom_array)))

    # Note: this is actual number of edges * 2.
    n_edges = adj.sum()
    return numpy.asarray([n_atoms, n_edges])


def mol_atom_type_feature(mol, atom_array, adj):
    atom_count = numpy.bincount(atom_array, minlength=MAX_ATOMIC_NUM + 1)
    return (atom_count > 0).astype(numpy.float)[1:]


def mol_atom_freq_feature(mol, atom_array, adj):
    atom_count = numpy.bincount(atom_array, minlength=MAX_ATOMIC_NUM + 1)
    return (atom_count / len(atom_array))[1:]


def mol_bond_type_feature(mol, atom_array, adj):
    if adj.ndim == 2:
        adj = numpy.expand_dims(adj, axis=0)
    adj = adj.reshape((adj.shape[0], -1))
    return adj.max(axis=1)


def mol_bond_freq_feature(mol, atom_array, adj):
    if adj.ndim == 2:
        adj = numpy.expand_dims(adj, axis=0)
    adj = adj.reshape((adj.shape[0], -1))
    adj_sum = adj.sum()
    if adj_sum == 0:
        return adj.sum(axis=1)
    else:
        return adj.sum(axis=1) / adj_sum


def construct_supernode_feature(mol, atom_array, adj, feature_functions=None):
    """Construct an input feature x' for a supernode

    `out_size` is automatically inferred by `atom_array` and `adj`

    Args:
        mol (rdkit.Chem.Mol): Input molecule
        atom_array (numpy.ndarray) : array of atoms
        adj (numpy.ndarray): N by N 2-way array, or |E| by N by N 3-way array
            where |E| is the number of edgetypes.
        feature_functions (None or list): list of callable

    Returns:
        super_node_x (numpy.ndarray); 1-way array, the supernode feature.
        len(super_node_x) will be 2 + 2 + MAX_ATOMIC_NUM*2 for 2-way adjs,
            2 + 4*2 + MAX_ATOMIC_NUM*2 for 3-way adjs

    """

    if feature_functions is None:
        feature_functions = [
            mol_basic_info_feature, mol_bond_type_feature,
            mol_bond_freq_feature, mol_atom_type_feature,
            mol_atom_freq_feature]
    super_node_x = numpy.concatenate(
        [func(mol, atom_array, adj) for func in feature_functions])
    super_node_x = super_node_x.astype(numpy.float32)
    return super_node_x
