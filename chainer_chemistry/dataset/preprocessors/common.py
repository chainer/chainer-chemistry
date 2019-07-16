"""Common preprocess method is gethered in this file"""

from collections import Counter
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


def construct_discrete_edge_matrix(mol, out_size=-1, self_connection=False):
    """Returns the edge-type dependent adjacency matrix of the given molecule.

    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        out_size (int): The size of the returned matrix.
            If this option is negative, it does not take any effect.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the adjacent
            matrix is expanded and zeros are padded to right
            columns and bottom rows.

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
    if self_connection:
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
    if self_connection:
        adjs[-1] = numpy.eye(N)
    return adjs

def construct_supernode_feature(mol, atom_array, adjs, largest_atomic_number=MAX_ATOMIC_NUM, out_size=-1):
    """
    Construct an input feature x' for a supernode

    Args:
        mol (rdkit.Chem.Mol): Input molecule
        atom_array (numpy.ndarray) : array of atoms
        adjs (numpy.ndarray): N by N 2-way array, or |E| by N by N 3-way array where |E| is the number of edgetypes.
        largest_atomic_number (int) : number of unique atom maximum index
        out_size (int): not used...

    Returns:
        super_node_x (numpy.ndarray); 1-way array, the supernode feature.
        len(super_node_x) will be 2 + 2 + MAX_ATOMIC_NUM*2 for 2-way adjs, 2 + 4*2 + MAX_ATOMIC_NUM*2 for 3-way adjs

    """
    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()
    E = numpy.sum(adjs.flatten())
    if E < 1.0:
        E = 1.0

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise MolFeatureExtractionError('out_size {} is smaller than number '
                                        'of atoms in mol {}'
                                        .format(out_size, N))

    # check the size of adjs
    if adjs.ndim == 2:
        super_node_x = numpy.zeros(2 + 2 + largest_atomic_number*2)
    elif adjs.ndim == 3:
        super_node_x = numpy.zeros(2 + 4*2 + largest_atomic_number*2)
    else:
        raise ValueError('adjs.ndim should be 2 or 3')
    # end if-else

    # number of nodes and edges
    super_node_x[0] = float(N)
    super_node_x[1] = float(E)

    # histogram of types of bins
    if adjs.ndim == 2:
        adjs_temp = numpy.reshape(adjs, (1, N*N))
        edge_type_histo = numpy.sum(adjs_temp, axis=1) / super_node_x[1]
        super_node_x[2] = numpy.max(adjs_temp, axis=1)
        super_node_x[3] = edge_type_histo

        idx_bias = 3
    elif adjs.ndim == 3:
        adjs_temp = numpy.reshape(adjs, (4, N*N))
        edge_type_histo = numpy.sum(adjs_temp, axis=1) / super_node_x[1]
        super_node_x[2:6] = numpy.max(adjs_temp, axis=1)
        super_node_x[6:10] = edge_type_histo

        idx_bias = 9
    # end if-else

    # histogram of types of nodes
    c = Counter(atom_array)
    keys = c.keys()
    values = c.values()
    for k, v in zip(keys, values):
        if k < largest_atomic_number:
            super_node_x[idx_bias+k] = 1.0
            super_node_x[idx_bias+largest_atomic_number+k] = float(v) / float(N)
        else:
            super_node_x[idx_bias+k] = 1.0
            super_node_x[idx_bias+largest_atomic_number+k] = float(v) / float(N)

    super_node_x = super_node_x.astype(numpy.float32)

    return super_node_x