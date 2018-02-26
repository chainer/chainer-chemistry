"""Common preprocess method is gethered in this file"""

import numpy
from rdkit.Chem import rdmolops


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
        numpy.ndarray: the adjcent matrix of the input molecule.
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
        raise ValueError('`out_size` (={}) must be negative or '
                         'larger than or equal to the number '
                         'of atoms in the input molecules (={})'
                         '.'.format(out_size, s0))
    return adj_array
