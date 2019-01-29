# " -*- coding: utf-8 -*-"
#-------------------------------------------------------------------------------
# Name:        gin_gwm_preprocessor.py
# Purpose:     Implementation of the data preprocessor for Graph Isomorphism Networks (GTN)
#              equipeed with GWM
#
#              inputs:
#
#              outputs:
#
# Author:      Katsuhiko Ishiguro <ishiguro@preferred.jp>
# License:     All rights reserved unless specified.
# Created:     13/12/18 (DD/MM/YY)
# Last update: 13/12/18 (DD/MM/YY)
#-------------------------------------------------------------------------------

import numpy as np

from collections import Counter

from chainer import functions as F


from chainer_chemistry.config import MAX_ATOMIC_NUM
from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor


def construct_discrete_edge_matrix(mol, out_size=-1):
    """construct discrete edge matrix

    :param mol (Chem.Mol):
    :param out_size (int):

    :return (numpy.ndarray):

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

    adjs = np.zeros((size, size), dtype=np.float32)
    for i in range(N):

        for j in range(N):
            bond = mol.GetBondBetweenAtoms(i, j)  # type: Chem.Bond
            if bond is not None:
                bond_type = str(bond.GetBondType())
                if bond_type == 'SINGLE':
                    adjs[i, j] = 1.0
                elif bond_type == 'DOUBLE':
                    adjs[i, j] = 1.0
                elif bond_type == 'TRIPLE':
                    adjs[i, j] = 1.0
                elif bond_type == 'AROMATIC':
                    adjs[i, j] = 1.0
                else:
                    raise ValueError("[ERROR] Unknown bond type {}"
                                     .format(bond_type))
    return adjs

def construct_supernode_feature(mol, atom_array, adjs, out_size=-1):
    """
    Construct the input feature x' for the super node

    :param mol:  Chem.mol instance, molecular state
    :param atom_array  numpy.array? set of node-features
    :param adjs:  adjacency matrix
    :param out_size: integer, the maximum size of the output feature
    :return: np.int32 numpy array, the output feature
    """

    largest_atomic_number = MAX_ATOMIC_NUM

    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()
    E = np.sum(adjs.flatten())
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

    super_node_x = np.zeros(2 + 4*2 + largest_atomic_number*2)

    # number of nodes and edges
    super_node_x[0] = float(N)
    super_node_x[1] = float(E)

    # histogram of types of bins
    adjs_temp = np.reshape(adjs, (1, N*N))
    edge_type_histo = np.sum(adjs_temp, axis=1) / super_node_x[1]
    super_node_x[2:6] = np.max(adjs_temp, axis=1)
    super_node_x[6:10] = edge_type_histo

    # histogram of types of nodes
    c = Counter(atom_array)
    keys = c.keys()
    values = c.values()
    for k, v in zip(keys, values):
        if k < largest_atomic_number:
            super_node_x[9+k] = 1.0
            super_node_x[9+largest_atomic_number+k] = float(v) / float(N)
        else:
            super_node_x[9+k] = 1.0
            super_node_x[9+largest_atomic_number+k] = float(v) / float(N)

    super_node_x = super_node_x.astype(np.float32)

    return super_node_x

class GIN_GWMPreprocessor(MolPreprocessor):
    """GIN + GWM Preprocessor

    """

    def __init__(self, max_atoms=-1, out_size=-1, out_size_super=-1, add_Hs=False):
        """
        initialize the GTN Preprocessor.

        :param max_atoms: integer, Max number of atoms for each molecule,
            if the number of atoms is more than this value,
            this data is simply ignored.
            Setting negative value indicates no limit for max atoms.
        :param out_size: integer, It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        :param out_size_super: integer, indicate the length of the super node feature.
        :param add_Hs: boolean. if true, add Hydrogens explicitly.
        """
        super(GIN_GWMPreprocessor, self).__init__(add_Hs=add_Hs)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size
        self.out_size_super = out_size_super

    def get_input_features(self, mol):
        """get input features

        :param mol: Chem.Mol, the molecule representation

        :return atom_array: num_sample by node_feature_dim numpy.ndarray, array of feature vectors of atom (local) nodes.
        :return adj_array: num_sample by bond-type by num_node by num_node numpy.ndarray, array of multi-relation (bonds) adjacency matrix of atom (local) nodes.
        :return super_node_x: num_sample by super-node_feature_dim numpy.ndarray, array of feature vectors of the super node.

        """

        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        adj_array = construct_discrete_edge_matrix(mol, out_size=self.out_size)
        super_node_x = construct_supernode_feature(mol, atom_array, adj_array, out_size=self.out_size_super)
        return atom_array, adj_array, super_node_x
