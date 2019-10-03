from logging import getLogger
import traceback

import numpy
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

from chainer_chemistry.dataset.preprocessors.common \
    import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor \
    import MolPreprocessor


def construct_distance_matrix(mol, out_size=-1, contain_Hs=False):
    """Construct distance matrix

    Args:
        mol (Chem.Mol):
        out_size (int):
        contain_Hs (bool):

    Returns (numpy.ndarray): 2 dimensional array which represents distance
        between atoms

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

    if contain_Hs:
        mol2 = mol
    else:
        mol2 = AllChem.AddHs(mol)

    conf_id = AllChem.EmbedMolecule(mol2)
    if not contain_Hs:
        mol2 = AllChem.RemoveHs(mol2)

    try:
        dist_matrix = rdmolops.Get3DDistanceMatrix(mol2, confId=conf_id)
    except ValueError as e:
        logger = getLogger(__name__)
        logger.info('construct_distance_matrix failed, type: {}, {}'
                    .format(type(e).__name__, e.args))
        logger.debug(traceback.format_exc())
        raise MolFeatureExtractionError

    if size > N:
        dists = numpy.zeros((size, size), dtype=numpy.float32)
        a0, a1 = dist_matrix.shape
        dists[:a0, :a1] = dist_matrix
    else:
        dists = dist_matrix
    return dists.astype(numpy.float32)


class SchNetPreprocessor(MolPreprocessor):
    """SchNet Preprocessor

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
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.

    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False,
                 kekulize=False):
        super(SchNetPreprocessor, self).__init__(
            add_Hs=add_Hs, kekulize=kekulize)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
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
        dist_array = construct_distance_matrix(mol, out_size=self.out_size,
                                               contain_Hs=self.add_Hs)
        return atom_array, dist_array
