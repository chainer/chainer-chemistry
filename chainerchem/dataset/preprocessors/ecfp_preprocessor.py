from logging import getLogger

import numpy
from rdkit.Chem import rdMolDescriptors

from chainerchem.dataset.preprocessors.mol_preprocessor import MolFeatureExtractFailure  # NOQA
from chainerchem.dataset.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA


class ECFPPreprocessor(MolPreprocessor):

    def __init__(self, radius=2):
        super(ECFPPreprocessor, self).__init__()
        self.radius = radius

    def get_input_features(self, mol):
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                                self.radius)
        except Exception as e:
            logger = getLogger(__name__)
            logger.debug('exception caught at ECFPPreprocessor:', e)
            # Extracting feature failed
            raise MolFeatureExtractFailure
        # TODO(Nakago): Test it.
        return numpy.asarray(fp, numpy.float32)
