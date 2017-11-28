"""
Preprocessor supports feature extraction for each model (network)

- atom
  - atom_list (need to converted by EmbedID)
  - MorganFingerprint
  - ECFP
    etc

^ edge
  - adj matrix
  - adj matrix by degree
  - adj matrix by edge type
  - adj matrix by edge type and distance
    etc
"""


class BasePreprocessor(object):
    """Base class for preprocessor"""
    def __init__(self):
        pass

    def process(self, filepath):
        pass
