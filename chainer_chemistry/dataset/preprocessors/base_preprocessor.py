"""
Preprocessor supports feature extraction for each model (network)
"""


class BasePreprocessor(object):
    """Base class for preprocessor"""

    def __init__(self):
        pass

    def process(self, filepath):
        pass
