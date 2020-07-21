class BaseParser(object):
    def __init__(self):
        pass


class BaseFileParser(BaseParser):
    """base class for file parser"""

    def __init__(self, preprocessor):
        super(BaseFileParser, self).__init__()
        self.preprocessor = preprocessor

    def parse(self, filepath):
        raise NotImplementedError
