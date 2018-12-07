from abc import ABCMeta
from abc import abstractmethod

from future.utils import with_metaclass


class BaseVisualizer(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def visualize(self, *args, **kwargs):
        raise NotImplementedError
