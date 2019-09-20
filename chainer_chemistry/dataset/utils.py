import numpy


class GaussianDistance:
    """Expand distance with Gaussian basis sit at centers and with width 0.5.

    Args:
        centers: (numpy.ndarray)
        width: (float)
    """

    def __init__(self, centers=numpy.linspace(0, 4, 20), width=0.5):
        self.centers = centers
        self.width = width

    def expand(self, d):
        """
        expand distance value d with given parameters

        Args:
            d (float): distance

        Returns
            expanded_distance (numpy.1darray):
                M dimension with M the length of centers
        """
        return numpy.exp(-(d-self.centers)**2 / self.width**2,
                         dtype=numpy.float32)

    def expand_from_distances(self, distances):
        """
        expand distance vectors with given parameters
        original implemantation is below
        https://github.com/txie-93/cgcnn/blob/fdcd7eec8771e223e60e1b0abf7e6c7bc7d006bf/cgcnn/data.py#L152

        Args:
            distances (numpy.ndarray): 1 dimensional array.

        Returns
            expanded_distances (numpy.ndarray): 2 dimensional array.
                First axis size is the number of distance,
                Second axis size is M dimension with M the length of centers
        """
        return numpy.exp(-(distances[..., numpy.newaxis] - self.centers)**2
                         / self.width**2, dtype=numpy.float32)
