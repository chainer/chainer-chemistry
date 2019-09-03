import numpy

class GaussianDistance:
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.

    Args:
        centers: (numpy.array)
        width: (float)
    """

    def __init__(self, centers=numpy.linspace(0, 4, 20), width=0.5):
        self.centers = centers
        self.width = width

    def convert(self, d):
        """
        expand distance vector d with given parameters

        Args:
            d: (value) distance

        Returns
            (vector) M dimension with M the length of centers
        """
        return numpy.exp(- (d - self.centers) ** 2 / self.width ** 2, dtype=numpy.float32)
