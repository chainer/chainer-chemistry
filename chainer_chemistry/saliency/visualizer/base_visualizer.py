class BaseVisualizer(object):

    """Base saliency visualizer"""

    def visualize(self, *args, **kwargs):
        """Main visualization routine

        Each concrete subclass should implement this method
        """
        raise NotImplementedError
