from logging import getLogger

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy

from chainer import cuda

from chainer_chemistry.saliency.visualizer.base_visualizer import BaseVisualizer  # NOQA
from chainer_chemistry.saliency.visualizer.visualizer_utils import abs_max_scaler  # NOQA


class ImageVisualizer(BaseVisualizer):

    """Saliency visualizer for image data

    Args:
        logger:
    """

    def __init__(self, logger=None):
        self.logger = logger or getLogger(__name__)

    def visualize(self, saliency, image=None, save_filepath=None,
                  scaler=abs_max_scaler, title='Image saliency map',
                  cmap=cm.jet, alpha=0.5, show_colorbar=False,
                  bbox_inches='tight'):
        """Visualize or save `saliency` of image.

        Args:
            saliency (numpy.ndarray): Saliency array. Must be either
                2-dim (h, w) or 3-dim (ch, h, w).
            image (numpy.ndarray or PIL.Image or None): If set, image is drawn
                in background, and saliency is shown in foreground.
                If numpy array, must be in the order of 2-dim (h, w) or
                3-dim (ch, h, w).
            save_filepath (str or None): If specified, file is saved to path.
            scaler (callable): function which takes `x` as input and outputs
                scaled `x`, for plotting.
            title (str or None): title of plot
            cmap: color map used to plot saliency
            alpha (float): alpha value of fore ground saliency. This option is
                used only when `image` is set.
            show_colorbar (bool): show colorbar in plot or not.
            bbox_inches (str or Bbox or None): used for `plt.savefig` option.
        """
        # --- type check ---
        if saliency.ndim == 3:
            # (ch, h, w) -> (h, w, ch)
            saliency = cuda.to_cpu(saliency)
            saliency_image = numpy.transpose(saliency, (1, 2, 0))
        elif saliency.ndim == 2:
            # (h, w)
            saliency_image = saliency
        else:
            raise ValueError("[ERROR] Unexpected value saliency.shape={}"
                             .format(saliency.shape))

        if image is not None:
            # If `image` is PIL Image, convert to numpy array
            image = numpy.asarray(image)
            if image.ndim == 3:
                # Convert to (h, w, ch) order
                if image.shape[0] == 3 or image.shape[0] == 4:
                    # Assume (ch, h, w) order -> (h, w, ch)
                    image = numpy.transpose(image, (1, 2, 0))
            elif image.ndim == 2:
                # (h, w) order
                pass
            else:
                raise ValueError("[ERROR] Unexpected value image.shape={}"
                                 .format(image.shape))
            if image.shape[:2] != saliency_image.shape[:2]:
                self.logger.warning(
                    'saliency and image height or width is different\n'
                    'saliency_image.shape {}, image.shape [}'
                    .format(saliency_image.shape, image.shape))

        # Normalize to [-1, 1] or [0, 1]
        if scaler is not None:
            saliency_image = scaler(saliency_image)

        fig = plt.figure()
        plt.clf()
        if title is not None:
            plt.title(title)

        if image is None:
            # Only show saliency image, not set alpha
            im = plt.imshow(saliency_image, cmap=cmap)
        else:
            # Show original image, and overlay saliency image with alpha
            plt.imshow(image)
            im = plt.imshow(saliency_image, alpha=alpha, cmap=cmap)

        if show_colorbar:
            fig.colorbar(im)
        if save_filepath:
            plt.savefig(save_filepath, bbox_inches=bbox_inches)
        else:
            plt.show()
