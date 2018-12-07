import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from chainer import cuda

from chainer_saliency.visualizer.base_visualizer import BaseVisualizer
from chainer_saliency.visualizer.common import abs_max_scaler


class ImageVisualizer(BaseVisualizer):

    def visualize(self, saliency, image=None, save_filepath=None,
                  scaler=abs_max_scaler, title='Image saliency map',
                  cmap=cm.jet, alpha=0.5, show_colorbar=False):
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
                    image = numpy.transpose(image (1, 2, 0))
            elif image.ndim == 2:
                # (h, w) order
                pass
            else:
                raise ValueError("[ERROR] Unexpected value image.shape={}"
                                 .format(image.shape))
            if image.shape[:2] != saliency_image.shape[:2]:
                print('[WARNING] saliency and image height or width is different\n'
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
            plt.savefig(save_filepath)
        else:
            plt.show()
