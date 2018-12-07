import numpy
import matplotlib.pyplot as plt

from chainer_saliency.visualizer.base_visualizer import BaseVisualizer
from chainer_saliency.visualizer.common import abs_max_scaler


class TableVisualizer(BaseVisualizer):

    def visualize(self, saliency, feature_names=None, save_filepath=None,
                  num_visualize=-1, scaler=abs_max_scaler,
                  sort='descending', title='Feature Importance', color='b',
                  xlabel='Importance'):  # legend='',
        # --- type check ---
        assert saliency.ndim == 1
        num_total_feat = saliency.shape[0]
        if feature_names is not None:
            assert len(feature_names) == num_total_feat
        else:
            feature_names = numpy.arange(num_total_feat)

        if sort == 'none':
            indices = numpy.arange(num_total_feat)
        elif sort == 'ascending':
            indices = numpy.argsort(saliency)[::-1]
        elif sort == 'descending':
            indices = numpy.argsort(saliency)
        else:
            raise ValueError("[ERROR] Unexpected value sort={}".format(sort))

        saliency = saliency[indices]
        feature_names = numpy.asarray(feature_names)[indices]

        if scaler is not None:
            # Normalize to [-1, 1] or [0, 1]
            saliency = scaler(saliency)

        if num_visualize > 0:
            saliency = saliency[:num_visualize]
            if feature_names is not None:
                feature_names = feature_names[:num_visualize]
        else:
            num_visualize = num_total_feat

        plt.figure()
        plt.clf()
        if title is not None:
            plt.title(title)
        plt.barh(range(num_visualize), saliency, color=color, align='center')
        plt.yticks(range(num_visualize), feature_names)
        plt.xlabel(xlabel)
        if save_filepath:
            plt.savefig(save_filepath)
        else:
            plt.show()
