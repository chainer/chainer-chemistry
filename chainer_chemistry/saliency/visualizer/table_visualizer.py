import matplotlib.pyplot as plt
import numpy

from chainer_chemistry.saliency.visualizer.base_visualizer import BaseVisualizer  # NOQA
from chainer_chemistry.saliency.visualizer.visualizer_utils import abs_max_scaler  # NOQA


class TableVisualizer(BaseVisualizer):

    """Saliency visualizer for table data"""

    def visualize(self, saliency, feature_names=None, save_filepath=None,
                  num_visualize=-1, scaler=abs_max_scaler,
                  sort='descending', title='Feature Importance', color='b',
                  xlabel='Importance', bbox_inches='tight'):
        """Visualize or save `saliency` in bar plot.

        Args:
            saliency (numpy.ndarray): 1-dim saliency array (num_feature,)
            feature_names (list or numpy.ndarray): Feature names of `saliency`
            save_filepath (str or None): If specified, file is saved to path.
            num_visualize (int): If positive value is set, only plot specified
               number of features.
            scaler (callable): function which takes `x` as input and outputs
                scaled `x`, for plotting.
            sort (str): Below sort options are supported.
                none: not sort
                ascending: plot in ascending order
                descending: plot in descending order
            title (str or None): title of plot
            color (str): color of bar in plot
            xlabel (str): x label legend
            bbox_inches (str or Bbox or None): used for `plt.savefig` option.

        """
        # --- type check ---
        if saliency.ndim != 1:
            raise ValueError("[ERROR] Unexpected value saliency.shape={}"
                             .format(saliency.shape))

        num_total_feat = saliency.shape[0]
        if feature_names is not None:
            # type check
            if len(feature_names) != num_total_feat:
                raise ValueError(
                    "feature_names={} must have same length with `saliency`"
                    .format(feature_names))
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
            plt.savefig(save_filepath, bbox_inches=bbox_inches)
        else:
            plt.show()
