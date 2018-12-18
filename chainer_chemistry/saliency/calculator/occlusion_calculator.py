import itertools
import six

import chainer
from chainer import cuda

from chainer_chemistry.saliency.calculator.base_calculator import BaseCalculator  # NOQA


def _to_tuple(x):
    if isinstance(x, int):
        x = (x,)
    elif isinstance(x, (list, tuple)):
        x = tuple(x)
    else:
        raise TypeError('Unexpected type {}'.format(type(x)))
    return x


class OcclusionCalculator(BaseCalculator):

    """Occlusion saliency calculator

    Use `compute`, `aggregate` method to calculate saliency.

    See: Matthew D Zeiler and Rob Fergus (2014).
        Visualizing and understanding convolutional networks.
        In European conference on computer vision, pp. 818-833. Springer.

    Args:
        model (chainer.Chain): target model to calculate saliency.
        target_extractor (VariableMonitorLinkHook or None):
            It determines `target_var`, target variable to calculate saliency.
            If `None`, first argument of input to the model is treated as
            `target_var`.
        output_extractor (VariableMonitorLinkHook or None):
            It determines `output_var`, output variable to calculate saliency.
            If `None`, output of the model is treated as `output_var`.
        eval_fun (callable): If
        enable_backprop (bool): chainer.config.enable_backprop option.
        size (int or tuple): occlusion window size.
            If `int`, window has same size along `slide_axis`.
            If `tuple`, its length must be same with `slide_axis`.
        slide_axis (int or tuple): slide axis which occlusion window moves.
        device (int or None): device id to calculate saliency.
            If `None`, device id is inferred automatically from `model`.
    """
    def __init__(self, model, target_extractor=None, output_extractor=None,
                 eval_fun=None, device=None,
                 enable_backprop=False, size=1, slide_axis=(2, 3)):
        super(OcclusionCalculator, self).__init__(
            model, target_extractor=target_extractor,
            output_extractor=output_extractor, device=device)

        self.eval_fun = eval_fun or model.__call__
        self.enable_backprop = enable_backprop
        self.slide_axis = _to_tuple(slide_axis)
        size = _to_tuple(size)
        if len(self.slide_axis) != size:
            size = size * len(self.slide_axis)
        self.size = size

    def _compute_core(self, *inputs):
        # Usually, backward() is not necessary for calculating occlusion
        with chainer.using_config('enable_backprop', self.enable_backprop):
            original_result = self.eval_fun(*inputs)
        target_var = self.get_target_var(inputs)
        original_target_array = target_var.array.copy()
        original_score = self.get_output_var(original_result)

        xp = cuda.get_array_module(target_var.array)
        value = 0.

        # fill with `value`
        target_dim = target_var.ndim
        batch_size = target_var.shape[0]
        occlusion_window_shape = [1] * target_dim
        occlusion_window_shape[0] = batch_size
        for axis, size in zip(self.slide_axis, self.size):
            occlusion_window_shape[axis] = size
        occlusion_scores_shape = [1] * target_dim
        occlusion_scores_shape[0] = batch_size
        for axis, size in zip(self.slide_axis, self.size):
            occlusion_scores_shape[axis] = target_var.shape[axis]
        occlusion_window = xp.ones(occlusion_window_shape,
                                   dtype=target_var.dtype) * value
        occlusion_scores = xp.zeros(occlusion_scores_shape, dtype=xp.float32)

        def _extract_index(slide_axis, size, start_indices):
            colon = slice(None)
            index = [colon] * target_dim
            for axis, size, start in zip(slide_axis, size, start_indices):
                index[axis] = slice(start, start + size, 1)
            return tuple(index)

        end_list = [target_var.data.shape[axis] - size + 1 for axis, size
                    in zip(self.slide_axis, self.size)]

        for start in itertools.product(*[six.moves.range(end)
                                         for end in end_list]):
            occlude_index = _extract_index(self.slide_axis, self.size, start)

            if self.target_extractor is None:
                inputs[0].array = original_target_array.copy()
                inputs[0].array[occlude_index] = occlusion_window
                with chainer.using_config('enable_backprop',
                                          self.enable_backprop):
                    occluded_result = self.eval_fun(*inputs)
            else:
                def mask_target_var(hook, args, _target_var):
                    _target_var.array = original_target_array.copy()
                    _target_var.array[occlude_index] = occlusion_window

                self.target_extractor.add_process(
                    '/saliency/mask_target_var', mask_target_var)
                with chainer.using_config('enable_backprop',
                                          self.enable_backprop):
                    occluded_result = self.eval_fun(*inputs)
                self.target_extractor.delete_process(
                    '/saliency/mask_target_var')

            occluded_score = self.get_output_var(occluded_result)
            score_diff_var = original_score - occluded_score  # (bs, 1)
            # expand_dim for ch_axis
            score_diff = xp.reshape(score_diff_var.array,
                                    occlusion_window_shape)
            occlusion_scores[occlude_index] += score_diff
        outputs = (occlusion_scores,)
        return outputs
