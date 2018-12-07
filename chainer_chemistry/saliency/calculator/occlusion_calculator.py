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

    def __init__(self, model, target_extractor=None, output_extractor=None,
                 eval_fun=None, device=None,
                 enable_backprop=False, size=1, slide_axis=(2, 3)):
        super(OcclusionCalculator, self).__init__(
            model, target_extractor=target_extractor,
            output_extractor=output_extractor, device=device)

        self.eval_fun = eval_fun
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
        # original_score = _extract_score(original_result)
        target_var = self.target_extractor.get_variable()
        original_score = self.output_extractor.get_variable()

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
            return index

        end_list = [target_var.data.shape[axis] - size for axis, size
                    in zip(self.slide_axis, self.size)]

        for start in itertools.product(*[six.moves.range(end) for end in end_list]):
            occlude_index = _extract_index(self.slide_axis, self.size, start)

            def mask_target_var(hook, args, target_var):
                target_var.array[occlude_index] = occlusion_window

            # self.target_extractor.set_process(mask_target_var)
            self.target_extractor.add_process(
                '/saliency/mask_target_var', mask_target_var)
            with chainer.using_config('enable_backprop', self.enable_backprop):
                occluded_result = self.eval_fun(*inputs)
            occluded_score = self.output_extractor.get_variable()
            self.target_extractor.delete_process(
                '/saliency/mask_target_var', mask_target_var)
            # occluded_score = _extract_score(occluded_result)
            score_diff_var = original_score - occluded_score
            # TODO: expand_dim dynamically
            score_diff = xp.broadcast_to(score_diff_var.data[:, :, None],
                                         occlusion_window_shape)
            occlusion_scores[occlude_index] += score_diff
        outputs = (occlusion_scores,)
        return outputs
