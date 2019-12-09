from logging import getLogger

import numpy

import chainer
from chainer import cuda
from chainer.dataset.convert import concat_examples, _concat_arrays_with_padding  # NOQA
from chainer.iterators import SerialIterator

from chainer_chemistry.link_hooks import is_link_hooks_available
from tqdm import tqdm

if is_link_hooks_available:
    from chainer import LinkHook
    from chainer_chemistry.link_hooks import VariableMonitorLinkHook

_sampling_axis = 0


def _to_tuple(x):
    if not isinstance(x, tuple):
        x = (x,)
    return x


def _to_variable(x):
    if not isinstance(x, chainer.Variable):
        x = chainer.Variable(x)
    return x


def _extract_numpy(x):
    if isinstance(x, chainer.Variable):
        x = x.data
    return cuda.to_cpu(x)


def _concat(batch_list):
    try:
        return numpy.concatenate(batch_list)
    except Exception as e:  # NOQA
        # Thre is a case that each input has different shape,
        # we cannot concatenate into array in this case.

        elem_list = [elem for batch in batch_list for elem in batch]
        return _concat_arrays_with_padding(elem_list, padding=0)


def add_linkhook(linkhook, prefix='', logger=None):
    link_hooks = chainer._get_link_hooks()
    name = prefix + linkhook.name
    if name in link_hooks:
        logger = logger or getLogger(__name__)
        logger.warning('hook {} already exists, overwrite.'.format(name))
        pass  # skip this case...
        # raise KeyError('hook %s already exists' % name)
    link_hooks[name] = linkhook
    linkhook.added(None)
    return linkhook


def delete_linkhook(linkhook, prefix='', logger=None):
    name = prefix + linkhook.name
    link_hooks = chainer._get_link_hooks()
    if name not in link_hooks.keys():
        logger = logger or getLogger(__name__)
        logger.warning('linkhook {} is not registered'.format(name))
        return
    link_hooks[name].deleted(None)
    del link_hooks[name]


class BaseCalculator(object):

    """Base class for saliency calculator

    Use `compute`, `aggregate` method to calculate saliency.
    This base class supports to calculate SmoothGrad[1] and BayesGrad[2] of
    concrete subclass.

    See: Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viegas, and Martin
        Wattenberg. SmoothGrad: removing noise by adding noise.
        `arXiv:1706.03825 <https://arxiv.org/abs/1706.03825>`_

    See: Akita, Hirotaka and Nakago, Kosuke and Komatsu, Tomoki and Sugawara,
        Yohei and Maeda, Shin-ichi and Baba, Yukino and Kashima, Hisashi
        BayesGrad: Explaining Predictions of Graph Convolutional Networks
        `arXiv:1807.01985 <https://arxiv.org/abs/1807.01985>`_

    Args:
        model (chainer.Chain): target model to calculate saliency.
        target_extractor (VariableMonitorLinkHook or None):
            It determines `target_var`, target variable to calculate saliency.
            If `None`, first argument of input to the model is treated as
            `target_var`.
        output_extractor (VariableMonitorLinkHook or None):
            It determines `output_var`, output variable to calculate saliency.
            If `None`, output of the model is treated as `output_var`.
        device (int or None): device id to calculate saliency.
            If `None`, device id is inferred automatically from `model`.
        logger:
    """

    def __init__(self, model, target_extractor=None, output_extractor=None,
                 device=None, logger=None):
        self.model = model  # type: chainer.Chain
        if device is not None:
            self._device = device
        else:
            self._device = cuda.get_device_from_array(*model.params()).id
        self.target_extractor = target_extractor
        self.output_extractor = output_extractor
        self.logger = logger or getLogger(__name__)

    def compute(self, data, M=1, batchsize=16,
                converter=concat_examples, retain_inputs=False,
                preprocess_fn=None, postprocess_fn=None, train=False,
                noise_sampler=None, show_progress=True):
        """computes saliency_samples

        Args:
            data: dataset to calculate saliency
            M (int): sampling size. `M > 1` may be set with SmoothGrad or
                BayesGrad configuration. See `train` and `noise_sampler`
                description.
            batchsize (int): batch size
            converter (function): converter to make batch from `data`
            retain_inputs (bool): retain input flag
            preprocess_fn (function or None): preprocess function
            postprocess_fn (function or None): postprocess function
            train (bool): chainer.config.train flag. When the `model` contains
                `dropout` (or other stochastic) function, `train=True`
                 corresponds to calculate BayesGrad.
            noise_sampler: noise sampler class with `sample` method.
                If this is set, noise is added to `target_var`. It can be
                used to calculate SmoothGrad.
                If `None`, noise is not sampled.
            show_progress (bool): Show progress bar or not.

        Returns:
            saliency_samples (numpy.ndarray): M samples of saliency array.
                Its shape is (M,) + target_var.shape, i.e., sampling axis is
                added to the first axis.
        """
        saliency_list = []
        for _ in tqdm(range(M), disable=not show_progress):
            with chainer.using_config('train', train):
                saliency = self._forward(
                    data, batchsize=batchsize,
                    converter=converter,
                    retain_inputs=retain_inputs, preprocess_fn=preprocess_fn,
                    postprocess_fn=postprocess_fn, noise_sampler=noise_sampler)
            saliency_array = cuda.to_cpu(saliency)
            saliency_list.append(saliency_array)
        return numpy.stack(saliency_list, axis=_sampling_axis)

    def aggregate(self, saliency_arrays, method='raw', ch_axis=None):
        """Aggregate saliency samples into one saliency score.

        Args:
            saliency_arrays (numpy.ndarray): M samples of saliency array
                calculated by `compute` method.
            method (str): It supports following methods for aggregation.
                raw: simply take mean of samples.
                absolute: calc absolute mean of samples.
                square: calc squared mean of samples.
            ch_axis (int, tuple or None): channel axis. The ch_axis is
                considered as reduced axis for saliency calculation.

        Returns:
            saliency (numpy.ndarray): saliency score
        """
        if method == 'raw':
            h = saliency_arrays  # do nothing
        elif method == 'abs':
            h = numpy.abs(saliency_arrays)
        elif method == 'square':
            h = saliency_arrays ** 2
        else:
            raise ValueError("[ERROR] Unexpected value method={}"
                             .format(method))

        if ch_axis is not None:
            h = numpy.sum(h, axis=ch_axis)
        sampling_axis = _sampling_axis
        return numpy.mean(h, axis=sampling_axis)

    def _compute_core(self, *inputs):
        """Core computation routine

        Each concrete subclass should implement this method
        """
        raise NotImplementedError

    def get_target_var(self, inputs):
        if isinstance(self.target_extractor, VariableMonitorLinkHook):
            target_var = self.target_extractor.get_variable()
        else:
            if isinstance(inputs, tuple):
                target_var = inputs[0]
            else:
                target_var = inputs

        if target_var is None:
            self.logger.warning(
                'target_var is None. This may be caused because "model" is not'
                ' forwarded in advance or "model" does not implement "forward"'
                ' method and LinkHook is not triggered.')
        return target_var

    def get_output_var(self, outputs):
        if isinstance(self.output_extractor, VariableMonitorLinkHook):
            output_var = self.output_extractor.get_variable()
        else:
            output_var = outputs
        if output_var is None:
            self.logger.warning(
                'output_var is None. This may be caused because "model" is not'
                ' forwarded in advance or "model" does not implement "forward"'
                ' method and LinkHook is not triggered.')
        return output_var

    def _forward(self, data, batchsize=16,
                 converter=concat_examples, retain_inputs=False,
                 preprocess_fn=None, postprocess_fn=None, noise_sampler=None):
        """Forward data by iterating with batch

        Args:
            data: "train_x array" or "chainer dataset"
            batchsize (int): batch size
            converter (Callable): convert from `data` to `inputs`
            retain_inputs (bool): If True, this instance keeps inputs in
                `self.inputs` or not.
            preprocess_fn (Callable): Its input is numpy.ndarray or
                cupy.ndarray, it can return either Variable, cupy.ndarray or
                numpy.ndarray
            postprocess_fn (Callable): Its input argument is Variable,
                but this method may return either Variable, cupy.ndarray or
                numpy.ndarray.

        Returns (tuple or numpy.ndarray): forward result
        """
        input_list = None
        output_list = None
        it = SerialIterator(data, batch_size=batchsize, repeat=False,
                            shuffle=False)
        if isinstance(self.target_extractor, LinkHook):
            add_linkhook(self.target_extractor, prefix='/saliency/target/',
                         logger=self.logger)
        if isinstance(self.output_extractor, LinkHook):
            add_linkhook(self.output_extractor, prefix='/saliency/output/',
                         logger=self.logger)

        for batch in it:
            inputs = converter(batch, self._device)
            inputs = _to_tuple(inputs)

            if preprocess_fn:
                inputs = preprocess_fn(*inputs)
                inputs = _to_tuple(inputs)

            inputs = [_to_variable(x) for x in inputs]

            # --- Main saliency computation ----
            if noise_sampler is None:
                # VanillaGrad computation
                outputs = self._compute_core(*inputs)
            else:
                # SmoothGrad computation
                if self.target_extractor is None:
                    # inputs[0] is considered as "target_var"
                    noise = noise_sampler.sample(inputs[0].array)
                    inputs[0].array += noise
                    outputs = self._compute_core(*inputs)
                else:
                    # Add process to LinkHook
                    def add_noise(hook, args, target_var):
                        noise = noise_sampler.sample(target_var.array)
                        target_var.array += noise
                    self.target_extractor.add_process('/saliency/add_noise',
                                                      add_noise)
                    outputs = self._compute_core(*inputs)
                    self.target_extractor.delete_process('/saliency/add_noise')
            # --- Main saliency computation end ---

            # Init
            if retain_inputs:
                if input_list is None:
                    input_list = [[] for _ in range(len(inputs))]
                for j, input in enumerate(inputs):
                    input_list[j].append(cuda.to_cpu(input))

            if output_list is None:
                output_list = [[] for _ in range(len(outputs))]

            if postprocess_fn:
                outputs = postprocess_fn(*outputs)
                outputs = _to_tuple(outputs)
            for j, output in enumerate(outputs):
                output_list[j].append(_extract_numpy(output))

        if isinstance(self.target_extractor, LinkHook):
            delete_linkhook(self.target_extractor, prefix='/saliency/target/',
                            logger=self.logger)
        if isinstance(self.output_extractor, LinkHook):
            delete_linkhook(self.output_extractor, prefix='/saliency/output/',
                            logger=self.logger)

        if retain_inputs:
            self.inputs = [numpy.concatenate(
                in_array) for in_array in input_list]

        result = [_concat(output) for output in output_list]
        if len(result) == 1:
            return result[0]
        else:
            self.logger.error('return multiple result handling is not '
                              'implemented yet and not supported.')
            return result
