from abc import ABCMeta
from abc import abstractmethod
from future.utils import with_metaclass
import numpy

import chainer
from chainer import cuda, LinkHook
from chainer.dataset.convert import concat_examples, _concat_arrays_with_padding  # NOQA
from chainer.iterators import SerialIterator

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
    except Exception as e:
        # Thre is a case that each input has different shape,
        # we cannot concatenate into array in this case.

        elem_list = [elem for batch in batch_list for elem in batch]
        return _concat_arrays_with_padding(elem_list, padding=0)


def add_linkhook(linkhook, prefix=''):
    link_hooks = chainer._get_link_hooks()
    name = prefix + linkhook.name
    if name in link_hooks:
        print('[WARNING] hook {} already exists, overwrite.'.format(name))
        pass  # skip this case...
        # raise KeyError('hook %s already exists' % name)
    link_hooks[name] = linkhook
    linkhook.added(None)
    return linkhook


def delete_linkhook(linkhook, prefix=''):
    name = prefix + linkhook.name
    link_hooks = chainer._get_link_hooks()
    if name not in link_hooks.keys():
        print('[WARNING] linkhook {} is not registered'.format(name))
        return
    link_hooks[name].deleted(None)
    del link_hooks[name]


class BaseCalculator(with_metaclass(ABCMeta, object)):

    """Base class for saliency calculator

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
    """

    def __init__(self, model, target_extractor=None, output_extractor=None,
                 device=None):
        self.model = model  # type: chainer.Chain
        if device is not None:
            self._device = device
        else:
            self._device = cuda.get_device_from_array(*model.params()).id
        if target_extractor is None:
            # First argument of input to the `model` is default target_var
            self.target_extractor = VariableMonitorLinkHook(
                self.model, timing='pre')
        else:
            self.target_extractor = target_extractor
        self.output_extractor = output_extractor

    def compute(self, data, M=1, batchsize=16,
                converter=concat_examples, retain_inputs=False,
                preprocess_fn=None, postprocess_fn=None, train=False,
                noise_sampler=None, ):
        saliency_list = []
        for _ in range(M):
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

    @abstractmethod
    def _compute_core(self, *inputs):
        raise NotImplementedError

    def get_target_var(self):
        return self.target_extractor.get_variable()

    def get_output_var(self, outputs):
        if isinstance(self.output_extractor, LinkHook):
            return self.output_extractor.get_variable()
        else:
            return outputs

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
            add_linkhook(self.target_extractor, prefix='/saliency/target/')
        if isinstance(self.output_extractor, LinkHook):
            add_linkhook(self.output_extractor, prefix='/saliency/output/')

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
            delete_linkhook(self.target_extractor, prefix='/saliency/target/')
        if isinstance(self.output_extractor, LinkHook):
            delete_linkhook(self.output_extractor, prefix='/saliency/output/')

        if retain_inputs:
            self.inputs = [numpy.concatenate(
                in_array) for in_array in input_list]

        result = [_concat(output) for output in output_list]
        if len(result) == 1:
            return result[0]
        else:
            return result
