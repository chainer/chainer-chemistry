import pickle

import chainer
from chainer import cuda
from chainer.dataset.convert import concat_examples
from chainer.iterators import SerialIterator
from chainer import link
import numpy


def _to_tuple(x):
    if not isinstance(x, tuple):
        x = (x,)
    return x


def _extract_numpy(x):
    if isinstance(x, chainer.Variable):
        x = x.data
    return cuda.to_cpu(x)


class BaseForwardModel(link.Chain):

    """A base model which supports forward functionality.

    It also supports `device` id management and pickle save/load functionality.

    Args:
        device (int): GPU device id of this model to be used.
            -1 indicates to use in CPU.

    Attributes:
        _device (int): Model's current device id

    """

    def __init__(self):
        super(BaseForwardModel, self).__init__()

        self.inputs = None
        self._device = None

    def get_device(self):
        return self._device

    def initialize(self, device=-1):
        """Initialization of the model.

        It must be executed **after** the link registration
        (often done by `with self.init_scope()` finished.

        Args:
            device (int): GPU device id of this model to be used.
            -1 indicates to use in CPU.

        """
        self.update_device(device=device)

    def update_device(self, device=-1):
        if self._device is None or self._device != device:
            # reset current state
            self.to_cpu()

            # update the model to specified device id
            self._device = device
            if device >= 0:
                chainer.cuda.get_device_from_id(device).use()
                self.to_gpu()  # Copy the model to the GPU

    def _forward(self, data, fn, batchsize=16,
                 converter=concat_examples, retain_inputs=False,
                 preprocess_fn=None, postprocess_fn=None):
        """Forward data by iterating with batch

        Args:
            data: "train_x array" or "chainer dataset"
            fn (Callable): Main function to forward. Its input argument is
                either Variable, cupy.ndarray or numpy.ndarray, and returns
                Variable.
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
        for batch in it:
            inputs = converter(batch, self._device)
            inputs = _to_tuple(inputs)

            if preprocess_fn:
                inputs = preprocess_fn(*inputs)
                inputs = _to_tuple(inputs)

            outputs = fn(*inputs)
            outputs = _to_tuple(outputs)

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

        if retain_inputs:
            self.inputs = [numpy.concatenate(
                in_array) for in_array in input_list]

        result = [numpy.concatenate(output) for output in output_list]
        if len(result) == 1:
            return result[0]
        else:
            return result

    def save_pickle(self, filepath, protocol=None):
        """Save the model to `filepath` as a pickle file

        This function send the parameters to CPU before saving the model so
        that the pickled file can be loaded with in CPU-only environment. 
        After the model is saved, it is sent back to the original device.

        Saved pickle file can be loaded with `load_pickle` static method.

        Note that the transportability of the saved file follows the
        specification of `pickle` module, namely serialized data depends on the
        specific class or attribute structure when saved. The file may not be
        loaded in different environment (version of python or dependent
        libraries), or after large refactoring of the pickled object class.
        If you want to avoid it, use `chainer.serializers.save_npz`
        method instead to save only model parameters.

    .. admonition:: Example

       >>> from chainer_chemistry.models import BaseForwardModel
       >>> class DummyForwardModel(BaseForwardModel):
       >>> 
       >>>     def __init__(self, device=-1):
       >>>         super(DummyForwardModel, self).__init__()
       >>>         with self.init_scope():
       >>>             self.l = chainer.links.Linear(3, 10)
       >>>         self.initialize(device)
       >>> 
       >>>     def __call__(self, x):
       >>>         return self.l(x)
       >>>
       >>> model = DummyForwardModel()
       >>> filepath = 'model.pkl'
       >>> model.save_pickle(filepath)  

        Args:
            filepath (str): file path of pickle file.
            protocol (int or None): protocol version used in `pickle`.
                Use 2 if you need python2/python3 compatibility.
                3 or higher is used for python3.
                Please refer the official document [1] for more details.
                [1]: https://docs.python.org/3.6/library/pickle.html#module-interface

        """  # NOQA
        current_device = self.get_device()

        # --- Move the model to CPU for saving ---
        self.update_device(-1)
        with open(filepath, mode='wb') as f:
            pickle.dump(self, f, protocol=protocol)

        # --- Revert the model to original device ---
        self.update_device(current_device)

    @staticmethod
    def load_pickle(filepath, device=-1):
        """Load the model from `filepath` of pickle file, and send to `device`

        The file saved by `save_pickle` method can be loaded, but it may fail
        to load when loading from different develop environment or after
        updating library version.
        See `save_pickle` method for the transportability of the saved file.

    .. admonition:: Example

       >>> from chainer_chemistry.models import BaseForwardModel
       >>> filepath = 'model.pkl'
       >>> # `load_pickle` is static method, call from Class to get an instance
       >>> model = BaseForwardModel.load_pickle(filepath)

        Args:
            filepath (str): file path of pickle file.
            device (int): GPU device id of this model to be used.
                -1 indicates to use in CPU.

        """
        with open(filepath, mode='rb') as f:
            model = pickle.load(f)

        if not isinstance(model, BaseForwardModel):
            raise TypeError('Unexpected type {}'.format(type(model)))

        # --- Revert the model to specified device ---
        model.initialize(device)
        return model
