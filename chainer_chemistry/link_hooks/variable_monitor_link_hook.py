from collections import OrderedDict
from logging import getLogger

import chainer
from chainer.link_hook import _ForwardPostprocessCallbackArgs, _ForwardPreprocessCallbackArgs  # NOQA


def _default_extract_pre(hook, args):
    """Default extract_fn when `timing='pre`

    Args:
        hook (VariableMonitorLinkHook):
        args (_ForwardPreprocessCallbackArgs):

    Returns (chainer.Variable): First input variable to the link.
    """
    return args.args[0]


def _default_extract_post(hook, args):
    """Default extract_fn when `timing='post`

    Args:
        hook (VariableMonitorLinkHook):
        args (_ForwardPostprocessCallbackArgs):

    Returns (chainer.Variable): Output variable to the link.
    """
    return args.out


class VariableMonitorLinkHook(chainer.LinkHook):
    """Monitor Variable of specific link input/output

    Args:
        target_link (chainer.Link): target link to monitor variable.
        name (str): name of this link hook
        timing (str): timing of this link hook to monitor. 'pre' or 'post'.
            If 'pre', the input of `target_link` is monitored.
            If 'post', the output of `target_link` is monitored.
        extract_fn (callable): Specify custom method to extract target variable
            Default behavior is to extract first input when `timing='pre'`,
            or extract output when `timing='post'`.
            It takes `hook, args` as argument.
        logger:

    .. admonition:: Example

       >>> import numpy
       >>> from chainer import cuda, links, functions  # NOQA
       >>> from chainer_chemistry.link_hooks.variable_monitor_link_hook import VariableMonitorLinkHook  # NOQA

       >>> class DummyModel(chainer.Chain):
       >>>    def __init__(self):
       >>>        super(DummyModel, self).__init__()
       >>>        with self.init_scope():
       >>>            self.l1 = links.Linear(None, 1)
       >>>        self.h = None
       >>>
       >>>    def forward(self, x):
       >>>        h = self.l1(x)
       >>>        out = functions.sigmoid(h)
       >>>        return out

       >>> model = DummyModel()
       >>> hook = VariableMonitorLinkHook(model.l1, timing='post')
       >>> x = numpy.array([1, 2, 3])

       >>> # Example 1. `get_variable` of `target_link`.
       >>> with hook:
       >>>     out = model(x)
       >>> # You can extract `h`, which is output of `model.l1` as follows.
       >>> var_h = hook.get_variable()

       >>> # Example 2. `add_process` to override value of target variable.
       >>> def _process_zeros(hook, args, target_var):
       >>>     xp = cuda.get_array_module(target_var.array)
       >>>     target_var.array = xp.zeros(target_var.array.shape)
       >>> hook.add_process('_process_zeros', _process_zeros)
       >>> with hook:
       >>>     # During the forward, `h` is overriden to value 0.
       >>>     out = model(x)
       >>> # Remove _process_zeros method
       >>> hook.delete_process('_process_zeros')
    """

    def __init__(self, target_link, name='VariableMonitorLinkHook',
                 timing='post', extract_fn=None, logger=None):
        if not isinstance(target_link, chainer.Link):
            raise TypeError('target_link must be instance of chainer.Link!'
                            'actual {}'.format(type(target_link)))
        if timing not in ['pre', 'post']:
            raise ValueError(
                "Unexpected value timing={}, "
                "must be either pre or post"
                .format(timing))
        super(VariableMonitorLinkHook, self).__init__()
        self.target_link = target_link

        # This LinkHook maybe instantiated multiple times.
        # So it is allowed to change name by argument.
        self.name = name
        self.logger = logger or getLogger(__name__)

        if extract_fn is None:
            if timing == 'pre':
                extract_fn = _default_extract_pre
            elif timing == 'post':
                extract_fn = _default_extract_post
            else:
                raise ValueError("Unexpected value timing={}"
                                 .format(timing))
        self.extract_fn = extract_fn
        self.process_fns = OrderedDict()  # Additional process, if necessary

        self.timing = timing
        self.result = None

    def add_process(self, key, fn):
        """Add additional process for target variable

        Args:
            key (str): id for this process, you may remove added process by
                `delete_process` with this key.
            fn (callable): function which takes `hook, args, target_var` as
                arguments.
        """
        if not isinstance(key, str):
            raise TypeError('key must be str, actual {}'.format(type(key)))
        if not callable(fn):
            raise TypeError('fn must be callable')
        self.process_fns[key] = fn

    def delete_process(self, key):
        """Delete process added at `add_process`

        Args:
            key (str): id for the process, named at `add_process`.
        """
        if not isinstance(key, str):
            raise TypeError('key must be str, actual {}'.format(type(key)))
        if key in self.process_fns.keys():
            del self.process_fns[key]
        else:
            # Nothing to delete
            self.logger.warning('{} is not in process_fns, skip delete_process'
                                .format(key))

    def get_variable(self):
        """Get target variable, which is input or output of `target_link`.

        Returns (chainer.Variable): target variable
        """
        return self.result

    def forward_preprocess(self, args):
        if self.timing == 'pre' and args.link is self.target_link:
            self.result = self.extract_fn(self, args)
            if self.process_fns is not None:
                for key, fn in self.process_fns.items():
                    fn(self, args, self.result)

    def forward_postprocess(self, args):
        if self.timing == 'post' and args.link is self.target_link:
            self.result = self.extract_fn(self, args)
            if self.process_fns is not None:
                for key, fn in self.process_fns.items():
                    fn(self, args, self.result)
