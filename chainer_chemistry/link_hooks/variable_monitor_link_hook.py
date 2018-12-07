from collections import OrderedDict

import chainer


def _default_extract_pre(hook, args):
    return args.args[0]


def _default_extract_post(hook, args):
    return args.out


class VariableMonitorLinkHook(chainer.LinkHook):
    """Monitor Variable of specific link input/output"""

    def __init__(self, target_link, name='VariableMonitorLinkHook',
                 timing='post', extract_fn=None):
        if not isinstance(target_link, chainer.Link):
            raise TypeError('target_link must be instance of chainer.Link!'
                            'actual {}'.format(type(target_link)))
        if timing not in ['pre', 'post']:
            raise ValueError(
                "[ERROR] Unexpected value timing={}, "
                "must be either pre or post"
                .format(timing))
        super(VariableMonitorLinkHook, self).__init__()
        self.target_link = target_link

        # This LinkHook maybe instantiated multiple times.
        # So it is allowed to change name by argument.
        self.name = name

        if extract_fn is None:
            if timing == 'pre':
                extract_fn = _default_extract_pre
            elif timing == 'post':
                extract_fn = _default_extract_post
            else:
                raise ValueError("[ERROR] Unexpected value timing={}"
                                 .format(timing))
        self.extract_fn = extract_fn
        self.process_fns = OrderedDict()  # Additional process, if necessary

        self.timing = timing
        self.result = None

    def add_process(self, key, fn):
        if not isinstance(key, str):
            raise TypeError('key must be str, actual {}'.format(type(key)))
        if not callable(fn):
            raise TypeError('fn must be callable')
        self.process_fns[key] = fn

    def delete_process(self, key):
        del self.process_fns[key]

    def forward_preprocess(self, args):
        if self.timing == 'pre' and args.link is self.target_link:
            self.result = self.extract_fn(self, args)
            if self.process_fns is not None:
                for key, fn in self.process_fns.items():
                    fn(self, args, self.result)

    def forward_postprocess(self, args):
        if self.timing == 'post' and args.link is self.target_link:
            print('matched at {}'.format(args.link.name))
            self.result = self.extract_fn(self, args)
            if self.process_fns is not None:
                for key, fn in self.process_fns.items():
                    fn(self, args, self.result)

    def get_variable(self):
        return self.result
