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
        assert isinstance(target_link, chainer.Link)
        assert timing in ['pre', 'post']
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
                raise ValueError("[ERROR] Unexpected value timing={}".format(timing))
        self.extract_fn = extract_fn
        self.process_fns = OrderedDict()  # Additional process, if necessary

        self.timing = timing
        self.result = None

    def add_process(self, key, fn):
        assert isinstance(key, str)
        assert callable(fn)
        # self.process.update({key: fn})
        self.process_fns[key] = fn

    def delete_process(self, key):
        del self.process_fns[key]

    def forward_preprocess(self, args):
        if self.timing == 'pre' and args.link is self.target_link:
            # print('[DEBUG] matched at {}'.format(args.link.name))
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
