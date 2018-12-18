try:
    from chainer_chemistry.link_hooks import variable_monitor_link_hook  # NOQA

    from chainer_chemistry.link_hooks.variable_monitor_link_hook import VariableMonitorLinkHook  # NOQA
    is_link_hooks_available = True
except ImportError:
    import warnings
    warnings.warn('link_hooks failed to import, you need to upgrade chainer '
                  'version to use link_hooks feature')
    is_link_hooks_available = False
