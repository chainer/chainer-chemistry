"""Config generator for Flex CI
Usage:
    $ python gen_config.py > config.pbtxt
"""

from __future__ import print_function
from collections import OrderedDict
import itertools


def test_config(python, chainer, target, chainerx):

    if chainerx:
        s_chainerx = '.chx'
    else:
        s_chainerx = ''
    key = 'chainerch.py{}.{}.{}{}'.format(python, chainer, target, s_chainerx)

    value = OrderedDict((
        ('requirement', OrderedDict((
            ('cpu', 4),
            ('memory', 16),
            ('disk', 10),
        ))),
        ('command', 'bash .flexci/pytest_script.sh'),
        ('environment_variables', [
            ('PYTHON', str(python)),
            ('CHAINER', chainer),
            ('CHAINERX', '1' if chainerx else '0'),
            ('GPU', '1' if target == 'gpu' else '0'),
        ]),
    ))

    if target == 'gpu':
        value['requirement']['gpu'] = 1

    return key, value


def main():
    configs = []

    for python, chainer in itertools.product(
            (37,), ('stable', 'latest', 'base')):
        for chainerx in (True, False):
            configs.append(test_config(python, chainer, 'cpu', chainerx))
            configs.append(test_config(python, chainer, 'gpu', chainerx))
    # small test in python 36
    configs.append(test_config(36, 'stable', 'gpu', False))

    print('# DO NOT MODIFY THIS FILE MANUALLY.')
    print('# USE gen_config.py INSTEAD.')
    print()

    dump_pbtxt('configs', configs)


def dump_pbtxt(key, value, level=0):
    indent = '  ' * level
    if isinstance(value, int):
        print('{}{}: {}'.format(indent, key, value))
    elif isinstance(value, str):
        print('{}{}: "{}"'.format(indent, key, value))
    elif isinstance(value, list):
        for k, v in value:
            print('{}{} {{'.format(indent, key))
            dump_pbtxt('key', k, level + 1)
            dump_pbtxt('value', v, level + 1)
            print('{}}}'.format(indent))
    elif isinstance(value, dict):
        print('{}{} {{'.format(indent, key))
        for k, v in value.items():
            dump_pbtxt(k, v, level + 1)
        print('{}}}'.format(indent))


if __name__ == '__main__':
    main()
