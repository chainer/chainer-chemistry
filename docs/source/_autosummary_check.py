import inspect
import os
import types

import chainer_chemistry.functions
import chainer_chemistry.links
import chainer_chemistry.models


def _is_rst_exists(entity):
    return os.path.exists('source/generated/{}.rst'.format(entity))


def check(app, exception):
    missing_entities = []

    missing_entities += [
        name for name in _list_chainer_functions()
        if not _is_rst_exists(name)]

    missing_entities += [
        name for name in _list_chainer_links()
        if not _is_rst_exists(name)]

    missing_entities += [
        name for name in _list_chainer_models()
        if not _is_rst_exists(name)]

    if len(missing_entities) != 0:
        app.warn('\n'.join([
            'Undocumented entities found.',
            '',
        ] + missing_entities))


def _list_chainer_functions():
    # List exported functions under chainer.functions.
    return ['chainer_chemistry.functions.{}'.format(name)
            for (name, func) in chainer_chemistry.functions.__dict__.items()
            if isinstance(func, types.FunctionType)]


def _list_chainer_links():
    # List exported classes under chainer.links.
    return ['chainer_chemistry.links.{}'.format(name)
            for (name, link) in chainer_chemistry.links.__dict__.items()
            if inspect.isclass(link)]


def _list_chainer_models():
    # List exported classes under chainer.links.
    return ['chainer_chemistry.models.{}'.format(name)
            for (name, model) in chainer_chemistry.models.__dict__.items()
            if inspect.isclass(model)]
