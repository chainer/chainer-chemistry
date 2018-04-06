import inspect
import os
from abc import abstractmethod, ABCMeta
from collections import OrderedDict

import numpy
import sys
from chainer import serializers, Chain
from chainer.serializers import DictionarySerializer
import six
from six import with_metaclass

from chainer_chemistry.utils.json import save_json, load_json


def retain_args(f):
    def wrapper(self, *args, **kwargs):
        f(self, *args, **kwargs)

        # Check python version
        if sys.version_info[0] <= 2:
            # Assuming python 2.7 is used
            spec = inspect.getargspec(f)
            callargs = inspect.getcallargs(f, self, *args, **kwargs)

            args_dict = OrderedDict()
            for key in spec.args:
                # `spec.args` is list, it contains argument name in ordered way
                # `callargs` is dict, it has key and value in unordered way
                # here, we combine both to make ordered key, value pair as
                # `arg_dict`
                args_dict[key] = callargs[key]
        else:
            # Assuming python >= 3.5 is used
            sig = inspect.signature(f)
            ba = sig.bind(self, *args, **kwargs)
            ba.apply_defaults()  # available after python 3.5
            args_dict = ba.arguments
        self._init_args_dict = args_dict
    return wrapper


class BaseArgsSerializer(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def serialize(self, k, v, dirpath):
        raise NotImplementedError


class ArgsSerializer(BaseArgsSerializer):
    def serialize(self, k, v, dirpath):
        """

        Args:
            k (str): name of this variable
            v (object): value

        Returns (tuple): tuple with length=2, made of str and value.
            first value represents method, and second value represents serialized
            value

        """
        if isinstance(v, InitSerializableChain):
            init_args_list = v.serialize_args(dirpath, self)
            result = {'method': 'init_args_list', 'value': init_args_list}
        else:
            # TODO(nakago): check case where we cannot serialize directly
            result = {'method': 'raw', 'value': v}
        return result


class BaseArgsDeserializer(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def deserialize(self, item, dirpath):
        raise NotImplementedError


class ArgsDeserializer(BaseArgsDeserializer):

    def deserialize(self, item, dirpath):
        method = item['method']
        if method == 'raw':
            return item['value']
        elif method == 'init_args_list':
            full_classname = item['full_classname']
            klass = my_import(full_classname)

            # child_dirpath = os.path.join(
            #     dirpath, item["value"]).replace('\\', '/')
            # v = klass.load(child_dirpath)

            init_args_list = item['value']
            v = klass._instantiate(klass, init_args_list, dirpath,
                                   self)
            # model_filename = 'model.npz'
            # model_path = os.path.join(
            #     dirpath, model_filename).replace('\\', '/')
            # serializers.load_npz(model_path, v)
            return v
        else:
            raise ValueError('Unsupported method {}'.format(method))


def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_full_classname(o):
    return o.__class__.__module__ + '.' + o.__class__.__name__


class InitSerializableChain(Chain):

    _init_args_dict = None

    def save_init_args(self, dirpath, path):
        pass

    def serialize_args(self, dirpath, args_serializer):
        init_args_list = []
        skip_self = True
        for k, v in self._init_args_dict.items():
            if skip_self and k == 'self':
                # Do not save self key & value
                continue
            item = {
                'name': k,  # name of this variable
                'classname': v.__class__.__name__,  # name of this class
                'full_classname': get_full_classname(v),  # name of this class
            }
            item.update(args_serializer.serialize(k, v, dirpath))
            init_args_list.append(item)
        return init_args_list

    def save(self, dirpath, args_serializer=None):
        args_serializer = args_serializer or ArgsSerializer()
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        # init args
        init_args_list = self.serialize_args(dirpath, args_serializer)

        # model
        model_filename = 'model.npz'
        model_path = os.path.join(
            dirpath, model_filename).replace('\\', '/')
        serializers.save_npz(model_path, self)
        # model_config_path = os.path.join(
        #     dirpath, 'model_config.npz').replace('\\', '/')
        # self._save_model_params(model_config_path, self, init_args_list)

        config_save_path = os.path.join(
            dirpath, 'config.json').replace('\\', '/')
        config = {
            'init_args': init_args_list,
            'model_path': model_filename,
            'classname': self.__class__.__name__,
            'full_classname': get_full_classname(self),
        }
        save_json(config_save_path, config, ignore_error=True)

    def _save_model_params(self, file, obj, init_args_list, compression=True):
        if isinstance(file, six.string_types):
            with open(file, 'wb') as f:
                self._save_model_params(f, obj, compression)
            return

        s = DictionarySerializer()
        # s.save(obj)

        # --- serialize model params ---
        obj.serialize(s)
        # --- serialize aaa ---
        config_dict = {
            'init_args': init_args_list,
            # 'model_params': **s.target
            'model_params': s.target
        }
        if compression:
            # numpy.savez_compressed(file, **s.target)
            numpy.savez_compressed(file, **config_dict)
        else:
            # numpy.savez(file, **s.target)
            numpy.savez(file, **config_dict)

    @staticmethod
    def _instantiate(klass, init_args_list, dirpath, args_deserializer):
        init_args_dict = OrderedDict()
        for item in init_args_list:
            k = item['name']
            v = args_deserializer.deserialize(item, dirpath)
            init_args_dict[k] = v
        v = klass(**init_args_dict)
        return v

    # @classmethod
    @staticmethod
    def load(dirpath, args_deserializer=None):
        """
        
        Args:
            dirpath: 

        Returns (cls):

        """
        args_deserializer = args_deserializer or ArgsDeserializer()

        config_path = os.path.join(dirpath, 'config.json')
        config = load_json(config_path)
        model_path = config['model_path']
        init_args_list = config['init_args']
        full_classname = config['full_classname']

        klass = my_import(full_classname)
        v = klass._instantiate(klass, init_args_list, dirpath, args_deserializer)
        # v = cls._instantiate(cls, init_args_list, dirpath, args_deserializer)

        model_filename = 'model.npz'
        model_path = os.path.join(
            dirpath, model_filename).replace('\\', '/')
        serializers.load_npz(model_path, v)
        return v
