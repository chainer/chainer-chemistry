import joblib
from logging import getLogger
import os
import shutil
import tarfile

import numpy
import pandas

from chainer.dataset import download

from chainer_chemistry.dataset.parsers.csv_file_parser import CSVFileParser
from chainer_chemistry.dataset.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from chainer_chemistry.dataset.splitters.base_splitter import BaseSplitter
from chainer_chemistry.dataset.splitters.scaffold_splitter import ScaffoldSplitter  # NOQA
from chainer_chemistry.dataset.splitters.deepchem_scaffold_splitter import DeepChemScaffoldSplitter  # NOQA
from chainer_chemistry.dataset.splitters import split_method_dict
from chainer_chemistry.datasets.molnet.molnet_config import molnet_default_config  # NOQA
from chainer_chemistry.datasets.molnet.pdbbind_time import get_pdbbind_time
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset

_root = 'pfnet/chainer/molnet'


def get_molnet_dataset(dataset_name, preprocessor=None, labels=None,
                       split=None, frac_train=.8, frac_valid=.1,
                       frac_test=.1, seed=777, return_smiles=False,
                       return_pdb_id=False, target_index=None, task_index=0,
                       **kwargs):
    """Downloads, caches and preprocess MoleculeNet dataset.

    Args:
        dataset_name (str): MoleculeNet dataset name. If you want to know the
            detail of MoleculeNet, please refer to
            `official site <http://moleculenet.ai/datasets-1>`_
            If you would like to know what dataset_name is available for
            chainer_chemistry, please refer to `molnet_config.py`.
        preprocessor (BasePreprocessor): Preprocessor.
            It should be chosen based on the network to be trained.
            If it is None, default `AtomicNumberPreprocessor` is used.
        labels (str or list): List of target labels.
        split (str or BaseSplitter or None): How to split dataset into train,
            validation and test. If `None`, this functions use the splitter
            that is recommended by MoleculeNet. Additionally You can use an
            instance of BaseSplitter or choose it from 'random', 'stratified'
            and 'scaffold'.
        return_smiles (bool): If set to ``True``,
            smiles array is also returned.
        return_pdb_id (bool): If set to ``True``,
            PDB ID array is also returned.
            This argument is only used when you select 'pdbbind_smiles'.
        target_index (list or None): target index list to partially extract
            dataset. If `None` (default), all examples are parsed.
        task_index (int): Target task index in dataset for stratification.
            (Stratified Splitter only)
    Returns (dict):
        Dictionary that contains dataset that is already split into train,
        valid and test dataset and 1-d numpy array with dtype=object(string)
        which is a vector of smiles for each example or `None`.

    """
    if dataset_name not in molnet_default_config:
        raise ValueError("We don't support {} dataset. Please choose from {}".
                         format(dataset_name,
                                list(molnet_default_config.keys())))

    if dataset_name == 'pdbbind_grid':
        pdbbind_subset = kwargs.get('pdbbind_subset')
        return get_pdbbind_grid(pdbbind_subset, split=split,
                                frac_train=frac_train, frac_valid=frac_valid,
                                frac_test=frac_test, task_index=task_index)
    if dataset_name == 'pdbbind_smiles':
        pdbbind_subset = kwargs.get('pdbbind_subset')
        time_list = kwargs.get('time_list')
        return get_pdbbind_smiles(pdbbind_subset, preprocessor=preprocessor,
                                  labels=labels, split=split,
                                  frac_train=frac_train, frac_valid=frac_valid,
                                  frac_test=frac_test,
                                  return_smiles=return_smiles,
                                  return_pdb_id=return_pdb_id,
                                  target_index=target_index,
                                  task_index=task_index,
                                  time_list=time_list)

    dataset_config = molnet_default_config[dataset_name]
    labels = labels or dataset_config['tasks']
    if isinstance(labels, str):
        labels = [labels, ]

    if preprocessor is None:
        preprocessor = AtomicNumberPreprocessor()

    if dataset_config['task_type'] == 'regression':
        def postprocess_label(label_list):
            return numpy.asarray(label_list, dtype=numpy.float32)
    elif dataset_config['task_type'] == 'classification':
        def postprocess_label(label_list):
            label_list = numpy.asarray(label_list)
            label_list[numpy.isnan(label_list)] = -1
            return label_list.astype(numpy.int32)

    parser = CSVFileParser(preprocessor, labels=labels,
                           smiles_col=dataset_config['smiles_columns'],
                           postprocess_label=postprocess_label)
    if dataset_config['dataset_type'] == 'one_file_csv':
        split = dataset_config['split'] if split is None else split

        if isinstance(split, str):
            splitter = split_method_dict[split]()
        elif isinstance(split, BaseSplitter):
            splitter = split
        else:
            raise TypeError("split must be None, str or instance of"
                            " BaseSplitter, but got {}".format(type(split)))

        if isinstance(splitter, (ScaffoldSplitter, DeepChemScaffoldSplitter)):
            get_smiles = True
        else:
            get_smiles = return_smiles

        result = parser.parse(get_molnet_filepath(dataset_name),
                              return_smiles=get_smiles,
                              target_index=target_index, **kwargs)
        dataset = result['dataset']
        smiles = result['smiles']
        train_ind, valid_ind, test_ind = \
            splitter.train_valid_test_split(dataset, smiles_list=smiles,
                                            task_index=task_index,
                                            frac_train=frac_train,
                                            frac_valid=frac_valid,
                                            frac_test=frac_test, **kwargs)
        train = NumpyTupleDataset(*dataset.features[train_ind])
        valid = NumpyTupleDataset(*dataset.features[valid_ind])
        test = NumpyTupleDataset(*dataset.features[test_ind])

        result['dataset'] = (train, valid, test)
        if return_smiles:
            train_smiles = smiles[train_ind]
            valid_smiles = smiles[valid_ind]
            test_smiles = smiles[test_ind]
            result['smiles'] = (train_smiles, valid_smiles, test_smiles)
        else:
            result['smiles'] = None
    elif dataset_config['dataset_type'] == 'separate_csv':
        result = {}
        train_result = parser.parse(get_molnet_filepath(dataset_name, 'train'),
                                    return_smiles=return_smiles,
                                    target_index=target_index)
        valid_result = parser.parse(get_molnet_filepath(dataset_name, 'valid'),
                                    return_smiles=return_smiles,
                                    target_index=target_index)
        test_result = parser.parse(get_molnet_filepath(dataset_name, 'test'),
                                   return_smiles=return_smiles,
                                   target_index=target_index)
        result['dataset'] = (train_result['dataset'], valid_result['dataset'],
                             test_result['dataset'])
        result['smiles'] = (train_result['smiles'], valid_result['smiles'],
                            test_result['smiles'])
    else:
        raise ValueError('dataset_type={} is not supported'
                         .format(dataset_config['dataset_type']))

    return result


def get_molnet_dataframe(dataset_name, pdbbind_subset=None):
    """Downloads, caches and get the dataframe of MoleculeNet dataset.

    Args:
        dataset_name (str): MoleculeNet dataset name. If you want to know the
            detail of MoleculeNet, please refer to
            `official site <http://moleculenet.ai/datasets-1>`_
            If you would like to know what dataset_name is available for
            chainer_chemistry, please refer to `molnet_config.py`.
        pdbbind_subset (str): PDBbind dataset subset name. If you want to know
            the detail of subset, please refer to `official site
            <http://www.pdbbind.org.cn/download/pdbbind_2017_intro.pdf>`
    Returns (pandas.DataFrame or tuple):
        DataFrame of dataset without any preprocessing. When the files of
        dataset are seprated, this function returns multiple DataFrame.

    """
    if dataset_name not in molnet_default_config:
        raise ValueError("We don't support {} dataset. Please choose from {}".
                         format(dataset_name,
                                list(molnet_default_config.keys())))
    if dataset_name == 'pdbbind_grid':
        raise ValueError('pdbbind_grid dataset is not supported. Please ',
                         'choose pdbbind_smiles dataset.')
    dataset_config = molnet_default_config[dataset_name]
    if dataset_config['dataset_type'] == 'one_file_csv':
        df = pandas.read_csv(get_molnet_filepath(
            dataset_name, pdbbind_subset=pdbbind_subset))
        return df
    elif dataset_config['dataset_type'] == 'separate_csv':
        train_df = pandas.read_csv(get_molnet_filepath(dataset_name, 'train'))
        valid_df = pandas.read_csv(get_molnet_filepath(dataset_name, 'valid'))
        test_df = pandas.read_csv(get_molnet_filepath(dataset_name, 'test'))
        return train_df, valid_df, test_df
    else:
        raise ValueError('dataset_type={} is not supported'
                         .format(dataset_config['dataset_type']))


def get_molnet_filepath(dataset_name, filetype='onefile',
                        download_if_not_exist=True, pdbbind_subset=None):
    """Construct a file path which stores MoleculeNet dataset.

    This method check whether the file exist or not, and downloaded it if
    necessary.

    Args:
        dataset_name (str): MoleculeNet dataset name.
        file_type (str): either 'onefile', 'train', 'valid', 'test'
        download_if_not_exist (bool): Download a file if it does not exist.

    Returns (str): filepath for specific MoleculeNet dataset

    """
    filetype_supported = ['onefile', 'train', 'valid', 'test']
    if filetype not in filetype_supported:
        raise ValueError("filetype {} not supported, please choose filetype "
                         "from {}".format(filetype, filetype_supported))
    if filetype == 'onefile':
        url_key = 'url'
    else:
        url_key = filetype + '_url'
    if dataset_name == 'pdbbind_smiles':
        file_url = molnet_default_config[dataset_name][url_key][pdbbind_subset]
    else:
        file_url = molnet_default_config[dataset_name][url_key]
    file_name = file_url.split('/')[-1]
    cache_path = _get_molnet_filepath(file_name)
    if not os.path.exists(cache_path):
        if download_if_not_exist:
            is_successful = download_dataset(file_url,
                                             save_filepath=cache_path)
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return cache_path


def _get_molnet_filepath(file_name):
    """Construct a filepath which stores MoleculeNet dataset in csv

    This method does not check if the file is already downloaded or not.

    Args:
        file_name (str): file name of MoleculeNet dataset

    Returns (str): filepath for one of MoleculeNet dataset

    """
    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, file_name)
    return cache_path


def download_dataset(dataset_url, save_filepath):
    """Download and caches MoleculeNet Dataset

    Args:
        dataset_url (str): URL of dataset
        save_filepath (str): filepath for dataset

    Returns (bool): If success downloading, returning `True`.

    """
    logger = getLogger(__name__)
    logger.warning('Downloading {} dataset, it takes time...'
                   .format(dataset_url.split('/')[-1]))
    download_file_path = download.cached_download(dataset_url)
    shutil.move(download_file_path, save_filepath)
    # pandas can load gzipped or tarball csv file
    return True


def get_pdbbind_smiles(pdbbind_subset, preprocessor=None, labels=None,
                       split=None, frac_train=.8, frac_valid=.1,
                       frac_test=.1, return_smiles=False, return_pdb_id=True,
                       target_index=None, task_index=0, time_list=None,
                       **kwargs):
    """Downloads, caches and preprocess PDBbind dataset.

    Args:
        pdbbind_subset (str): PDBbind dataset subset name. If you want to know
            the detail of subset, please refer to `official site
            <http://www.pdbbind.org.cn/download/pdbbind_2017_intro.pdf>`
        preprocessor (BasePreprocessor): Preprocessor.
            It should be chosen based on the network to be trained.
            If it is None, default `AtomicNumberPreprocessor` is used.
        labels (str or list): List of target labels.
        split (str or BaseSplitter or None): How to split dataset into train,
            validation and test. If `None`, this functions use the splitter
            that is recommended by MoleculeNet. Additionally You can use an
            instance of BaseSplitter or choose it from 'random', 'stratified'
            and 'scaffold'.
        return_smiles (bool): If set to ``True``,
            smiles array is also returned.
        return_pdb_id (bool): If set to ``True``,
            PDB ID array is also returned.
            This argument is only used when you select 'pdbbind_smiles'.
        target_index (list or None): target index list to partially extract
            dataset. If `None` (default), all examples are parsed.
        task_index (int): Target task index in dataset for stratification.
            (Stratified Splitter only)
    Returns (dict):
        Dictionary that contains dataset that is already split into train,
        valid and test dataset and 1-d numpy arrays with dtype=object(string)
        which are vectors of smiles and pdb_id for each example or `None`.

    """
    config = molnet_default_config['pdbbind_smiles']
    labels = labels or config['tasks']
    if isinstance(labels, str):
        labels = [labels, ]

    if preprocessor is None:
        preprocessor = AtomicNumberPreprocessor()

    def postprocess_label(label_list):
        return numpy.asarray(label_list, dtype=numpy.float32)

    parser = CSVFileParser(preprocessor, labels=labels,
                           smiles_col=config['smiles_columns'],
                           postprocess_label=postprocess_label)
    split = config['split'] if split is None else split
    if isinstance(split, str):
        splitter = split_method_dict[split]()
    elif isinstance(split, BaseSplitter):
        splitter = split
    else:
        raise TypeError("split must be None, str or instance of"
                        " BaseSplitter, but got {}".format(type(split)))

    result = parser.parse(get_molnet_filepath('pdbbind_smiles',
                                              pdbbind_subset=pdbbind_subset),
                          return_smiles=return_smiles,
                          return_is_successful=True,
                          target_index=target_index)
    dataset = result['dataset']
    smiles = result['smiles']
    is_successful = result['is_successful']

    if return_pdb_id:
        df = pandas.read_csv(
            get_molnet_filepath('pdbbind_smiles',
                                pdbbind_subset=pdbbind_subset))
        pdb_id = df['id'][is_successful]
    else:
        pdb_id = None

    train_ind, valid_ind, test_ind = \
        splitter.train_valid_test_split(dataset, time_list=time_list,
                                        smiles_list=smiles,
                                        task_index=task_index,
                                        frac_train=frac_train,
                                        frac_valid=frac_valid,
                                        frac_test=frac_test, **kwargs)
    train = NumpyTupleDataset(*dataset.features[train_ind])
    valid = NumpyTupleDataset(*dataset.features[valid_ind])
    test = NumpyTupleDataset(*dataset.features[test_ind])

    result['dataset'] = (train, valid, test)

    if return_smiles:
        train_smiles = smiles[train_ind]
        valid_smiles = smiles[valid_ind]
        test_smiles = smiles[test_ind]
        result['smiles'] = (train_smiles, valid_smiles, test_smiles)
    else:
        result['smiles'] = None

    if return_pdb_id:
        train_pdb_id = pdb_id[train_ind]
        valid_pdb_id = pdb_id[valid_ind]
        test_pdb_id = pdb_id[test_ind]
        result['pdb_id'] = (train_pdb_id, valid_pdb_id, test_pdb_id)
    else:
        result['pdb_id'] = None
    return result


def get_pdbbind_grid(pdbbind_subset, split=None, frac_train=.8, frac_valid=.1,
                     frac_test=.1, task_index=0, **kwargs):
    """Downloads, caches and grid-featurize PDBbind dataset.

    Args:
        pdbbind_subset (str): PDBbind dataset subset name. If you want to know
            the detail of subset, please refer to `official site
            <http://www.pdbbind.org.cn/download/pdbbind_2017_intro.pdf>`
        split (str or BaseSplitter or None): How to split dataset into train,
            validation and test. If `None`, this functions use the splitter
            that is recommended by MoleculeNet. Additionally You can use an
            instance of BaseSplitter or choose it from 'random', 'stratified'
            and 'scaffold'.
        task_index (int): Target task index in dataset for stratification.
            (Stratified Splitter only)
    Returns (dict):
        Dictionary that contains dataset that is already split into train,
        valid and test dataset and 1-d numpy arrays with dtype=object(string)
        which are vectors of smiles and pdb_id for each example or `None`.

    """
    result = {}
    dataset = get_grid_featurized_pdbbind_dataset(pdbbind_subset)
    if split is None:
        split = molnet_default_config['pdbbind_grid']['split']
    if isinstance(split, str):
        splitter = split_method_dict[split]()
    elif isinstance(split, BaseSplitter):
        splitter = split
    else:
        raise TypeError("split must be None, str, or instance of"
                        " BaseSplitter, but got {}".format(type(split)))
    time_list = get_pdbbind_time()
    train_ind, valid_ind, test_ind = \
        splitter.train_valid_test_split(dataset, time_list=time_list,
                                        smiles_list=None,
                                        task_index=task_index,
                                        frac_train=frac_train,
                                        frac_valid=frac_valid,
                                        frac_test=frac_test, **kwargs)
    train = NumpyTupleDataset(*dataset.features[train_ind])
    valid = NumpyTupleDataset(*dataset.features[valid_ind])
    test = NumpyTupleDataset(*dataset.features[test_ind])

    result['dataset'] = (train, valid, test)
    result['smiles'] = None
    return result


def get_grid_featurized_pdbbind_dataset(subset):
    """Downloads and caches grid featurized PDBBind dataset.

    Args:
        subset (str): subset name of PDBBind dataset.

    Returns (NumpyTupleDataset):
        grid featurized PDBBind dataset.

    """
    x_path, y_path = get_grid_featurized_pdbbind_filepath(subset)
    x = joblib.load(x_path).astype('i')
    y = joblib.load(y_path).astype('f')
    dataset = NumpyTupleDataset(x, y)
    return dataset


def get_grid_featurized_pdbbind_dirpath(subset, download_if_not_exist=True):
    """Construct a directory path which stores grid featurized PDBBind dataset.

    This method check whether the file exist or not, and downloaded it if
    necessary.

    Args:
        subset (str): subset name of PDBBind dataset.
        download_if_not_exist (bool): Download a file if it does not exist.

    Returns (str): directory path for specific subset of PDBBind dataset.

    """
    subset_supported = ['core', 'full', 'refined']
    if subset not in subset_supported:
        raise ValueError("subset {} not supported, please choose filetype "
                         "from {}".format(subset, subset_supported))
    file_url = \
        molnet_default_config['pdbbind_grid']['url'][subset]
    file_name = file_url.split('/')[-1]
    cache_path = _get_molnet_filepath(file_name)
    if not os.path.exists(cache_path):
        if download_if_not_exist:
            is_successful = download_dataset(file_url,
                                             save_filepath=cache_path)
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return cache_path


def get_grid_featurized_pdbbind_filepath(subset):
    """Construct a filepath which stores featurized PDBBind dataset in joblib

    This method does not check if the file is already downloaded or not.

    Args:
        subset (str): subset name of PDBBind dataset

    Returns:
        x_path (str): filepath for feature vectors
        y_path (str): filepath for -logKd/Ki

    """
    dirpath = get_grid_featurized_pdbbind_dirpath(subset=subset)
    savedir = '/'.join(dirpath.split('/')[:-1]) + '/'
    with tarfile.open(dirpath, 'r:gz') as tar:
        tar.extractall(savedir)
        x_path = savedir + subset + '_grid/shard-0-X.joblib'
        y_path = savedir + subset + '_grid/shard-0-y.joblib'
    return x_path, y_path
