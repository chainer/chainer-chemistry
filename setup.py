from distutils.core import setup
import imp
import os

from setuptools import find_packages

setup_requires = []
install_requires = [
    'chainer>=2.0',
    'pandas',
    'scikit-learn',
    'scipy',
    'tqdm',
]


here = os.path.abspath(os.path.dirname(__file__))
__version__ = imp.load_source(
    '_version', os.path.join(here,
                             'chainer_chemistry', '_version.py')).__version__

setup(name='chainer-chemistry',
      version=__version__,
      description='Chainer Chemistry: A Library for Deep Learning in Biology\
      and Chemistry',
      author='Kosuke Nakago',
      author_email='nakago@preferred.jp',
      packages=find_packages(),
      license='MIT',
      url='http://chainer-chemistry.readthedocs.io/en/latest/index.html',
      setup_requires=setup_requires,
      install_requires=install_requires
      )
