from distutils.core import setup
import os

from setuptools import find_packages

setup_requires = []
install_requires = [
    'chainer >=7.0.0',
    'joblib',
    'matplotlib',
    'pandas',
    'scikit-learn',
    'scipy',
    'tqdm',
    'typing'
]


here = os.path.abspath(os.path.dirname(__file__))
# Get __version__ variable
exec(open(os.path.join(here, 'chainer_chemistry', '_version.py')).read())

setup(name='chainer-chemistry',
      version=__version__,  # NOQA
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
