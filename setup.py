from distutils.core import setup

from setuptools import find_packages

setup_requires = []
install_requires = [
    'chainer>=2.0',
    'pandas',
    'tqdm',
]

setup(name='chainerchem',
      version='0.0.1',
      description='Deep learning library for chemistry.',
      packages=find_packages(),
      license='MIT',
      setup_requires=setup_requires,
      install_requires=install_requires
      )
