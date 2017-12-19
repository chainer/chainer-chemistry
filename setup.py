from distutils.core import setup

from setuptools import find_packages

setup_requires = []
install_requires = [
    'chainer>=2.0',
    'pandas',
    'tqdm',
]

setup(name='chainer-chemistry',
      version='0.1.0',
      description='Chainer Chemistry: A Library for Deep Learning in Biology and Chemistry',
      author='Kosuke Nakago',
      author_email='nakago@preferred.jp',
      packages=find_packages(),
      license='MIT',
      url='http://chainer-chemistry.readthedocs.io/en/latest/index.html',
      setup_requires=setup_requires,
      install_requires=install_requires
      )
