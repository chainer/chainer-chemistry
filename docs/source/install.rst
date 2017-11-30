============
Installation
============

Dependency
========================

Following packages are required to install this library and are automatically
installed when you install the library by `pip` command.

* `chainer <https://docs.chainer.org/en/stable/index.html>`_
* `pandas <https://pandas.pydata.org>`_
* `tqdm <https://pypi.python.org/pypi/tqdm>`_

Also, it uses following library, you need to manually install it.

* `rdkit <https://github.com/rdkit/rdkit>`_

See the `official document <http://www.rdkit.org/docs/Install.html>`_ for installation.
If you have setup ``anaconda``, you may install ``rdkit`` by following command::

   $ conda install -c rdkit rdkit


Install via pip
========================

It can be installed by ``pip`` command::

   $ pip install chainerchem

Install from source
========================

The tarball of the source tree is available via ``pip download chainerchem``.
You can use ``setup.py`` to install Chainer from the tarball::

   $ tar zxf chainerchem-x.x.x.tar.gz
   $ cd chainerchem-x.x.x
   $ python setup.py install

Install from the latest source from the master branch::

   $ git clone https://github.com/pfnet/chainerchem.git
   $ cd chainerchem
   $ python setup.py install
