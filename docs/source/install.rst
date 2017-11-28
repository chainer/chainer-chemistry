============
Installation
============

Requirements
========================
ChainerChem depends on following library, you need to install it.

* RDKit

::
 conda install -c rdkit rdkit

Dependency
========================

* chainer
* pandas
* tqdm

How to install
========================

For user,

::

 git clone https://github.com/pfn-intern/chainer-chem.git
 python setup.py install


For developer,

::

 git clone https://github.com/pfn-intern/chainer-chem.git
 pip install -e chainer-chem


Supported Model
========================
Currently, following graph convolutional neural networks are implemented.

* NFP: Neural fingerprint
* GGNN: Gated Graph Neural Network
* Weave:
* SchNet:
