===========
Quick Start
===========

1. Install
----------

This library can be easily installed by ``pip`` command::

   $ pip install chainerchem

Note that it uses `RDKit <https://github.com/rdkit/rdkit>`_ ,
open-source software for cheminformatics.
Below code is an example to install ``rdkit`` by ``conda`` command provided by
`anaconda <https://www.anaconda.com/what-is-anaconda/>`_::

   $ conda install -c rdkit rdkit

2. Run example training code
----------------------------

`The official repository <https://github.com/pfnet-research/chainerchem>`_ provides examples
several graph convolution networks with the Tox21 and QM9 datasets
(the Tox21 example has inference code as well). You can obtain the code by cloning
the repository::

   $ git clone https://github.com/pfnet-research/chainerchem.git

The following code is how to train Neural Fingerprint (NFP) with the Tox21 dataset on CPU::

   $ cd chainerchem/examples/tox21
   $ python train_tox21.py --method=nfp  --gpu=-1  # set --gpu=0 if you have GPU