# Chainer Chemistry: A Library for Deep Learning in Biology and Chemistry

Chainer Chemistry is a collection of tools to train and run neural networks for 
tasks in biology and chemistry using Chainer[1].

It supports various state-of-the-art deep learning neural network models 
(especially Graph Convolution Neural Network) 
for chemical molecule property prediction.

For more information, you can refer to [documentation](http://chainer-chemistry.readthedocs.io/en/latest/index.html).

### Notes

This repository is currently under construction.
There is no guarantee that example programs work correctly and tests pass.
Please use it at your own risk.

## Quick start

### 1. Installation

Chainer Chemistry can be installed by `pip` command.

Note that it uses [rdkit](https://github.com/rdkit/rdkit),
Open-Source Cheminformatics Software.
Below code is an example to install `rdkit` by `conda` command provided by
[anaconda](https://www.anaconda.com/what-is-anaconda/).

```bash
pip install chainer-chemistry
conda install -c rdkit rdkit
```

### 2. Run example training code

[The official repository](https://github.com/pfnet-research/chainer-chemistry) provides examples
several graph convolution networks with the Tox21 and QM9 datasets
(the Tox21 example has inference code as well). You can obtain the code by cloning
the repository:

```bash
git clone https://github.com/pfnet-research/chainer-chemistry.git
```

The following code is how to train Neural Fingerprint (NFP) with the Tox21 dataset on CPU:

```
cd chainer-chemistry/examples/tox21
python train_tox21.py --method=nfp  --gpu=-1  # set --gpu=0 if you have GPU
```

## Installation

Usual users can install this library via PyPI:
```
pip install chainer-chemistry
```

Chainer Chemistry is still in experimental development.
If you would like to use latest sources.
please install master branch with the command:

```
git clone https://github.com/pfnet-research/chainer-chemistry.git
pip install -e chainer-chemistry
```

#### Note:
You can install this library via pip after v0.1.0 release.

### Dependencies

Following packages are required to install Chainer Chemistry and are automatically
installed when you install it by `pip` command.

 - [`chainer`](https://docs.chainer.org/en/stable/index.html)
 - [`pandas`](https://pandas.pydata.org)
 - [`tqdm`](https://pypi.python.org/pypi/tqdm)

Also, it uses following library, you need to manually install it.

 - [`rdkit`](https://github.com/rdkit/rdkit)
 
See the [official document](http://www.rdkit.org/docs/Install.html) 
for installation.
If you have setup `anaconda`, you may install `rdkit` by following command.

```conda install -c rdkit rdkit```


## Supported model

Currently, following graph convolutional neural networks are implemented.

- NFP: Neural fingerprint [2, 3]
- GGNN: Gated Graph Neural Network [4, 3]
- Weave: [5, 3]
- SchNet: [6]

## Supported dataset

Currently, following dataset is supported.

- QM9 [7, 8]
- Tox21 [9]

## License

MIT License. 

We provide no warranty or support for this implementation.
Each model performance is not guaranteed, and may not achieve the score reported in each paper.
Use it at your own risk.

Please see the [LICENSE](https://github.com/pfnet-research/chainer-chemistry/blob/master/LICENSE) file for details.

## Reference

[1] Tokui, S., Oono, K., Hido, S., & Clayton, J. (2015). Chainer: a next-generation open source framework for deep learning. In Proceedings of workshop on machine learning systems (LearningSys) in the twenty-ninth annual conference on neural information processing systems (NIPS) (Vol. 5).

[2] Duvenaud, D. K., Maclaurin, D., Iparraguirre, J., Bombarell, R., Hirzel, T., Aspuru-Guzik, A., & Adams, R. P. (2015). Convolutional networks on graphs for learning molecular fingerprints. In Advances in neural information processing systems (pp. 2224-2232).

[3] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. arXiv preprint arXiv:1704.01212.

[4] Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2015). Gated graph sequence neural networks. arXiv preprint arXiv:1511.05493.

[5] Kearnes, S., McCloskey, K., Berndl, M., Pande, V., & Riley, P. (2016). Molecular graph convolutions: moving beyond fingerprints. Journal of computer-aided molecular design, 30(8), 595-608.

[6] Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko, Klaus-Robert Müller (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.
arXiv preprint arXiv:1706.08566

[7] L. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond, Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17, J. Chem. Inf. Model. 52, 2864–2875, 2012.

[8] R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, Quantum chemistry structures and properties of 134 kilo molecules, Scientific Data 1, 140022, 2014.

[9] Huang R, Xia M, Nguyen D-T, Zhao T, Sakamuru S, Zhao J, Shahane SA, Rossoshek A and Simeonov A (2016) Tox21Challenge to Build Predictive Models of Nuclear Receptor and Stress Response Pathways as Mediated by Exposure to Environmental Chemicals and Drugs. Front. Environ. Sci. 3:85. doi: 10.3389/fenvs.2015.00085