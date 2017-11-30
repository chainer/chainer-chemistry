# A Library for Deep Learning in Biology and Chemistry

This library is a collection of tools to train and run neural networks for 
tasks in biology and chemistry using Chainer[1].

It supports various state-of-the-art deep learning neural network models 
(especially Graph Convolution Neural Network) 
for chemical molecule property prediction.


## Quick start

### 1. Install

This library can be installed by `pip` command.

Note that it uses [rdkit](https://github.com/rdkit/rdkit),
Open-Source Cheminformatics Software.
Below code is an example to install `rdkit` by `conda` command provided by
[anaconda](https://www.anaconda.com/what-is-anaconda/).

```bash
pip install chainerchem
conda install -c rdkit rdkit
```

### 2. Run example training code

[The official repository](https://github.com/pfnet/chainerchem) provides examples
several graph convolution networks with the Tox21 and QM9 datasets
(the Tox21 example has inference code as well). You can obtain the code by cloning
the repository:

```bash
git clone https://github.com/pfnet/chainerchem.git
```

The following code is how to train Neural Fingerprint (NFP) with the Tox21 dataset on CPU:

```
cd chainerchem/examples/tox21
python train_tox21.py --method=nfp  --gpu=-1  # set --gpu=0 if you have GPU
```

## Installation

For usual user, `chainerchem` can be installed via PyPI:
```
pip install chainerchem
```

The software is still beta version and in experimental development.
If you would like to use latest sources or develop `chainerchem`, 
please install master branch with the command:

```
git clone https://github.com/pfnet/chainerchem.git
pip install -e chainerchem
```

## Dependencies

Following packages are required to install this library and are automatically
installed when you install the library by `pip` command.

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
- tox21

## License

MIT License. 

PFN provides no warranty or support for this implementation. 
Each model performance is not guaranteed, and may not achieve the score reported in each paper.
Use it at your own risk.

Please see the LICENSE file for details.

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
