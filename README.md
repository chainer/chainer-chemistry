# Chainer Chemistry: A Library for Deep Learning in Biology and Chemistry
dios.ros.md
[![PyPI](https://img.shields.io/pypi/v/chainer-chemistry.svg)](https://pypi.python.org/pypi/chainer-chemistry)
[![GitHub license](https://img.shields.io/github/license/pfnet-research/chainer-chemistry.svg)](https://github.com/pfnet-research/chainer-chemistry/blob/master/LICENSE)
[![travis](https://img.shields.io/travis/pfnet-research/chainer-chemistry/master.svg)](https://travis-ci.org/pfnet-research/chainer-chemistry)
[![Read the Docs](https://readthedocs.org/projects/chainer-chemistry/badge/?version=latest)](http://chainer-chemistry.readthedocs.io/en/latest/?badge=latest)

<p align="center">
  <img src="assets/chainer-chemistry-overview.png" alt="Chainer Chemistry Overview" width="600" />
</p>

Chainer Chemistry is a deep learning framework (based on Chainer) with
applications in Biology and Chemistry. It supports various state-of-the-art
models (especially GCNN - Graph Convolutional Neural Network) for chemical property prediction.

For more information, please refer to the [documentation](http://chainer-chemistry.readthedocs.io/en/latest/index.html).
Also, a quick introduction to deep learning for molecules and Chainer Chemistry
is available [here](https://www.slideshare.net/KentaOono/deep-learning-for-molecules-introduction-to-chainer-chemistry-93288837).

## Dependencies

Chainer Chemistry depends on the following packages:

 - [`chainer`](https://docs.chainer.org/en/stable/index.html)
 - [`pandas`](https://pandas.pydata.org)
 - [`scikit-learn`](http://scikit-learn.org/stable/)
 - [`tqdm`](https://pypi.python.org/pypi/tqdm)

These are automatically added to the system when installing the library via the
`pip` command (see _Installation_). However, the following  needs to be
installed manually:

 - [`rdkit (release 2017.09.3.0)`](https://github.com/rdkit/rdkit)
 
Please refer to the RDKit [documentation](http://www.rdkit.org/docs/Install.html)
for more information regarding the installation steps.

Note that only the following versions of Chainer Chemistry's dependencies are
currently supported:

| Chainer Chemistry   | Chainer         | RDKit          |
| ------------------: | --------------: | -------------: |
| v0.1.0 ~ v0.3.0     | v2.0 ~ v3.0     | 2017.09.3.0    |
| v0.4.0              | v3.0 ~ v4.0 *1  | 2017.09.3.0    |
| master branch       | v3.0 ~ v5.0     | 2017.09.3.0    |

## Installation

Chainer Chemistry can be installed using the `pip` command, as follows:

```
pip install chainer-chemistry
```

If you would like to use the latest sources, please checkout the master branch
and install with the following commands:

```
git clone https://github.com/pfnet-research/chainer-chemistry.git
pip install -e chainer-chemistry
```

## Sample Code

Sample code is provided with this repository. This includes, but is not limited
to, the following:

- Training a new model on a given dataset
- Performing inference on a given dataset, using a pretrained model
- Evaluating and reporting performance metrics of different models on a given
dataset

Please refer to the `examples` directory for more information.

## Supported Models

The following graph convolutional neural networks are currently supported:

- NFP: Neural Fingerprint [2, 3]
- GGNN: Gated Graph Neural Network [4, 3]
- WeaveNet [5, 3]
- SchNet [6] 
- RSGCN: Renormalized Spectral Graph Convolutional Network [10]<br/>
 \* The name is not from the original paper - see [PR #89](https://github.com/pfnet-research/chainer-chemistry/pull/89) for the naming convention.
- RelGCN: Relational Graph Convolutional Network [14]

## Supported Datasets

The following datasets are currently supported:

- QM9 [7, 8]
- Tox21 [9]
- MoleculeNet [11]
- ZINC (only 250k dataset) [12, 13]
- User (own) dataset

## Research Projects

If you use Chainer Chemistry in your research, feel free to submit a
pull request and add the name of your project to this list:

 - BayesGrad: Explaining Predictions of Graph Convolutional Networks ([paper](https://arxiv.org/abs/1807.01985), [code](https://github.com/pfnet-research/bayesgrad))

## Useful Links

Chainer Chemistry:

 - [Documentation](https://chainer-chemistry.readthedocs.io)
 - [Research Blog](https://preferredresearch.jp/2017/12/18/chainer-chemistry-beta-release/)

Other Chainer frameworks:

 - [Chainer: A Flexible Framework of Neural Networks for Deep Learning](https://chainer.org/)
 - [ChainerRL: Deep Reinforcement Learning Library Built on Top of Chainer](https://github.com/chainer/chainerrl)
 - [ChainerCV: A Library for Deep Learning in Computer Vision](https://github.com/chainer/chainercv)
 - [ChainerMN: Scalable Distributed Deep Learning with Chainer](https://github.com/chainer/chainermn)
 - [ChainerUI: User Interface for Chainer](https://github.com/chainer/chainerui)

## License

This project is released under the MIT License. Please refer to the
[this page](https://github.com/pfnet-research/chainer-chemistry/blob/master/LICENSE)
for more information.

Please note that Chainer Chemistry is still in experimental development.
We continuously strive to improve its functionality and performance, but at
this stage we cannot guarantee the reproducibility of any results published in
papers. Use the library at your own risk.


## References

[1] Seiya Tokui, Kenta Oono, Shohei Hido, and Justin Clayton. Chainer: a next-generation open source framework for deep learning. In *Proceedings of Workshop on Machine Learning Systems (LearningSys) in Advances in Neural Information Processing System (NIPS) 28*, 2015.

[2] David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alan Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, *Advances in Neural Information Processing Systems (NIPS) 28*, pages 2224–2232. Curran Asso- ciates, Inc., 2015.

[3] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. *arXiv preprint arXiv:1704.01212*, 2017.

[4] Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. *arXiv preprint arXiv:1511.05493*, 2015.

[5] Steven Kearnes, Kevin McCloskey, Marc Berndl, Vijay Pande, and Patrick Riley. Molecular graph convolutions: moving beyond fingerprints. *Journal of computer-aided molecular design*, 30(8):595–608, 2016.

[6] Kristof Schütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Rober Müller. Schnet: A continuous-filter convolutional neural network for modeling quantum interactions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, *Advances in Neural Information Processing Systems (NIPS) 30*, pages 992–1002. Curran Associates, Inc., 2017.

[7] Lars Ruddigkeit, Ruud Van Deursen, Lorenz C Blum, and Jean-Louis Reymond. Enumeration of 166 billion organic small molecules in the chemical universe database gdb-17. *Journal of chemical information and modeling*, 52(11):2864–2875, 2012.

[8] Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific data*, 1:140022, 2014.

[9] Ruili Huang, Menghang Xia, Dac-Trung Nguyen, Tongan Zhao, Srilatha Sakamuru, Jinghua Zhao, Sampada A Shahane, Anna Rossoshek, and Anton Simeonov. Tox21challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs. *Frontiers in Environmental Science*, 3:85, 2016.

[10] Kipf, Thomas N. and Welling, Max. Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*, 2017.

[11] Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, Vijay Pande, MoleculeNet: A Benchmark for Molecular Machine Learning, arXiv preprint, arXiv: 1703.00564, 2017.

[12] J. J. Irwin, T. Sterling, M. M. Mysinger, E. S. Bolstad, and R. G. Coleman. Zinc: a free tool to discover chemistry for biology. *Journal of chemical information and modeling*, 52(7):1757–1768, 2012.

[13] Preprocessed csv file downloaded from https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv

[14] Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling. Modeling Relational Data with Graph Convolutional Networks. *Extended Semantic Web Conference (ESWC)*, 2018.
