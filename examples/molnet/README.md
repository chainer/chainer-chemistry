# MoleculeNet

[MoleculeNet](http://moleculenet.ai/) provides various dataset, which ranges
Physics, Chemistry, Bio and Physiology.

You can specify dataset type, and train the model for the dataset.

## How to run the code

### Train the model by specifying dataset

You can specify dataset type by `--dataset` option.
Please refer [molnet_config.py](https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/datasets/molnet/molnet_config.py) 
for the list of available dataset in Chainer Chemistry.

For example, if you want to train "bbbp" dataset,

With CPU:
```angular2html
python train_molnet.py --dataset=bbbp
```

With GPU:
```angular2html
python train_molnet.py --dataset=bbbp -g 0
```
