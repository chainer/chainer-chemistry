# Training graph convolution models with Tox21 dataset

This is an example of learning toxicity of chemical molecules with graph convolution networks in a multi-task supervised setting.

We use graph convolution models that takes molecules represented as graphs as predictor.
Chainer Chemistry provides off-the-shelf graph convolution models including [NFP](https://arxiv.org/abs/1509.09292), [GGNN](https://arxiv.org/abs/1511.05493), [SchNet](https://arxiv.org/abs/1706.08566) and so on.

We use Tox21 dataset, provided by [The Toxicology in the 21st Century (Tox21)](https://ncats.nih.gov/tox21).
It is one of the most widely used datasets in bio and chemo informatics
and consists of the chemical information of molecules and their assessments of toxicity.

## How to run the code

### Train the model with tox21 dataset

With CPU:
```angular2html
python train_tox21.py
```

With GPU:
```angular2html
python train_tox21.py -g 0
```

This script trains the model with the tox21 dataset
and outputs trained parameters and other information to a specified directory.
We specify an ID of GPU in use by `-g` or `--gpu` option.
Negative value indicate running the code with CPU.
The output directory can be specified by `-o` option.
Its default value is `result`.
The Tox21 dataset consists of several assays.
Some molecules can have more than one types of assay results.
We can specify which assay to use by specifying an assay name with `-l` option.
Assay names are available by running the script with `-h` or `--help`
or execute the following command:

```
python -c import chainer_chemistry; chainer_chemistry.datasets.get_tox21_label_names()
```

If `-l` option is not specified, this script conducts multitask learning with all labels.

The full options available including `-g` and `-o` are found
by running the following command:

```
python train_tox21.py -h
```

### Inference with a trained model using Classifier

As of v0.3.0, `Classifier` class is introduced which supports `predict` and
`predict_proba` methods for easier inference.

`Classifier` also supports `load_pickle` method, user may load
the instance of pretrained-model using `pickle` file.

The example implemented in `predict_tox21_with_classifier.py`.

With CPU:
```
python predict_tox21_with_classifier.py [-i /path/to/training/result/directory]
```

With GPU:
```
python predict_tox21_with_classifier.py -g 0 [-i /path/to/training/result/directory]
```

### Evaluation of Models
`seaborn` is required to run this script.

```
bash examples/tox21/evaluate_models_tox21.sh
```

This script evaluates each method and generate a graph.
