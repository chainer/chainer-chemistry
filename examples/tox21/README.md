# Training graph convolution models with Tox21 dataset

This is an example of learning toxicity of chemical molecules with graph convolution networks in a multi-task supervised setting.

We use graph convolution models that takes molecules represented as graphs as predictor.
ChainerChem provides off-the-shelf graph convolution models including [NFP](https://arxiv.org/abs/1509.09292), [GGNN](https://arxiv.org/abs/1511.05493), [SchNet](https://arxiv.org/abs/1706.08566) and so on.

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
The default value is `result`.

The full options available including `-g` and `-o` are found
by running the following command `python train_tox21.py --help`.

### Inference with a trained model

With CPU:
```
python inference_tox21.py [-i /path/to/training/result/directory]
```

With GPU:t
```
python inference_tox21.py -g 0 [-i /path/to/training/result/directory]
```

This script loads trained parameters of a model and
makes a prediction for the test dataset of Tox21 with the model.
It loads parameters and other configurations from directory specified by `-i` option,
whose default value is same as that of `-o` option of `train_tox21.py`.
The prediction results are saved in the current directory as a `npy` file.
As with training, we can specify GPU/CPU to use by `-g` option.

The full options available including `-g` and `-i` are found
by running the following command `python inference_tox21.py --help`.
