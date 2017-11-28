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
