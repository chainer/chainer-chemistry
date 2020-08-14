# Weisfeiler-Lehman Embedding preprocessor implementations

In this directory, we provide an implementaion of [Weisfeiler-Lehman Embedding (WLE)](https://arxiv.org/abs/2006.06909) [1] preprocessor for ChainerChemistry GNN models. 

## How to run the code

### Test run command

```bash
# Training tox21 dataset using RSGCN-CWLE model. Short 3 epoch for testing.
python train_molnet_wle.py --dataset tox21 --method rsgcn_cwle --epoch 3  --device 0

# Prediction with trained model
python predict_molnet_wle.py --dataset tox21 --method rsgcn_cwle --in-dir result --device 0
```

### Train the model by specifying dataset

Basically, no changes from the original molnet examples (examples/molnet/train_molnet.py).
The main difference is the choice of '--method' option.
To test WLE, choose one of 'xxx_wle', 'xxx_cwle', and 'xxx_gwle' where 'xxx' is a GNN architecture identifier (e.g. 'rsgcn', 'relgat').

- xxx_wle: apply the naive WLE to the GNN 'xxx'
- xxx_cwle (recommended): apply the Concat WLE to the GNN 'xxx'
- xxx_gwle: apply the Gated-sum WLE to the GNN 'xxx'

#### Additional options

Introducing the WLE, we have some more additional options.
In general you do not need to specify these options (use the default values!).


## Performance

The paper [1] shows that the use of (C)WLE consistently improves the generalization (test) performance of the several GNN architectures (if hyperparameters are optimized by a Black-box optimizer such as [Optuna] (https://preferred.jp/ja/projects/optuna/).



## References

[1] Katsuhiko Ishiguro, Kenta Oono, and Kohei Hayashi, "Weisfeiler-Lehman Embedding for Molecular Graph Neural Networks", arXiv: 2006.06909, 2020. [paper link](https://arxiv.org/abs/2006.06909) 

