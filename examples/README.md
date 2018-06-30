# Chainer Chemistry examples

These examples are implemented to train the model.

* Tox21: 12 types of toxity classification
* QM9: Chemical property regression
* Own dataset: Own dataset (prepared in csv format) regression
* Molcule Net: Various dataset for both classification and regression

## Test

To test code of all examples, run

```
bash -x test_examples.sh -1 # for CPU
bash -x test_examples.sh 0  # for GPU
```

If you encounter errors, please report them to
[Github issues](https://github.com/pfnet-research/chainer-chemistry/issues)
along with error logs. We appreciate your help.
