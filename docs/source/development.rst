.. _development-policy:

==================
Development policy
==================

In this section, we describe the development policy that the core developers follow.
Developers who are thinking to send PRs to the repository are encouraged to read the following sections
before starting implementation.


Versioning policy
=================

Basically, we follow the `semantic versioning v2.0.0 <https://semver.org/spec/v2.0.0.html>`_.
In Chainer Chemistry, *public APIs* in the sense of semantic versioning are ones in `the document <http://chainer-chemistry.readthedocs.io/en/latest/index.html>`_.

We follow these rules about versioning during the major version zero in addition to ones described in the the semantic versioning:

* We do not plan any scheduled releases.
* We do not plan any pre releases.
* We release the minor version when the core development team agrees. Typically, we do so when (1) sufficient number of features are added since the last minor release (2) the latest release cannot run the example code in the master branch of the repository (3) critical bugs are found. But we are not restricted to them.
* If we find critical bugs, we should release a patch version or a minor version that fixes them. The core development team will determine which version to release.

We do not have a concrete plan about versioning strategy after v1.0.0.


Compatibiity policy
===================

As an immediate consequence of the semantic versioning, we may break compatibility of public APIs including addition, deletion, and changes in their semantics anytime in the major version zero.
Since APIs of Chainer Chemistry are still immature and unstable, we expect introduction of new features can sometime involve compatibility break.
If we are faced with a dilemma between cost for backward compatibility and benefit of new features, we are likely to give up the former because we want to place importance on introducing new features as soon as possible. Of course, we care backward compatibility whenever it is easy and low-cost.

Like `ChainerCV <https://twitter.com>`_, Chainer Chemistry provides several off-the-shelf deep learning models (e.g. Neural Finger Print) whose papers are available in such as arXiv or conferences related to machine learning.
Although, most of published papers reports evaluation results of the models with publicly available datasets, we do *NOT* guarantee the reproducibility of experiments in the papers.

At some point, coding examples in the master branch of the official repository may not work even with the latest release. In that case, users are recommended to either use the example code of the latest release or update the library code to the master branch.

As of v0.3.0, we have introduced `BaseForwardModel`, which provides methods for serializing itself to and loading from a file.
As these methods intenally use `pickle <https://docs.python.org/3/library/pickle.html>`_, portability of the class depends on that of pickling.
Especially, serialized instances of `BaseForwardModel` made with older Chainer Chemistry may not be loaded with newer one, partly because we may change their internal structures for refactoring, performance improvement, and so on.
See the document of `BaseForwardModel` and their subclasses (e.g. `Classifier`, `Regressor`).

Branch strategy
===============

The official repository of Chainer Chemistry is https://github.com/pfnet-research/chainer-chemistry. 
We use the *master* branch of the repository for development. Therefore, developer who makes PRs should send them to the master branch.

During major version zero, we do not maintain any released versions.
When a bug is found, changes for the bug should be merged to the next version (either minor or patch). If the bug is critical, we will release the next version as soon as possible.


Coding guideline
================

We basically adopt `PEP8 <https://www.python.org/dev/peps/pep-0008/>_` as a style guide.
You can check it with `flake8`, which we can install by::

   $ pip install flake8

and run with ``flake8`` command.

In addition to PEP8, we use upper camel case (e.g. ``FooBar``) for class names and snake case (e.g. ``foo_bar``) for function, method, variable and package names.
Although we recommend developers to follow these rules as well, they are not mandatory.

For documents, we follow the `Google Python Style Guide <http://google.github.io/styleguide/pyguide.html#Comments>`_
and compile it with `Napoleon <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html>`_,
which is an extension of `Sphinx <http://www.sphinx-doc.org/en/stable/>`_.


Testing guideline
=================

Chainer Chemistry uses `pytest <https://docs.pytest.org/en/latest/index.html>`_  as a unit-test framework.
All unit tests are located in ``tests/`` directory. We can run tests with normal usage of pytest.
For example, the following command runs all unit tests::

   $ pytest tests

Some unit tests require GPUs, which are annotated with ``@pytest.mark.gpu``.
Therefore, you can skip them with ``-m`` option::

   $ pytest -m "not gpu" tests

If a develop who write a unit test that uses GPUs, you must anotate it with ``@pytest.mark.gpu``.

Similarly, some unit tests take long time to complete.
We annotated them with ``@pytest.mark.slow`` and can skip them with ``-m`` option::

   $ pytest -m "not slow" tests

Any unit test that uses GPUs muct be annotated with ``@pytest.mark.slow``.

We can skip both GPU and slow tests with the following command::

   $ pytest -m "not (gpu or slow)" tests


Terminology
===========

In the context of machine learning, especially chemoinformatics, we use several terms such as feature, feature vectors, descriptor and so on
to indicate representation of inputs. To avoid disambiguity and align naming convention within the library code, we use these terms in the following way:

* *Feature* is a representation of a sample of interest (typically molecules in Chainer Chemistry).
* *Label* is a target value of we want to predict.
* *Input feature* is a representation of a sample from which we want to predict the target value.

For example, consider a suepervised learning task whose dataset consisting of input-output pairs ``((x_1, y_1), ..., (x_N, y_N))``, where ``N`` is the number of samples.
In Chainer Chemistry ``x_i` and ``y_i`` are called input feature and label, respectively and a pair of ``(x_i, y_i)`` is feature for each ``i``.


Relation to Chainer
===================

`Chainer <https://chainer.org>`_ is a deep learning framework written in Python that features dynamic
computational graph construction (the "define-by-run" paradigm) for flexible and intuitive model development.
As the name indicates, Chainer Chemistry is an extension library of Chainer built on top of it.
The core development team members of Chainer and that of Chainer Chemistry work together tightly.