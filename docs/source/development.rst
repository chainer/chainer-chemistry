==================
Development policy
==================

In this section, we describe development policy of Chainer Chemistry.
Developers who are thinking to send PRs to the repository are encouraged to read the following sections
before starting implementation.


Overall
=======

We place importance on adding new features as early as possible.

As the name indicates, Chainer Chemistry is built on top of Chainer.
The core development team of Chainer and that of Chainer Chemistry work together tightly.


Versioning policy
=================

Basically, we follow `the semantic versioning v2.0.0 <https://semver.org>`_.
In Chainer Chemistry, public APIs in the sense of semantic versioning v2.0.0 means ones that is in `the document <http://chainer-chemistry.readthedocs.io/en/latest/index.html>`_. 

During major version zero, we will follow these policies in addition to v2.0.0:

* We do not plan any scheduled releases.
* We do not plan any pre releases in major version zero.
* We release the minor version up when the core development team agrees. Typically, we will do so when (1) sufficient number of features are added since last minor release (2) the latest release cannot run the example code in the master branch of the repository (3) critical bugs are found. But we are not restricted to them.
* If we find critical bugs, we SHOULD release patch version or minor version that fixes them.
The core development team will determine which version to release.

Compatibiity policy
===================

The following is the immediate consequence of following the semantic versioning v2.0.0:

* Until v1.0.0, we MAY break compatibility of public APIs including addition, deletion, change in semantics of them.
* Any changes of public APIs that break compatibility SHOULD be included in major (the first digit) version up after v1.0.0.
* Any changes of public APIs that do not break compatibility SHOULD be included in major or minor (the second digit) version up after v1.0.0.
* Any other changes, including changes in undocumented APIs, bug fixes, improvements in document and performance, and so on MAY be included in any version up.

As the API is still immature and unstable, introduction of new features can sometime involve compatibility break.
Of course we care backward compatibility as much as possible,
But if we are faced with dilemma of breaking backward compatibility, we are likely to give up compatibility.

Chainer Chemistry implements several published deep learning models such as Neural Finger Print like `ChainerCV <https://twitter.com>`_.
We do *NOT* guarantee the reproducibility of experiments the authors did in the published papers.


Branch strategy
===============

The official repository of Chainer Chemistry is https://github.com/pfnet-research/chainer-chemistry. 
We use *the master branch* of the repository for development.
During major version zero, we do not maintain any released versions.
Therefore, developer who makes a PR should send it to the master branch.
When a bug is found, changes that fix the bug SHOULD be merged to the next version (either minor or patch). If the bug is critical, we SHOULD release the next version as soon as possible.

At some point, coding examples in the master branch of the official repository may not work even with the latest release. In that case, Users are recommended to either download the example code of the latest release or update the library code to the master branch.


Coding guideline
================

We use upper camel case (e.g. ``FooBar``) for class names and snake case (e.g. ``foo_bar``) for function, method, variable and package names.

In the context of machine learning, especially chemoinformatics, we use several terms such as feature, feature vectors, descriptor and so on.
to indicate representation of inputs. To avoid disambiguity and align within the library code, we use these terms in the following way:
*Feature* is a representation of a sample of interest (typically molecules in Chainer Chemistry). *Label* is a 
For example, consider a suepervised learning task whose dataset consisting of input-output pairs ``((x_1, y_1), ..., (x_N, y_N))``, where ``N`` is the number of samples.
In Chainer Chemistry ``x_i` and ``y_i`` are called input feature and label, respectively and a pair of ``(x_i, y_i)`` is feature for each ``i``.