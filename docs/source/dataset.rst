=======
Dataset
=======


Converters
==========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer_chemistry.dataset.converters.concat_mols


Indexers
========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer_chemistry.dataset.indexer.BaseIndexer
   chainer_chemistry.dataset.indexer.BaseFeatureIndexer
   chainer_chemistry.dataset.indexers.NumpyTupleDatasetFeatureIndexer


Parsers
=======

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer_chemistry.dataset.parsers.BaseParser
   chainer_chemistry.dataset.parsers.CSVFileParser
   chainer_chemistry.dataset.parsers.SDFFileParser
   chainer_chemistry.dataset.parsers.DataFrameParser
   chainer_chemistry.dataset.parsers.SmilesParser


Preprocessors
=============

Base preprocessors
------------------


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer_chemistry.dataset.preprocessors.BasePreprocessor
   chainer_chemistry.dataset.preprocessors.MolPreprocessor

Concrete preprocessors
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:


   chainer_chemistry.dataset.preprocessors.AtomicNumberPreprocessor
   chainer_chemistry.dataset.preprocessors.ECFPPreprocessor
   chainer_chemistry.dataset.preprocessors.GGNNPreprocessor
   chainer_chemistry.dataset.preprocessors.NFPPreprocessor
   chainer_chemistry.dataset.preprocessors.SchNetPreprocessor
   chainer_chemistry.dataset.preprocessors.WeaveNetPreprocessor

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer_chemistry.dataset.preprocessors.MolFeatureExtractionError
   chainer_chemistry.dataset.preprocessors.type_check_num_atoms
   chainer_chemistry.dataset.preprocessors.construct_atomic_number_array
   chainer_chemistry.dataset.preprocessors.construct_adj_matrix



Splitters
==========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer_chemistry.dataset.splitters.RandomSplitter
   chainer_chemistry.dataset.splitters.StratifiedSplitter
   chainer_chemistry.dataset.splitters.ScaffoldSplitter
