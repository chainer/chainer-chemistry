=======
Dataset
=======


Converters
==========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerchem.dataset.converters.concat_mols


Indexers
========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerchem.dataset.indexer.BaseIndexer
   chainerchem.dataset.indexer.BaseFeatureIndexer
   chainerchem.dataset.indexers.NumpyTupleDatasetFeatureIndexer


Parsers
=======

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerchem.dataset.parsers.BaseParser
   chainerchem.dataset.parsers.CSVFileParser
   chainerchem.dataset.parsers.SDFFileParser


Preprocessors
=============

Base preprocessors
------------------


.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerchem.dataset.preprocessors.BasePreprocessor
   chainerchem.dataset.preprocessors.MolPreprocessor

Concrete preprocessors
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:


   chainerchem.dataset.preprocessors.AtomicNumberPreprocessor
   chainerchem.dataset.preprocessors.ECFPPreprocessor
   chainerchem.dataset.preprocessors.GGNNPreprocessor
   chainerchem.dataset.preprocessors.NFPPreprocessor
   chainerchem.dataset.preprocessors.SchNetPreprocessor
   chainerchem.dataset.preprocessors.WeaveNetPreprocessor

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainerchem.dataset.preprocessors.MolFeatureExtractionError
   chainerchem.dataset.preprocessors.type_check_num_atoms
   chainerchem.dataset.preprocessors.construct_atomic_number_array
   chainerchem.dataset.preprocessors.construct_adj_matrix