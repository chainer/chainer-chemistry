from rdkit import Chem

from chainer_chemistry.dataset.preprocessors.base_preprocessor import BasePreprocessor  # NOQA


class MolPreprocessor(BasePreprocessor):
    """preprocessor class specified for rdkit mol instance"""

    def __init__(self, add_Hs=False):
        super(MolPreprocessor, self).__init__()
        self.add_Hs = add_Hs

    def prepare_smiles_and_mol(self, mol):
        """Prepare `smiles` and `mol` used in following preprocessing.

        This method is called before `get_input_features` is called, by parser
        class.
        This method may be overriden to support custom `smile`/`mol` extraction

        Args:
            mol (mol): mol instance

        Returns (tuple): (`smiles`, `mol`)
        """
        # Note that smiles expression is not unique.
        # we should re-obtain smiles from `mol`, so that the
        # smiles order does not contradict with input_features'
        # order.
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        if self.add_Hs:
            mol = Chem.AddHs(mol)
            smiles = Chem.MolToSmiles(mol)
        return smiles, mol

    def get_label(self, mol, label_names=None):
        """Extracts label information from a molecule.

        This method extracts properties whose keys are
        specified by ``label_names`` from a molecule ``mol``
        and returns these values as a list.
        The order of the values is same as that of ``label_names``.
        If the molecule does not have a
        property with some label, this function fills the corresponding
        index of the returned list with ``None``.

        Args:
            mol (rdkit.Chem.Mol): molecule whose features to be extracted
            label_names (None or iterable): list of label names.

        Returns:
            list of str: label information. Its length is equal to
            that of ``label_names``. If ``label_names`` is ``None``,
            this function returns an empty list.

        """
        if label_names is None:
            return []

        label_list = []
        for label_name in label_names:
            if mol.HasProp(label_name):
                label_list.append(mol.GetProp(label_name))
            else:
                label_list.append(None)

                # TODO(Nakago): Review implementation
                # Label -1 work in case of classification.
                # However in regression, assign -1 is not a good strategy...
                # label_list.append(-1)

                # Failed to GetProp for label, skip this case.
                # print('no label')
                # raise MolFeatureExtractionError

        return label_list

    def get_input_features(self, mol):
        """get molecule's feature representation, descriptor.

        Each subclass must override this method.

        Args:
            mol (Mol): molecule whose feature to be extracted.
                `mol` is prepared by the method `prepare_smiles_and_mol`.
        """
        raise NotImplementedError

    def process(self, filepath):
        # Not used now...
        pass
