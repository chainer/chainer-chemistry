import os

import numpy
import pytest
from rdkit import Chem

from chainer_chemistry.saliency.visualizer.mol_visualizer import MolVisualier  # NOQA
from chainer_chemistry.saliency.visualizer.mol_visualizer import SmilesVisualizer  # NOQA


def test_mol_visualizer(tmpdir):
    # Only test file is saved without error
    smiles = 'OCO'
    mol = Chem.MolFromSmiles(smiles)
    saliency = numpy.array([0.5, 0.3, 0.2])
    visualizer = MolVisualier()

    # 1. test with setting save_filepath
    save_filepath = os.path.join(str(tmpdir), 'tmp.svg')
    visualizer.visualize(saliency, mol, save_filepath=save_filepath)
    assert os.path.exists(save_filepath)

    # 2. test with `save_filepath=None` runs without error
    visualizer.visualize(
        saliency, mol, save_filepath=None, visualize_ratio=0.5,)


def test_smiles_visualizer(tmpdir):
    # Only test file is saved without error
    smiles = 'OCO'
    saliency = numpy.array([0.5, 0.3, 0.2])
    visualizer = SmilesVisualizer()

    # 1. test with setting save_filepath
    save_filepath = os.path.join(str(tmpdir), 'tmp.svg')
    visualizer.visualize(saliency, smiles, save_filepath=save_filepath,
                         add_Hs=False)
    assert os.path.exists(save_filepath)
    save_filepath = os.path.join(str(tmpdir), 'tmp.png')
    visualizer.visualize(saliency, smiles, save_filepath=save_filepath,
                         add_Hs=False)
    # TODO(nakago): support png save test.
    # Do not test for now (cairosvg is necessary)
    # assert os.path.exists(save_filepath)

    # 2. test with `save_filepath=None` runs without error
    visualizer.visualize(
        saliency, smiles, save_filepath=None, visualize_ratio=0.5,
        add_Hs=False, use_canonical_smiles=True)


def test_mol_visualizer_assert_raises(tmpdir):
    visualizer = MolVisualier()
    smiles = 'OCO'
    mol = Chem.MolFromSmiles(smiles)

    with pytest.raises(ValueError):
        # --- Invalid saliency shape ---
        saliency = numpy.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]])
        visualizer.visualize(saliency, mol)

    with pytest.raises(ValueError):
        # --- Invalid sort key ---
        saliency = numpy.array([0.5, 0.3, 0.2])
        invalid_ext_filepath = os.path.join(str(tmpdir), 'tmp.hoge')
        visualizer.visualize(saliency, mol, save_filepath=invalid_ext_filepath)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
