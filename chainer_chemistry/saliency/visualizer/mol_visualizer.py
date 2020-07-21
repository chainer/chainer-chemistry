from logging import getLogger

import numpy

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor


from chainer_chemistry.saliency.visualizer.base_visualizer import BaseVisualizer  # NOQA
from chainer_chemistry.saliency.visualizer.visualizer_utils import red_blue_cmap, abs_max_scaler  # NOQA


def _convert_to_2d(axes, nrows, ncols):
    if nrows == 1 and ncols == 1:
        axes = numpy.array([[axes]])
    elif nrows == 1:
        axes = axes[None, :]
    elif ncols == 1:
        axes = axes[:, None]
    else:
        pass
    assert axes.ndim == 2
    return axes


def is_visible(begin, end):
    if begin <= 0 or end <= 0:
        return 0
    elif begin >= 1 or end >= 1:
        return 1
    else:
        return (begin + end) * 0.5


class MolVisualizer(BaseVisualizer):

    """Saliency visualizer for mol data

    Args:
        logger:
    """

    def __init__(self, logger=None):
        self.logger = logger or getLogger(__name__)

    def visualize(self, saliency, mol, save_filepath=None,
                  visualize_ratio=1.0, color_fn=red_blue_cmap,
                  scaler=abs_max_scaler, legend='', raise_import_error=False
                  ):
        """Visualize or save `saliency` with molecule

        returned value can be used for visualization.

        .. admonition:: Example

           >>> svg = visualizer.visualize(saliency, mol)
           >>>
           >>> # For a Jupyter user, it will show figure on notebook.
           >>> from IPython.core.display import SVG
           >>> SVG(svg.replace('svg:', ''))
           >>>
           >>> # For a user who want to save a file as png
           >>> import cairosvg
           >>> cairosvg.svg2png(bytestring=svg, write_to="foo.png")

        Args:
            saliency (numpy.ndarray): 1-dim saliency array (num_node,)
            mol (Chem.Mol): mol instance of this saliency
            save_filepath (str or None): If specified, file is saved to path.
            visualize_ratio (float): If set, only plot saliency color of top-X
                atoms.
            color_fn (callable): color function to show saliency
            scaler (callable): function which takes `x` as input and outputs
                scaled `x`, for plotting.
            legend (str): legend for the plot
            raise_import_error (bool): raise error when `ImportError` is raised

        Returns:
            svg (str): drawed svg text.
        """
        rdDepictor.Compute2DCoords(mol)
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
        num_atoms = mol.GetNumAtoms()

        # --- type check ---
        if saliency.ndim != 1:
            raise ValueError("Unexpected value saliency.shape={}"
                             .format(saliency.shape))

        # Cut saliency array for unnecessary tail part
        saliency = saliency[:num_atoms]
        if scaler is not None:
            # Normalize to [-1, 1] or [0, 1]
            saliency = scaler(saliency)

        abs_saliency = numpy.abs(saliency)
        if visualize_ratio < 1.0:
            threshold_index = int(num_atoms * visualize_ratio)
            idx = numpy.argsort(abs_saliency)
            idx = numpy.flip(idx, axis=0)
            # set threshold to top `visualize_ratio` saliency
            threshold = abs_saliency[idx[threshold_index]]
            saliency = numpy.where(abs_saliency < threshold, 0., saliency)
        else:
            threshold = numpy.min(saliency)

        highlight_atoms = list(map(lambda g: g.__int__(), numpy.where(
            abs_saliency >= threshold)[0]))
        atom_colors = {i: color_fn(e) for i, e in enumerate(saliency)}
        bondlist = [bond.GetIdx() for bond in mol.GetBonds()]

        def color_bond(bond):
            begin = saliency[bond.GetBeginAtomIdx()]
            end = saliency[bond.GetEndAtomIdx()]
            return color_fn(is_visible(begin, end))
        bondcolorlist = {i: color_bond(bond)
                         for i, bond in enumerate(mol.GetBonds())}
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 375)
        drawer.DrawMolecule(
            mol, highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors, highlightBonds=bondlist,
            highlightBondColors=bondcolorlist, legend=legend)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        if save_filepath:
            extention = save_filepath.split('.')[-1]
            if extention == 'svg':
                with open(save_filepath, 'w') as f:
                    f.write(svg)
            elif extention == 'png':
                # TODO(nakago): check it is possible without cairosvg or not
                try:
                    import cairosvg
                    cairosvg.svg2png(bytestring=svg, write_to=save_filepath)
                except ImportError as e:
                    self.logger.error(
                        'cairosvg is not installed! '
                        'Please install cairosvg to save by png format.\n'
                        'pip install cairosvg')
                    if raise_import_error:
                        raise e
            else:
                raise ValueError(
                    'Unsupported extention {} for save_filepath {}'
                    .format(extention, save_filepath))
        return svg


class SmilesVisualizer(MolVisualizer):

    def visualize(self, saliency, smiles, save_filepath=None,
                  visualize_ratio=1.0, color_fn=red_blue_cmap,
                  scaler=abs_max_scaler, legend='', add_Hs=False,
                  use_canonical_smiles=True, raise_import_error=False):
        """Visualize or save `saliency` with molecule

        See parent `MolVisualizer` class for further usage.

        Args:
            saliency (numpy.ndarray): 1-dim saliency array (num_node,)
            smiles (str): smiles of the molecule.
            save_filepath (str or None): If specified, file is saved to path.
            visualize_ratio (float): If set, only plot saliency color of top-X
                atoms.
            color_fn (callable): color function to show saliency
            scaler (callable): function which takes `x` as input and outputs
                scaled `x`, for plotting.
            legend (str): legend for the plot
            add_Hs (bool): Add explicit H or not
            use_canonical_smiles (bool): If `True`, smiles are converted to
                canonical smiles before constructing `mol`
            raise_import_error (bool): raise error when `ImportError` is raised

        Returns:
            svg (str): drawed svg text.
        """
        mol = Chem.MolFromSmiles(smiles)
        if use_canonical_smiles:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.MolFromSmiles(smiles)
        if add_Hs:
            mol = Chem.AddHs(mol)
        return super(SmilesVisualizer, self).visualize(
            saliency, mol, save_filepath=save_filepath,
            visualize_ratio=visualize_ratio, color_fn=color_fn, scaler=scaler,
            legend=legend, raise_import_error=raise_import_error)
