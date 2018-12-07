import numpy

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


from chainer_chemistry.saliency.visualizer.base_visualizer import BaseVisualizer  # NOQA
from chainer_chemistry.saliency.visualizer.common import red_blue_cmap, abs_max_scaler  # NOQA


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


class MolVisualier(BaseVisualizer):

    def __init__(self, logger=None):
        self.logger = logger(__name__)

    def visualize(self, saliency, mol, save_filepath=None,
                  visualize_ratio=1.0, color_fn=red_blue_cmap,
                  scaler=abs_max_scaler, legend=''):
        rdDepictor.Compute2DCoords(mol)
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
        num_atoms = mol.GetNumAtoms()

        # --- type check ---
        if not saliency.ndim == 1:
            raise ValueError("[ERROR] Unexpected value saliency.shape={}"
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
                # TODO: check it is possible without cairosvg or not
                try:
                    import cairosvg
                except ImportError as e:
                    print('cairosvg is not installed! '
                          'Please install cairosvg to save by png format.')
                    raise e
                cairosvg.svg2png(bytestring=svg, write_to=save_filepath)
            else:
                raise ValueError(
                    'Unsupported extention {} for save_filepath {}'
                    .format(extention, save_filepath))
        else:
            from IPython.core.display import SVG
            return SVG(svg.replace('svg:', ''))


class SmilesVisualizer(MolVisualier):

    def visualize(self, saliency, smiles, save_filepath=None,
                  visualize_ratio=1.0, color_fn=red_blue_cmap,
                  scaler=abs_max_scaler, legend='', add_Hs=False,
                  use_canonical_smiles=True):
        mol = Chem.MolFromSmiles(smiles)
        if use_canonical_smiles:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            mol = Chem.MolFromSmiles(smiles)
        if add_Hs:
            mol = Chem.AddHs(mol)
        super(SmilesVisualizer, self).visualize(
            saliency, mol, save_filepath=save_filepath,
            visualize_ratio=visualize_ratio, color_fn=color_fn, scaler=scaler,
            legend=legend)
