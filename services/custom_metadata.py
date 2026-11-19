"""Inspect single-cell metadata without loading expression matrices."""
from pathlib import Path

import h5py
from anndata.io import read_elem


def inspect_custom_metadata(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() not in {'.h5ad', '.h5mu'}:
        raise ValueError('Choose an .h5ad or .h5mu file')
    result = {}
    with h5py.File(path, 'r') as handle:
        groups = dict(handle['mod'].items()) if path.suffix.lower() == '.h5mu' else {'AnnData': handle}
        for name, group in groups.items():
            obs = read_elem(group['obs'])
            var = read_elem(group['var'])
            result[name] = {
                'obs': obs,
                'var': var,
                'layers': list(group['layers'].keys()) if 'layers' in group else [],
                'has_x': 'X' in group and group['X'].attrs.get('encoding-type') != 'null',
                'n_cells': len(obs),
                'n_genes': len(var),
            }
    return result
