"""Validate compact observed cis results; never infer significance from power."""
import numpy as np
import pandas as pd


def observed_significance_by_gene(candidates, observed, fdr_threshold):
    required = {'element', 'gene', 'empirical_fdr'}
    if missing := required - set(observed.columns):
        raise ValueError('Observed results require columns: ' + ', '.join(sorted(missing)))
    if not 0 <= fdr_threshold <= 1:
        raise ValueError('FDR threshold must be in [0, 1]')
    frame = observed[list(required)].copy()
    if frame[['element', 'gene']].isna().any().any():
        raise ValueError('Observed pair identifiers must not be missing')
    for col in ['element', 'gene']:
        frame[col] = frame[col].astype(str).str.strip()
        if frame[col].eq('').any():
            raise ValueError('Observed pair identifiers must not be empty')
    if frame.duplicated(['element', 'gene']).any():
        raise ValueError('Observed results contain duplicate element/gene pairs')
    fdr = pd.to_numeric(frame.empirical_fdr, errors='coerce')
    if not (np.isfinite(fdr) & fdr.between(0, 1)).all():
        raise ValueError('Observed empirical_fdr values must be finite and in [0, 1]')
    frame['empirical_fdr'] = fdr
    pairs = candidates[['element', 'gene']].drop_duplicates().copy()
    pairs[['element', 'gene']] = pairs[['element', 'gene']].astype(str)
    joined = pairs.merge(frame, on=['element', 'gene'], how='left', validate='one_to_one')
    joined['observed'] = joined.empirical_fdr.notna()
    joined['significant'] = joined.empirical_fdr.le(fdr_threshold)
    return joined.groupby('gene', as_index=False).agg(
        observed_pairs=('observed', 'sum'), significant_pairs=('significant', 'sum'))
