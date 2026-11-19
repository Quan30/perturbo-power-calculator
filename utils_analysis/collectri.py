"""CollecTRI membership, independent of expression baselines and effect signs."""
from __future__ import annotations

import pandas as pd


def prepare_collectri(raw: pd.DataFrame) -> pd.DataFrame:
    """Unique TF-target symbols; retain complexes intact and exclude self edges.

    No evidence or sign filter is applied. Complex-level evidence is not
    reassigned to individual subunits without a biological mapping decision.
    """
    columns = {"source_genesymbol": "tf", "target_genesymbol": "target_gene"}
    missing = set(columns) - set(raw.columns)
    if missing:
        raise ValueError("CollecTRI is missing: " + ", ".join(sorted(missing)))
    out = raw[list(columns)].rename(columns=columns).dropna().copy()
    for col in out.columns:
        out[col] = out[col].astype(str).str.strip()
    out = out.loc[out.tf.ne("") & out.target_gene.ne("") & out.tf.ne(out.target_gene)]
    return out.drop_duplicates().sort_values(["tf", "target_gene"]).reset_index(drop=True)


def collectri_tfs(programs: pd.DataFrame) -> list[str]:
    return sorted(programs["tf"].dropna().unique().tolist())


def collectri_targets(programs: pd.DataFrame, tf: str) -> list[str]:
    return sorted(programs.loc[programs["tf"].eq(tf), "target_gene"].unique().tolist())
