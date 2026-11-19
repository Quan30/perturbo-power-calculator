"""Analysis 1 — enhancer/variant to nearby-gene cis-regulatory power."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from services.interface import show_navigation, show_method_status

from utils_analysis.analysis1 import (
    evaluate_cis_element,
    evaluate_cis_dataset,
    fraction_powered_vs_cells,
    fraction_powered_vs_depth,
    gene_power_curves,
    select_variant_nearby,
    summarize_cis_power,
)
from utils_analysis.reference_data import (
    available_gasperini_elements,
    build_gasperini_cis_reference,
    counts_to_depth_multiplier,
    reference_depth_counts,
)
from services.baseline_catalog import source_signature
from services.interface import show_baseline_summary
from services.file_selection import choose_file
from services.power_calibration import load_power_calibration, select_matching_power_calibration


st.set_page_config(page_title="PerTurbo Cis-Regulatory Power", layout="wide")

_APP_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASELINE = os.environ.get(
    "CORTADO_GASPERINI_BASELINE",
    str(_APP_ROOT / "baselines" / "gasperini_baseline_params.npz"),
)
_DEFAULT_PAIR_TABLE = os.environ.get(
    "CORTADO_GASPERINI_PAIR_TABLE",
    str(_APP_ROOT / "artifacts" / "GSE120861_gene_gRNAgroup_pair_table.at_scale.txt.gz"),
)
_DEFAULT_DETECTION = os.environ.get(
    "CORTADO_GASPERINI_DETECTION_SUMMARY",
    str(_APP_ROOT / "artifacts" / "gene_detection_summary.parquet"),
)
_CALIBRATIONS = tuple(
    load_power_calibration(path)
    for path in (
        _APP_ROOT / "artifacts" / "gasperini_cis_power_calibration.json",
        _APP_ROOT / "artifacts" / "gasperini_cis_increase_power_calibration_10rep.json",
    )
)


@st.cache_data(show_spinner=False)
def _load_reference(pair_table_path: str, baseline_path: str, detection_path: str, signatures: tuple) -> pd.DataFrame:
    return build_gasperini_cis_reference(
        pair_table_path=pair_table_path,
        baseline_path=baseline_path,
        detection_path=detection_path,
    )


def _plot_fraction_curve(
    df: pd.DataFrame, x: str, xlabel: str, target_power: float, reference_depth: float
):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x_values = reference_depth * df[x] if x == "depth_multiplier" else df[x]
    ax.plot(x_values, 100 * df["fraction_powered"], marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Genes reaching ≥{target_power * 100:.0f}% power (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


st.title("Analysis 1 — Cis-Regulatory Power")
st.caption(
    "For a selected Gasperini enhancer or genomic position, estimate which nearby genes are detectable "
    "at a chosen effect size, cells per guide, and sequencing depth."
)

with st.expander("What this analysis means", expanded=False):
    st.markdown(
        """
        This page uses a **fast negative-binomial analytical approximation** for interactive
        design exploration. For supported Gasperini cis-enhancer settings, its decision
        threshold is corrected using repeated Perturbo simulations and held-out enhancer–gene
        pairs. The uncorrected analytical estimate remains available below for comparison.

        **Reference expression** is `exp(beta_0)` from the fitted baseline model. It is not
        literal WT expression. **Detected in cells** is the fraction of quality-filtered
        Gasperini screen cells with nonzero expression for the gene; it is not an NTC-only
        detection rate because the screen contains too few clean NTC-only cells for that.
        """
    )

show_navigation()
show_method_status()

with st.popover('Change data files'):
    selected_baseline = st.session_state.get("gpu_baseline_path") if st.session_state.get("gpu_dataset") == "Gasperini" else None
    baseline_path = choose_file("Baseline parameters", selected_baseline or _DEFAULT_BASELINE, {".npz"}, _APP_ROOT / "baselines" / "uploads", key="cis_baseline")
    pair_table_path = choose_file("Gasperini pair table", _DEFAULT_PAIR_TABLE, {".tsv", ".txt", ".gz"}, _APP_ROOT / "data" / "uploads", key="cis_pairs")
    detection_path = choose_file("Gene detection summary", _DEFAULT_DETECTION, {".parquet", ".csv", ".tsv", ".txt"}, _APP_ROOT / "data" / "uploads", key="cis_detection")

paths = {
    "Baseline parameters": baseline_path,
    "Gasperini pair table": pair_table_path,
    "Gene detection summary": detection_path,
}
missing = [f"{name}: {path}" for name, path in paths.items() if not Path(path).exists()]
if missing:
    st.warning(
        "Analysis 1 needs three local precomputed artifacts. Choose files on this page or "
        "the documented environment variables.\n\n" + "\n".join(f"- {item}" for item in missing)
    )
    st.stop()

try:
    with st.spinner("Loading Gasperini reference data..."):
        signatures = tuple(source_signature(Path(p)) for p in (pair_table_path, baseline_path, detection_path))
        reference = _load_reference(pair_table_path, baseline_path, detection_path, signatures)
        reference_depth = reference_depth_counts(baseline_path)
except Exception as exc:
    st.error(f"Could not load Analysis 1 reference data: {exc}")
    st.stop()

show_baseline_summary(baseline_path, "Gasperini")
st.caption("Supporting data: " + Path(pair_table_path).name + " · " + Path(detection_path).name)

elements = available_gasperini_elements(reference)
if not elements:
    st.error("No eligible enhancer elements were found after excluding NTC/TSS/positive controls.")
    st.stop()

st.sidebar.header("Analysis settings")
window_kb = st.sidebar.number_input(
    "Genomic window (kb)", min_value=10, max_value=2000, value=1000, step=50,
    help="Include genes no farther than this distance from the selected locus."
)
direction = st.sidebar.radio("Assumed direction", ["decrease", "increase"], horizontal=True, help="The expected effect direction. Decreases use the established calibration; increases use the provisional 10-replicate calibration.")
effect_percent = st.sidebar.slider("Assumed expression change (%)", min_value=1, max_value=90, value=50, step=1, help="The effect size assumed for every evaluated nearby gene.")
target_power_percent = st.sidebar.slider(
    "Target power (%)", min_value=50, max_value=99, value=80, step=1,
    help="Minimum probability of detecting the assumed effect required to call a gene powered."
)
cells_per_guide = st.sidebar.number_input(
    "Cells per guide", min_value=5, max_value=20000, value=200, step=25,
    help="Expected number of quality-controlled cells carrying each guide."
)
expected_counts_per_cell = st.sidebar.number_input(
    "Expected counts per cell",
    min_value=1,
    max_value=max(100_000, int(round(reference_depth * 10))),
    value=max(1, int(round(reference_depth))),
    step=max(1, int(round(reference_depth / 100))),
    help=(
        "Expected total modeled UMI/count depth per quality-controlled cell. "
        f"The fitted Gasperini reference is approximately {reference_depth:,.0f} counts/cell."
    ),
)
min_detection_percent = st.sidebar.slider(
    "Minimum cells detecting gene (%)", min_value=0, max_value=100, value=0, step=1,
    help="Exclude genes detected in fewer than this percentage of reference cells."
)

st.sidebar.caption("Depth is modeled UMI/counts per cell, not sequencing reads. The default is the fitted reference depth.")
st.sidebar.caption("The default 50% change is a stronger assumed effect. It does not guarantee statistical significance.")

target_power = target_power_percent / 100.0
depth_multiplier = counts_to_depth_multiplier(expected_counts_per_cell, reference_depth)
st.subheader("Dataset overview")
st.caption('Candidate cis pairs passing the genomic-window and detection filters. Power estimates are analytical; adequately powered pairs are not observed significant hits.')
overview_settings = dict(window_bp=float(window_kb) * 1000, effect_percent=float(effect_percent),
    cells_per_guide=int(cells_per_guide), depth_multiplier=float(depth_multiplier),
    target_power=target_power, min_detection_fraction=min_detection_percent / 100,
    direction=direction)
overview_calibration = select_matching_power_calibration(calibrations=_CALIBRATIONS,
    dataset='Gasperini', mode='cis', direction=direction, locus_type='enhancer',
    effect_percent=float(effect_percent), cells_per_guide=int(cells_per_guide),
    depth_multiplier=float(depth_multiplier))
overview_result = evaluate_cis_dataset(reference, elements, **overview_settings,
                                     critical_z=overview_calibration.critical_z)
overview_summary = summarize_cis_power(overview_result, target_power)
for column, label, value in zip(st.columns(4),
    ['Eligible enhancers', 'Candidate cis pairs', 'Genes represented', 'Fraction of pairs powered'],
    [overview_result.element.nunique(), len(overview_result), overview_result.gene.nunique(),
     '—' if not np.isfinite(overview_summary['fraction_powered']) else f"{overview_summary['fraction_powered']:.1%}"]):
    column.metric(label, value)
st.caption(overview_calibration.message)
if not overview_result.empty:
    overview_panels = st.columns(3, gap='large')
    design_cells = [50, 100, 200, 500, 1000, 2000]
    design_effects = [10, 25, 50, 75]
    for panel, values, setting, xlabel in [
        (overview_panels[0], design_cells, 'cells_per_guide', 'Cells per guide'),
        (overview_panels[2], design_effects, 'effect_percent', 'Assumed expression change (%)')]:
        fractions = []
        for value in values:
            settings = {**overview_settings, setting: value}
            calibration = select_matching_power_calibration(calibrations=_CALIBRATIONS,
                dataset='Gasperini', mode='cis', direction=direction, locus_type='enhancer',
                effect_percent=float(settings['effect_percent']),
                cells_per_guide=int(settings['cells_per_guide']), depth_multiplier=float(depth_multiplier))
            evaluated = evaluate_cis_dataset(reference, elements, **settings, critical_z=calibration.critical_z)
            fractions.append(summarize_cis_power(evaluated, target_power)['fraction_powered'])
        with panel:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            ax.plot(values, fractions, marker='o')
            ax.set(xlabel=xlabel, ylabel='Fraction of cis pairs powered', ylim=(0, 1))
            st.pyplot(fig, clear_figure=True)
    with overview_panels[1]:
        from utils_analysis.analysis3 import summarize_power_by_expression_decile
        expression_bins = summarize_power_by_expression_decile(overview_result.rename(columns={
            'reference_expression': 'baseline_expression', 'analytical_power': 'empirical_power'}), target_power)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(expression_bins['expression_bin'].astype(int), expression_bins['fraction_powered'], marker='o')
        ax.set(xlabel='Expression quantile (low → high)', ylabel='Fraction of cis pairs powered', ylim=(0, 1))
        st.pyplot(fig, clear_figure=True)
    st.caption('Cells and effect curves recalculate the supported calibration at each design. Effect changes are shown as percentages; the model converts them to signed log fold changes. Expression bins keep tied values together.')
    with st.expander('Dataset overview table and download'):
        st.download_button('Download all candidate cis-pair estimates (CSV)', overview_result.to_csv(index=False).encode(), 'analysis1_candidate_cis_pairs.csv', 'text/csv')
        st.dataframe(overview_result[['element', 'gene_symbol', 'reference_expression', 'analytical_power', 'powered']], hide_index=True, width="stretch")

st.subheader('Gene-level analysis')
if not overview_result.empty:
    gene_summary = overview_result.groupby(['gene', 'gene_symbol'], as_index=False).agg(
        candidate_pairs=('element', 'nunique'), adequately_powered_pairs=('powered', 'sum'))
    with st.expander('Candidate and powered pairs by gene', expanded=False):
        st.dataframe(gene_summary.rename(columns={'gene': 'Stable gene ID', 'gene_symbol': 'Gene',
            'candidate_pairs': 'Candidate enhancer pairs', 'adequately_powered_pairs': 'Pairs reaching target power'}), hide_index=True, width="stretch")
        st.caption('Observed significant-pair counts require observed pair-level FDR results. They cannot be inferred from predicted power and are not included in the bundled annotations.')

    with st.expander('Add observed significant-pair counts (optional)'):
        st.caption('Upload a compact CSV with element, gene (stable gene ID), and empirical_fdr. One row per pair, no duplicate pairs. Unmatched pairs are excluded. A zero significant count with zero observed pairs means no results were supplied for that gene.')
        observed_file = st.file_uploader('Observed cis-pair results', type=['csv'])
        observed_threshold = st.number_input('Observed FDR threshold', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        if observed_file is not None:
            try:
                from services.observed_pairs import observed_significance_by_gene
                observed = pd.read_csv(observed_file)
                counts = observed_significance_by_gene(overview_result, observed, observed_threshold)
                st.dataframe(gene_summary[['gene', 'gene_symbol']].merge(counts, on='gene').rename(columns={
                    'gene': 'Stable gene ID', 'gene_symbol': 'Gene', 'observed_pairs': 'Pairs with observed results',
                    'significant_pairs': 'Observed significant pairs'}), hide_index=True, width="stretch")
                n_matched = int(counts.observed_pairs.sum())
                st.caption(f'{n_matched:,} of {len(observed):,} uploaded pairs matched the current candidate filters. Significance uses the uploaded FDR values; the app does not rerun a significance test.')
            except (ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
                st.error(f'Cannot use observed pair results: {exc}')

    gene_labels = dict(zip(gene_summary.gene.astype(str), gene_summary.gene_symbol.astype(str)))
    overview_gene = st.selectbox('Gene to inspect', gene_summary.gene.astype(str).tolist(),
                               format_func=lambda g: f"{gene_labels[g]} ({g})")
    elements = sorted(overview_result.loc[overview_result.gene.astype(str).eq(overview_gene), 'element'].astype(str).unique())
    st.caption('The enhancer picker below now lists candidate pairs for this gene. The nearby-gene table also shows other targets around the selected enhancer.')
st.subheader('Pair / locus-level analysis')
locus_type = st.radio("Locus type", ["Enhancer", "Variant (SNP position)"], horizontal=True, help="Enhancers use measured Gasperini loci. A variant uses only its genomic position and is not empirically calibrated.")
if locus_type == "Enhancer":
    element = st.selectbox("Enhancer / element", options=elements, help="The perturbed Gasperini enhancer whose nearby genes will be evaluated.")
else:
    st.caption("Location only: a variant has no measured effect in this screen.")
    exemplar = reference.loc[pd.to_numeric(reference["gene_position"], errors="coerce").notna()].iloc[0]
    chromosome = st.text_input("Variant chromosome", value=str(exemplar["gene_chr"]), help="Chromosome containing the hypothetical variant.")
    position = st.number_input("Variant position (1-based)", min_value=1, value=int(exemplar["gene_position"]), step=1000, help="One-based genomic coordinate in the selected genome build.")
    genome_build = os.environ.get("CORTADO_GASPERINI_GENOME_BUILD", "")
    if not genome_build:
        st.warning("Set CORTADO_GASPERINI_GENOME_BUILD to the build used for the Gasperini gene coordinates (for example GRCh38) before using variant positions.")
        st.stop()
    selected_build = st.selectbox("Variant genome build", ["GRCh37", "GRCh38"], index=["GRCh37", "GRCh38"].index(genome_build) if genome_build in {"GRCh37", "GRCh38"} else 0, help="Must match the genome build used for the reference gene coordinates.")
    if selected_build != genome_build:
        st.warning(f"Variant build {selected_build} differs from reference build {genome_build}.")
        st.stop()
    element = f"chr{str(chromosome).removeprefix('chr')}:{position}"
if locus_type != "Enhancer":
    reference = select_variant_nearby(
        reference, chromosome, int(position), int(window_kb * 1000),
        reference_build=genome_build, variant_build=selected_build,
    )
calibration_selection = select_matching_power_calibration(
    calibrations=_CALIBRATIONS,
    dataset="Gasperini",
    mode="cis",
    direction=direction,
    locus_type="enhancer" if locus_type == "Enhancer" else "variant",
    effect_percent=float(effect_percent),
    cells_per_guide=int(cells_per_guide),
    depth_multiplier=float(depth_multiplier),
)
result = evaluate_cis_element(
    reference=reference,
    element=element,
    window_bp=float(window_kb) * 1000.0,
    effect_percent=float(effect_percent),
    cells_per_guide=int(cells_per_guide),
    depth_multiplier=float(depth_multiplier),
    target_power=target_power,
    min_detection_fraction=min_detection_percent / 100.0,
    direction=direction,
    critical_z=calibration_selection.critical_z,
)
summary = summarize_cis_power(result, target_power)

if result.empty:
    st.info(
        f"No genes for {element} pass the {window_kb:,} kb window and the current "
        "expression-detection filter."
    )
    st.stop()

headline = (
    f"{summary['n_powered']} of {summary['n_evaluable']} evaluable genes within "
    f"{window_kb:,} kb are expected to reach ≥{target_power_percent}% power "
    f"for an assumed {effect_percent}% {direction}."
)
st.subheader(headline)
if calibration_selection.status == "validated":
    st.success(calibration_selection.message)
elif calibration_selection.status == "extrapolated":
    st.warning(calibration_selection.message)
else:
    st.info(calibration_selection.message)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Nearby genes", summary["n_nearby"])
m2.metric("Evaluable genes", summary["n_evaluable"])
m3.metric("Powered genes", summary["n_powered"])
m4.metric(
    "Fraction powered",
    "—" if not np.isfinite(summary["fraction_powered"]) else f"{summary['fraction_powered'] * 100:.1f}%",
)
if summary["n_excluded_invalid"]:
    st.caption(
        f"{summary['n_excluded_invalid']} nearby gene(s) lacked valid reference-expression/dispersion "
        "parameters and were excluded from the power denominator."
    )

display = result.copy()
display["distance_kb"] = display["distance_bp"] / 1000.0
display["detected_percent"] = 100 * display["fraction_cells_detected"]
display["power_percent"] = 100 * display["analytical_power"]
display["raw_power_percent"] = 100 * display["raw_analytical_power"]
display["status"] = np.where(display["powered"], "Powered", "Below target")
show_cols = [
    "gene_symbol",
    "distance_kb",
    "reference_expression",
    "detected_percent",
    "power_percent",
    "status",
]
show = display[show_cols].rename(
    columns={
        "gene_symbol": "Gene",
        "gene": "Ensembl ID",
        "distance_kb": "Distance (kb)",
        "reference_expression": "Reference expression (expected counts/cell at 1× depth)",
        "n_cells_detected": "Cells detected",
        "detected_percent": "Detected in cells (%)",
        "theta": "Theta",
        "power_percent": "Analytical power (%)",
        "status": "Status",
    }
)
st.markdown("### How design changes power")
curve_left, curve_right = st.columns(2, gap="large")

cell_grid = np.unique(
    np.clip(
        np.geomspace(max(5, cells_per_guide / 8), max(cells_per_guide * 8, 50), 14).round().astype(int),
        5,
        50000,
    )
)
cells_curve = fraction_powered_vs_cells(
    reference,
    element,
    float(window_kb) * 1000.0,
    float(effect_percent),
    cell_grid,
    float(depth_multiplier),
    target_power,
    min_detection_percent / 100.0,
    direction,
    calibration_selection.critical_z,
)

depth_grid = np.unique(np.round(np.geomspace(0.25, 4.0, 14), 2))
depth_curve = fraction_powered_vs_depth(
    reference,
    element,
    float(window_kb) * 1000.0,
    float(effect_percent),
    int(cells_per_guide),
    depth_grid,
    target_power,
    min_detection_percent / 100.0,
    direction,
    calibration_selection.critical_z,
)

with curve_left:
    st.pyplot(
        _plot_fraction_curve(cells_curve, "cells_per_guide", "Cells per guide", target_power, reference_depth),
        width="stretch",
    )
    st.caption(f"Fraction of nearby genes meeting the power target as cells per guide change; sequencing depth is held at {expected_counts_per_cell:,.0f} expected counts per cell.")
with curve_right:
    st.pyplot(
        _plot_fraction_curve(depth_curve, "depth_multiplier", "Expected counts per cell", target_power, reference_depth),
        width="stretch",
    )
    st.caption(f"Fraction of nearby genes meeting the power target as sequencing depth changes; cells per guide are held at {cells_per_guide:,}.")

st.markdown("### One nearby gene in detail")
st.caption("Reference expression is the model's expected counts per cell at reference sequencing depth, not TPM or a measured wild-type mean. These curves assume the selected change occurs.")
labels = dict(zip(result["gene"].astype(str), result["gene_symbol"].astype(str)))
detail_genes = result['gene'].astype(str).tolist()
preferred_gene = locals().get('overview_gene')
gene_id = st.selectbox("Gene", detail_genes,
    index=detail_genes.index(preferred_gene) if preferred_gene in detail_genes else 0,
    key=f"cis_detail_gene_{element}_{preferred_gene}",
    format_func=lambda g: f"{labels[g]} ({g})", help="Choose one nearby gene to inspect its power curve in detail. Defaults to the gene chosen above when eligible.")
gene_cells, gene_depths = gene_power_curves(
    reference, element, gene_id, cell_grid, depth_grid, float(effect_percent), direction,
    int(cells_per_guide), float(depth_multiplier),
    calibration_selection.critical_z,
)
left_gene, right_gene = st.columns(2)
for container, curve, x, label, selected in [
    (left_gene, gene_cells, "cells_per_guide", "Cells per guide", cells_per_guide),
    (right_gene, gene_depths, "depth_multiplier", "Expected counts per cell", expected_counts_per_cell),
]:
    with container:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        x_values = reference_depth * curve[x] if x == "depth_multiplier" else curve[x]
        ax.plot(x_values, 100 * curve["analytical_power"], marker="o")
        ax.axvline(selected, color="darkorange", linestyle="--", label="Current design")
        ax.axhline(target_power_percent, color="gray", linestyle=":", label="Target power")
        ax.set(xlabel=label, ylabel="Analytical power (%)", ylim=(0, 100))
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        varied = "cells per guide" if x == "cells_per_guide" else "expected counts per cell"
        held = f"sequencing depth is held at {expected_counts_per_cell:,.0f} counts/cell" if x == "cells_per_guide" else f"cells per guide are held at {cells_per_guide:,}"
        st.caption(f"Displayed power for {labels[gene_id]} as {varied} changes; {held}. Dashed and dotted lines mark the current design and target power.")

st.markdown('### Tables and supporting details')
with st.expander('Nearby-gene results and download', expanded=False):
    st.dataframe(
        show.style.format(
            {
                "Distance (kb)": "{:.1f}",
                "Reference expression (expected counts/cell at 1× depth)": "{:.3f}",
                "Detected in cells (%)": "{:.1f}",
                "Theta": "{:.3f}",
                "Analytical power (%)": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
        height=min(600, 80 + 35 * len(show)),
    )
with st.expander('Gene identifiers and model details'):
    st.dataframe(display[['gene_symbol', 'gene', 'n_cells_detected', 'theta']].rename(columns={'gene_symbol': 'Gene', 'gene': 'Stable gene ID', 'n_cells_detected': 'Cells detected', 'theta': 'NB dispersion (θ)'}), width="stretch", hide_index=True)
st.download_button(
    "Download gene-level results (CSV)",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name=f"analysis1_{element.replace(':', '_')}_cis_power.csv",
    mime="text/csv",
)
with st.expander("Raw analytical estimate and calibration details", expanded=False):
    st.markdown(
        "The displayed estimate applies a matching calibration when available. "
        "Matching fitted values were learned on other "
        "enhancer–gene pairs and evaluated on held-out pairs. The increase model is provisional "
        "and uses 10 simulation replicates per design."
    )
    comparison = display[["gene_symbol", "gene", "raw_power_percent", "power_percent"]].rename(
        columns={
            "gene_symbol": "Gene",
            "gene": "Ensembl ID",
            "raw_power_percent": "Raw analytical power (%)",
            "power_percent": "Displayed power (%)",
        }
    )
    st.dataframe(
        comparison.style.format({"Raw analytical power (%)": "{:.1f}", "Displayed power (%)": "{:.1f}"}),
        width="stretch",
        hide_index=True,
    )
