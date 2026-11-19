"""Analysis 2 — power for a predefined downstream target-gene program."""

from __future__ import annotations

from io import StringIO
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from services.interface import show_navigation, show_method_status

from utils_analysis.analysis2 import (
    available_builtin_tfs,
    baseline_expression_overview,
    evaluate_target_program,
    fraction_powered_program_vs_cells,
    fraction_powered_program_vs_depth,
    match_target_genes,
    parse_target_gene_table,
    parse_target_gene_text,
    select_builtin_target_program,
    summarize_target_program,
    summarize_power_by_expression_bins,
)
from utils_analysis.collectri import collectri_tfs, collectri_targets
from utils_analysis.reference_data import (
    counts_to_depth_multiplier,
    load_gene_reference,
    reference_depth_from_gene_reference,
)
from services.baseline_catalog import (
    baseline_catalog_signature,
    discover_baselines,
    order_baseline_choices,
    source_signature,
)
from services.dataset_labels import dataset_label
from services.interface import show_baseline_summary, show_analysis_support
from services.file_selection import choose_file, file_label
from services.power_calibration import load_power_calibration, select_matching_power_calibration


st.set_page_config(page_title="PerTurbo Target-Program Power", layout="wide")

_APP_ROOT = Path(__file__).resolve().parent.parent
_DATASETS = {
    "Replogle": os.environ.get(
        "CORTADO_REPLOGLE_BASELINE",
        str(_APP_ROOT / "baselines" / "replogle_baseline_params.npz"),
    ),
    "Gasperini": os.environ.get(
        "CORTADO_GASPERINI_BASELINE",
        str(_APP_ROOT / "baselines" / "gasperini_baseline_params.npz"),
    ),
}

_COLLECTRI_PATH = _APP_ROOT / "data" / "collectri_programs.tsv"

_BUNDLED_K562_PROGRAMS = _APP_ROOT / "data" / "K562_M2Kb_replogle_programs.parquet"
_REPO_K562_PROGRAMS = (
    _APP_ROOT.parent
    / "benchmarks"
    / "replogle"
    / "resources"
    / "k562_m2kb"
    / "chip_seq_regulons"
    / "data"
    / "regulons"
    / "K562_M2Kb_replogle_programs.parquet"
)
_DEFAULT_K562_PROGRAMS = (
    _BUNDLED_K562_PROGRAMS if _BUNDLED_K562_PROGRAMS.exists() else _REPO_K562_PROGRAMS
)
_K562_PROGRAMS_PATH = os.environ.get(
    "CORTADO_K562_M2KB_PROGRAMS",
    str(_DEFAULT_K562_PROGRAMS),
)
_CALIBRATIONS = (
    load_power_calibration(
        _APP_ROOT / "artifacts" / "replogle_trans_decrease_power_calibration_10rep.json"
    ),
)


@st.cache_data(show_spinner=False)
def _load_gene_reference(path: str, signature: tuple) -> pd.DataFrame:
    return load_gene_reference(path)


@st.cache_data(show_spinner=False)
def _load_builtin_programs(path: str, signature: tuple) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    # Validate the schema immediately so a stale artifact fails clearly.
    available_builtin_tfs(frame)
    return frame


@st.cache_data(show_spinner=False)
def _load_collectri(path: str, signature: tuple) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


@st.cache_data(show_spinner=False)
def _discover_baselines_cached(root: str, signature: tuple) -> list[Path]:
    return discover_baselines(Path(root))


def _plot_fraction_curve(
    df: pd.DataFrame, x: str, xlabel: str, target_power: float, reference_depth: float
):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x_values = reference_depth * df[x] if x == "depth_multiplier" else df[x]
    ax.plot(x_values, 100 * df["fraction_powered"], marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Target genes reaching ≥{target_power * 100:.0f}% power (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _plot_expression_power(result: pd.DataFrame, target_power: float, reference_expression=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = pd.to_numeric(result["reference_expression"], errors="coerce")
    y = 100 * pd.to_numeric(result["analytical_power"], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    ax.scatter(x[valid], y[valid], alpha=0.75)
    ax.axhline(target_power * 100, linestyle="--", linewidth=1.2)
    if reference_expression is not None and reference_expression > 0:
        ax.axvline(reference_expression, color='darkorange', linestyle='--', label='Selected gene reference expression')
        ax.legend()
    if valid.any():
        ax.set_xscale("log")
    ax.set_xlabel("Reference expression, exp(beta_0)")
    ax.set_ylabel("Analytical power (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _read_uploaded_targets(uploaded) -> list[str]:
    raw = uploaded.getvalue().decode("utf-8-sig")
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".txt":
        return parse_target_gene_text(raw)

    sep = "\t" if suffix == ".tsv" else ","
    try:
        frame = pd.read_csv(StringIO(raw), sep=sep)
    except Exception:
        return parse_target_gene_text(raw)

    preferred = {"target_gene", "gene", "gene_id", "gene_symbol", "target"}
    if any(str(c).lower() in preferred for c in frame.columns):
        return parse_target_gene_table(frame)

    # No recognizable header: preserve the first line as data.
    try:
        no_header = pd.read_csv(StringIO(raw), sep=sep, header=None)
        return parse_target_gene_table(no_header)
    except Exception:
        return parse_target_gene_text(raw)


st.title("Analysis 2 — Target-Program Power")
st.caption(
    "Estimate how much of a predefined downstream gene set can be detected for a selected perturbation, "
    "and how coverage changes with cells per guide and sequencing depth."
)

with st.expander("What this analysis means", expanded=False):
    st.markdown(
        """
        Choose a curated CollecTRI regulon, a precomputed K562 ChIP-seq target program, or supply plausible downstream
        targets of a perturbation. The app does **not infer a GRN**. It asks whether those genes are sufficiently
        measurable under the chosen design.

        Power uses a **fast analytical negative-binomial/Wald approximation**, the same interactive
        engine used in Analysis 1. For the Replogle trans-decrease setting, its decision threshold is
        provisionally corrected against repeated Perturbo simulations. Other datasets use the raw approximation.
        """
    )

show_navigation()
show_method_status()

st.header("Dataset")
dataset_options = list(_DATASETS) + ["Custom"]
active_dataset = st.session_state.get("gpu_dataset", "Replogle")
dataset = st.selectbox(
    "Baseline dataset",
    options=dataset_options,
    index=dataset_options.index(active_dataset) if active_dataset in dataset_options else 0,
    format_func=dataset_label,
    help="Select the fitted baseline model that supplies gene expression and dispersion values.",
)
with st.popover('Change baseline file'):
    with st.expander("Advanced: baseline folder"):
        data_folder = Path(st.text_input("Server baseline folder", value=os.environ.get("CORTADO_DATA_ROOT", str(_APP_ROOT / "baselines")), help="Folder containing fitted baseline parameter files.")).expanduser()
    found_baselines = _discover_baselines_cached(
        str(data_folder), baseline_catalog_signature(data_folder)
    )
    initial_path = st.session_state.get("gpu_baseline_path") if dataset == active_dataset else None
    default_path = _DATASETS.get(dataset, "")
    default_choice = Path(default_path) if default_path and Path(default_path).exists() else None
    choices = order_baseline_choices(
        preferred=Path(initial_path) if initial_path else default_choice,
        discovered=found_baselines,
        default=default_choice,
    )
    if choices:
        baseline_path = str(st.selectbox("Baseline file in folder", choices, key=f"program_file_{dataset}", format_func=file_label, help="The baseline parameter file used for this calculation."))
    else:
        st.info("No baseline found in this folder. Use Data Setup to fit/upload data or select another folder.")
        baseline_path = default_path
    baseline_path = choose_file("Baseline parameters", baseline_path, {".npz"}, _APP_ROOT / "baselines" / "uploads", key=f"program_baseline_{dataset}")

if not Path(baseline_path).exists():
    st.warning(
        f"Baseline parameters were not found at `{baseline_path}`. Choose a server folder or use Data Setup."
    )
    st.stop()

try:
    gene_reference = _load_gene_reference(baseline_path, source_signature(Path(baseline_path)))
    reference_depth = reference_depth_from_gene_reference(gene_reference)
except Exception as exc:
    st.error(f"Could not load baseline gene parameters: {exc}")
    st.stop()

show_baseline_summary(baseline_path, dataset, n_genes=len(gene_reference))
if dataset == 'Custom':
    show_analysis_support(dataset)
with st.expander('Inspect baseline expression distribution'):
    overview = baseline_expression_overview(gene_reference)
    intro_a, intro_b, intro_c = st.columns(3)
    intro_a.metric("Genes in baseline", f"{overview['n_genes']:,}")
    intro_b.metric("Genes with positive expression", f"{overview['n_expressed']:,}")
    intro_c.metric("Median reference expression", f"{overview['median_expression']:.2f} counts/cell")
    expression = pd.to_numeric(gene_reference["reference_expression"], errors="coerce")
    expression = expression[np.isfinite(expression) & (expression > 0)]
    if len(expression):
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.hist(np.log10(expression), bins=40, color="#457b9d")
        ax.set_xlabel("log₁₀ reference expression (expected counts/cell at reference depth)")
        ax.set_ylabel("Number of genes")
        st.pyplot(fig, clear_figure=True)
    st.caption("Distribution of expected gene counts in the selected baseline. Genes farther left have lower expression and generally require more cells or sequencing depth to detect an effect.")

st.sidebar.header("Analysis settings")
effect_percent = st.sidebar.slider(
    "Assumed expression decrease (%)", min_value=1, max_value=90, value=50, step=1,
    help="50% decrease means half the reference expression (fold change 0.5); this is an assumed effect, not a significance claim."
)
target_power_percent = st.sidebar.slider(
    "Target power (%)", min_value=50, max_value=99, value=80, step=1,
    help="Minimum detection probability required to call a target gene powered."
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
        f"The selected baseline reference is approximately {reference_depth:,.0f} counts/cell."
    ),
)
st.sidebar.caption("Depth is modeled UMI/counts per cell, not sequencing reads. The default is the fitted reference depth.")
st.sidebar.caption("The default 50% decrease is a stronger starting assumption; adjust it for your study.")
depth_multiplier = counts_to_depth_multiplier(expected_counts_per_cell, reference_depth)
min_reference_expression = st.sidebar.number_input(
    "Minimum reference expression", min_value=0.0, value=0.1, step=0.05, format="%.3f",
    help="Optional filter on exp(beta_0). Genes below this value are excluded after identifier matching."
)

st.markdown("### Target program")

builtin_path = Path(_K562_PROGRAMS_PATH)
builtin_available = dataset == "Replogle" and builtin_path.exists()

if dataset == "Replogle" and not builtin_available:
    st.warning(
        "The precomputed K562 TF-program resource was not found, so only custom target lists are available. "
        "Set `CORTADO_K562_M2KB_PROGRAMS` to `K562_M2Kb_replogle_programs.parquet` to enable built-in programs."
    )

source_options = ["Custom gene list"]
if _COLLECTRI_PATH.exists():
    source_options.insert(0, "CollecTRI curated regulon")
if builtin_available:
    source_options.insert(0, "K562 built-in TF program")
program_source = st.radio("Target program source", source_options, horizontal=True, help="Use a bundled TF target set or provide your own list of expected downstream genes.")

program_metadata = None
if program_source == "CollecTRI curated regulon":
    programs = _load_collectri(str(_COLLECTRI_PATH), source_signature(_COLLECTRI_PATH))
    tfs = collectri_tfs(programs)
    perturbation = st.selectbox(
        "CollecTRI transcription factor / complex", tfs,
        index=tfs.index("GATA1") if "GATA1" in tfs else 0,
        help="Regulators are not restricted to perturbations present in the selected screen.",
    )
    targets = collectri_targets(programs, perturbation)
    st.caption(
        "Human CollecTRI from OmniPath, downloaded 28 October 2026. Unique target symbols; "
        "self-targets excluded; complexes retained as named regulons. "
        "Evidence and regulatory signs are not filtered. "
        "These curated interactions may be active in different cellular contexts."
    )
    st.info(
        "Power uses the selected baseline and assumed decrease. It does not establish that this "
        "TF is expressed, was perturbed, or regulates every listed target in this dataset."
    )
    st.download_button(
        "Download selected regulon (CSV)",
        pd.DataFrame({"tf": perturbation, "target_gene": targets}).to_csv(index=False).encode("utf-8"),
        file_name=f"collectri_{perturbation}_targets.csv", mime="text/csv",
    )
elif program_source == "K562 built-in TF program":
    try:
        builtin_programs = _load_builtin_programs(str(builtin_path), source_signature(builtin_path))
    except Exception as exc:
        st.error(f"Could not load the K562 TF-program resource: {exc}")
        st.stop()

    tfs = available_builtin_tfs(builtin_programs)
    if not tfs:
        st.error("The K562 TF-program resource contains no TF programs.")
        st.stop()

    default_tf_index = tfs.index("GATA1") if "GATA1" in tfs else 0
    perturbation = st.selectbox(
        "K562 transcription factor",
        options=tfs,
        index=default_tf_index,
        help="Only TF perturbations present in both the K562 ReMap regulon and the Replogle dataset are shown.",
    )
    min_support = st.number_input(
        "Minimum supporting ChIP peaks", min_value=1, value=1, step=1,
        help="1 preserves the original M2Kb program. Higher counts apply a stricter support filter; peak count is not a calibrated probability of regulation.")
    selected_program = select_builtin_target_program(
        builtin_programs, tf=perturbation, min_supporting_peaks=min_support
    )
    full_program = select_builtin_target_program(
        builtin_programs, tf=perturbation, min_supporting_peaks=1
    )
    with st.expander("Inspect peak support before filtering"):
        st.bar_chart(full_program['n_supporting_peaks'].value_counts().sort_index().rename('Number of targets'))
        st.caption('Horizontal axis: supporting peak count. Vertical axis: target genes. Increase the minimum to retain genes with more peak support.')
    targets = selected_program["ensembl_gene_id"].astype(str).tolist()
    program_metadata = selected_program[["ensembl_gene_id", "n_supporting_peaks"]].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("M2Kb targets", f"{len(full_program):,}")
    c2.metric("Targets used", f"{len(selected_program):,}")
    c3.metric("Minimum peak support", min_support)
    st.caption(
        "K562-specific program precomputed from ReMap2022 ChIP-seq peaks and K562 ENCODE RNA-seq; "
        "targets are restricted to genes measurable in the Replogle PerTurbo baseline."
    )
else:
    perturbation = st.text_input(
        "Perturbation / regulator label",
        value="GATA1" if dataset == "Replogle" else "enhancer_of_interest",
        help="Used to label this target program. Power depends on the supplied target genes and design settings.",
    )
    custom_source = st.radio(
        "Provide target set", ["Paste gene list", "Upload CSV/TSV/TXT"], horizontal=True,
        help="Choose how to supply the target-gene identifiers."
    )
    if custom_source == "Paste gene list":
        text = st.text_area(
            "Gene symbols or Ensembl IDs",
            value="",
            height=150,
            placeholder="GATA2\nMYC\nENSG00000102145",
            help="Separate identifiers with new lines, commas, spaces, or semicolons.",
        )
        targets = parse_target_gene_text(text)
    else:
        uploaded = st.file_uploader("Target-gene file", type=["csv", "tsv", "txt"], help="Upload gene symbols or Ensembl IDs in a text or tabular file.")
        targets = _read_uploaded_targets(uploaded) if uploaded is not None else []

if not targets:
    st.info("Provide at least one target gene to run Analysis 2.")
    st.stop()

match = match_target_genes(targets, gene_reference)
if program_metadata is not None and not match.matched.empty:
    support_map = dict(
        zip(
            program_metadata["ensembl_gene_id"].astype(str),
            program_metadata["n_supporting_peaks"].astype(int),
        )
    )
    match.matched["n_supporting_peaks"] = (
        match.matched["gene_id"].astype(str).map(support_map).astype("Int64")
    )
if match.matched.empty:
    st.error("None of the supplied genes matched the selected baseline dataset.")
    if match.unmatched:
        st.write("Unmatched:", ", ".join(match.unmatched[:50]))
    if match.ambiguous:
        st.write("Ambiguous symbols:", ", ".join(match.ambiguous[:50]))
    st.stop()

target_power = target_power_percent / 100.0
calibration_selection = select_matching_power_calibration(
    calibrations=_CALIBRATIONS,
    dataset=dataset,
    mode="trans",
    direction="decrease",
    locus_type="target_program",
    effect_percent=float(effect_percent),
    cells_per_guide=int(cells_per_guide),
    depth_multiplier=float(depth_multiplier),
)
result = evaluate_target_program(
    match.matched,
    effect_percent=float(effect_percent),
    cells_per_guide=int(cells_per_guide),
    depth_multiplier=float(depth_multiplier),
    target_power=target_power,
    min_reference_expression=float(min_reference_expression),
    critical_z=calibration_selection.critical_z,
)
summary = summarize_target_program(
    result,
    target_power,
    n_requested=len(targets),
    n_matched_before_filter=len(match.matched),
    n_unmatched=len(match.unmatched),
    n_ambiguous=len(match.ambiguous),
)

st.subheader(
    f"{summary['n_powered']} of {summary['n_evaluable']} evaluable targets are expected to reach "
    f"≥{target_power_percent}% power for a {effect_percent}% decrease."
)
program_label = perturbation or "unnamed perturbation"
if program_source == "K562 built-in TF program":
    program_label += f" · K562 M2Kb · support ≥{min_support}"
if program_source == "CollecTRI curated regulon":
    program_label += " · CollecTRI"
st.caption(f"Target program: {program_label} · {dataset} baseline")
if calibration_selection.status == "validated":
    st.success(calibration_selection.message)
elif calibration_selection.status == "extrapolated":
    st.warning(calibration_selection.message)
else:
    st.info(calibration_selection.message)
st.caption(f'Assumes every target decreases by {effect_percent}%; predicted detectability, not the fraction of genes expected to respond biologically.')

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Requested", summary["n_requested"])
m2.metric("Matched", summary["n_matched_before_filter"])
m3.metric("After expression filter", summary["n_after_expression_filter"])
m4.metric("Powered", summary["n_powered"])
m5.metric(
    "Fraction powered",
    "—" if not np.isfinite(summary["fraction_powered"]) else f"{summary['fraction_powered'] * 100:.1f}%",
)

if program_source == "CollecTRI curated regulon":
    st.caption(
        f"Whole-regulon coverage: {summary['n_powered']} / {len(targets)} curated targets "
        f"({100 * summary['n_powered'] / len(targets):.1f}%) reach the power target. "
        "This denominator keeps unmatched, ambiguous and expression-filtered targets as not powered; "
        "the Fraction powered metric above uses only evaluable targets."
    )

if match.unmatched or match.ambiguous:
    with st.expander("Identifier matching details", expanded=False):
        if match.unmatched:
            st.markdown(f"**Unmatched ({len(match.unmatched)}):** " + ", ".join(match.unmatched))
        if match.ambiguous:
            st.markdown(
                f"**Ambiguous symbols ({len(match.ambiguous)}):** " + ", ".join(match.ambiguous)
                + "  \nUse stable Ensembl IDs for these genes so the app does not silently choose among duplicate symbols."
            )

if result.empty:
    st.info("All matched genes were removed by the minimum-reference-expression filter.")
    st.stop()

display = result.copy()
display["power_percent"] = 100 * display["analytical_power"]
display["raw_power_percent"] = 100 * display["raw_analytical_power"]
display["status"] = np.where(display["powered"], "Powered", "Below target")
display_columns = [
    "gene_symbol", "reference_expression"
]
display_columns += ["power_percent", "status"]
show = display[display_columns].rename(
    columns={
        "query": "Input",
        "gene_symbol": "Gene symbol",
        "gene_id": "Stable gene ID",
        "match_source": "Matched by",
        "reference_expression": "Reference expression",
        "theta": "NB dispersion (θ)",
        "n_supporting_peaks": "Supporting ChIP peaks",
        "power_percent": "Analytical power (%)",
        "status": "Status",
    }
)
st.markdown("### What limits target-program coverage?")
gene_labels = dict(zip(result['gene_id'].astype(str), result['gene_symbol'].astype(str)))
expression_gene = st.selectbox('Gene to mark on expression plot', result['gene_id'].astype(str).tolist(), format_func=lambda g: f"{gene_labels[g]} ({g})")
marked_expression = float(result.loc[result['gene_id'].astype(str).eq(expression_gene), 'reference_expression'].iloc[0])
left, right = st.columns(2, gap="large")
with left:
    st.pyplot(_plot_expression_power(result, target_power, marked_expression), width="stretch")
    st.caption("Each point is one target gene. The plot shows how predicted power changes with baseline expression; the horizontal line is the selected power target. The dashed orange line marks the selected gene’s fitted reference expression, not an observed WT mean.")

cell_grid = np.unique(
    np.clip(
        np.geomspace(max(5, cells_per_guide / 8), max(cells_per_guide * 8, 50), 14).round().astype(int),
        5,
        50000,
    )
)
cells_curve = fraction_powered_program_vs_cells(
    match.matched,
    float(effect_percent),
    cell_grid,
    float(depth_multiplier),
    target_power,
    float(min_reference_expression),
    calibration_selection.critical_z,
)
with right:
    st.pyplot(_plot_fraction_curve(cells_curve, "cells_per_guide", "Cells per guide", target_power, reference_depth), width="stretch")
    st.caption(f"Fraction of target genes meeting the power target as cells per guide change; sequencing depth is held at {expected_counts_per_cell:,.0f} expected counts per cell.")

st.markdown("### Sequencing-depth sensitivity")
depth_grid = np.unique(np.round(np.geomspace(0.25, 4.0, 14), 2))
depth_curve = fraction_powered_program_vs_depth(
    match.matched,
    float(effect_percent),
    int(cells_per_guide),
    depth_grid,
    target_power,
    float(min_reference_expression),
    calibration_selection.critical_z,
)
st.pyplot(_plot_fraction_curve(depth_curve, "depth_multiplier", "Expected counts per cell", target_power, reference_depth), width="stretch")
st.caption(f"Fraction of target genes meeting the power target as sequencing depth changes; cells per guide are held at {cells_per_guide:,}. The matching calibration status shown above also applies to this curve.")

st.markdown('### Tables and supporting details')
with st.expander('Target-gene results and download', expanded=False):
    st.dataframe(
        show.style.format(
            {
                "Reference expression": "{:.3f}",
                "NB dispersion (θ)": "{:.3f}",
                "Analytical power (%)": "{:.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
        height=min(650, 80 + 35 * len(show)),
    )
with st.expander('Gene identifiers and model details'):
    detail_columns = ['query', 'gene_symbol', 'gene_id', 'match_source', 'theta']
    if 'n_supporting_peaks' in display.columns:
        detail_columns.append('n_supporting_peaks')
    st.dataframe(display[detail_columns].rename(columns={'query': 'Input', 'gene_symbol': 'Gene', 'gene_id': 'Stable gene ID', 'match_source': 'Matched by', 'theta': 'NB dispersion (θ)', 'n_supporting_peaks': 'Supporting ChIP peaks'}), width="stretch", hide_index=True)

safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in (perturbation or "program"))
st.download_button(
    "Download target-gene results (CSV)",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name=f"analysis2_{safe_label}_target_power.csv",
    mime="text/csv",
)

with st.expander("Raw analytical estimate and calibration details", expanded=False):
    comparison = display[["gene_symbol", "gene_id", "raw_power_percent", "power_percent"]].rename(
        columns={
            "gene_symbol": "Gene",
            "gene_id": "Stable gene ID",
            "raw_power_percent": "Raw analytical power (%)",
            "power_percent": "Displayed power (%)",
        }
    )
    st.dataframe(
        comparison.style.format(
            {"Raw analytical power (%)": "{:.1f}", "Displayed power (%)": "{:.1f}"}
        ),
        width="stretch",
        hide_index=True,
    )


with st.expander('Power by target expression — summary table', expanded=False):
    expression_summary = summarize_power_by_expression_bins(result, target_power)
    if not expression_summary.empty:
        expression_display = expression_summary.copy()
        expression_display["fraction_powered"] = 100 * expression_display["fraction_powered"]
        expression_display["expression_range"] = expression_display.apply(
            lambda row: f"{row['expression_min']:.3g}–{row['expression_max']:.3g}", axis=1
        )
        expression_display = expression_display[
            ["expression_group", "expression_range", "n_targets", "n_powered", "fraction_powered"]
        ].rename(
            columns={
                "expression_group": "Expression group",
                "expression_range": "Reference-expression range",
                "n_targets": "Targets",
                "n_powered": "≥ target power",
                "fraction_powered": "Fraction powered (%)",
            }
        )
        st.dataframe(
            expression_display.style.format({"Fraction powered (%)": "{:.1f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Targets are split into three equal-count groups after the expression filter. "
            "This makes the expression dependence of detectability easier to interpret."
        )
