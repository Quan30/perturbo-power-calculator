from __future__ import annotations

from services.interface import show_navigation, show_method_status, preferred_simulation_dataset, median_design_value

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils_analysis.analysis3 import (
    available_dataset_modes,
    available_pairs,
    current_design_summary,
    design_replicate_progress,
    filter_pair_power,
    fraction_powered_vs_cells,
    fraction_powered_vs_depth,
    load_analysis3_artifact,
    pair_design_curves,
    summarize_power_by_expression_decile,
)
from services.baseline_catalog import source_signature
from services.dataset_labels import dataset_label
from services.file_selection import choose_file


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT = APP_ROOT / "artifacts" / "analysis3_pair_power.parquet"
CALIBRATION_SUMMARY = APP_ROOT / "artifacts" / "calibration_pilot_summary_10rep.csv"


@st.cache_data(show_spinner=False)
def _load_pair_power(path: str, signature: tuple) -> pd.DataFrame:
    return load_analysis3_artifact(path)


def _configured_path() -> Path:
    return Path(os.environ.get("CORTADO_ANALYSIS3_PAIR_POWER", str(DEFAULT_ARTIFACT)))



def _mode_label(dataset: str, mode: str) -> str:
    if dataset.lower().startswith("replogle") and mode == "self-target":
        return "Self-target"
    return mode.replace("_", " ").title()


st.set_page_config(page_title="Analysis 3 — Screen-Wide Empirical Power", layout="wide")
st.title("Analysis 3 — Screen-Wide Empirical Power")
st.caption(
    "Repeated Perturbo simulations estimate detection probability for each fixed "
    "perturbation–gene pair. This page reads the precomputed pair-power artifact."
)

show_navigation()
show_method_status()

with st.popover('Change simulation file'):
    artifact_path = Path(choose_file("Pair-power artifact", str(_configured_path()), {".parquet", ".csv", ".tsv", ".txt"}, APP_ROOT / "data" / "uploads", key="screen_power"))

if not artifact_path.exists():
    st.info(
        "Analysis 3 needs the precomputed repeated-simulation artifact. "
        "Run the HPC preprocessing workflow first, then set "
        "`CORTADO_ANALYSIS3_PAIR_POWER` or select a file on this page.\n\n"
        f"Current path: `{artifact_path}`"
    )
    st.stop()

try:
    with st.spinner("Loading empirical pair-power artifact..."):
        power_df = _load_pair_power(str(artifact_path), source_signature(artifact_path))
except Exception as exc:
    st.error(f"Could not load Analysis 3 artifact: {exc}")
    st.stop()

combos = available_dataset_modes(power_df)
datasets = sorted({d for d, _ in combos})

st.sidebar.header("Design")
dataset = st.selectbox(
    "Dataset",
    datasets,
    index=datasets.index(preferred_simulation_dataset(datasets)),
    format_func=dataset_label,
    help="Choose which simulation dataset to summarize.",
)
if not any(str(d).lower().startswith('replogle') for d in datasets):
    st.info('Replogle is preferred when available. This simulation file contains no Replogle designs, so the available dataset is shown.')
modes = [m for d, m in combos if d == dataset]
mode = st.selectbox("Mode", modes, format_func=lambda m: _mode_label(dataset, m), help="Choose the biological pair type or analysis mode represented by the simulations.")

st.info(f"Simulation data in use · {artifact_path.name} · {dataset_label(dataset)} · {_mode_label(dataset, mode)}")
with st.expander('Simulation source details'):
    st.write('Simulation file:', str(artifact_path.resolve()))

subset = power_df.loc[
    power_df["dataset"].astype(str).eq(dataset)
    & power_df["mode"].astype(str).eq(mode)
].copy()

effects = sorted(subset["effect_percent"].unique().astype(float).tolist())
cells_values = sorted(subset["cells_per_guide"].unique().astype(int).tolist())
depth_values = sorted(subset["depth_multiplier"].unique().astype(float).tolist())

preferred_effect = 25.0 if 25.0 in effects else effects[0]
effect_percent = st.sidebar.selectbox(
    "Effect size",
    effects,
    index=effects.index(preferred_effect),
    format_func=lambda x: f"{x:g}% decrease",
    help="Assumed expression decrease used in the repeated simulations.",
)

preferred_cells = median_design_value(cells_values)
cells_per_guide = st.sidebar.selectbox(
    "Cells per guide",
    cells_values,
    index=cells_values.index(preferred_cells),
    help="Number of simulated cells carrying each guide. The default is the median available design, not the observed median guide size.",
)

preferred_depth = 1.0 if 1.0 in depth_values else depth_values[0]
depth_multiplier = st.sidebar.selectbox(
    "Relative sequencing depth",
    depth_values,
    index=depth_values.index(preferred_depth),
    format_func=lambda x: f"{100 * x:g}%",
    help="Simulated reads per cell relative to the reference dataset; 100% matches the baseline.",
)

target_power_pct = st.sidebar.slider(
    "Target empirical power (%)",
    min_value=50,
    max_value=99,
    value=80,
    step=1,
    help="Minimum empirical detection probability required to count a pair as powered.",
)
target_power = target_power_pct / 100.0
min_expression = st.sidebar.number_input(
    "Minimum baseline expression",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Exclude pairs whose target gene has lower Perturbo reference expression exp(beta_0).",
)

current = filter_pair_power(
    power_df,
    dataset=dataset,
    mode=mode,
    effect_percent=effect_percent,
    cells_per_guide=cells_per_guide,
    depth_multiplier=depth_multiplier,
    min_expression=min_expression,
)

if current.empty:
    st.warning("No precomputed pairs match this combination of settings.")
    st.stop()

summary = current_design_summary(current, target_power)
progress = design_replicate_progress(current)

st.subheader("Current design")
if progress["is_provisional"]:
    if progress["completed"] == 0:
        st.info(
            f"No completed simulations yet for this design "
            f"(0 / {progress['expected']} replicates). Its result is left blank."
        )
    else:
        st.warning(
            f"Provisional result: {progress['completed']} / {progress['expected']} "
            "simulation replicates completed for this design."
        )

metric_cols = st.columns(4)
if progress["completed"] == 0:
    metric_cols[0].metric("Eligible pairs", "—")
    metric_cols[1].metric(f"≥{target_power:.0%} power", "—")
    metric_cols[2].metric("Fraction adequately powered", "—")
    metric_cols[3].metric("Median pair power", "—")
else:
    metric_cols[0].metric("Eligible pairs", f"{summary['n_pairs']:,}")
    metric_cols[1].metric(
        f"≥{target_power:.0%} power",
        f"{summary['n_powered']:,}",
    )
    metric_cols[2].metric("Fraction adequately powered", f"{summary['fraction_powered']:.1%}")
    metric_cols[3].metric("Median pair power", f"{summary['median_empirical_power']:.1%}")

if progress["completed"] > 0:
    st.markdown(
        f"At **{cells_per_guide:,} cells/guide** and **{depth_multiplier:g}× depth**, "
        f"**{summary['fraction_powered']:.1%}** of eligible {dataset} "
        f"{_mode_label(dataset, mode).lower()} pairs have at least **{target_power:.0%} empirical power** "
        f"to detect a **{effect_percent:g}% decrease**."
    )

st.caption(
    "Empirical power is the fraction of repeated simulated screens in which the same fixed pair "
    "is detected at the simulation FDR threshold. The standardized effect is held fixed across pairs."
)

st.subheader("How design changes screen-wide sensitivity")
curve_cols = st.columns(2, gap="large")

with curve_cols[0]:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for effect in effects:
        curve = fraction_powered_vs_cells(
            power_df,
            dataset=dataset,
            mode=mode,
            effect_percent=effect,
            depth_multiplier=depth_multiplier,
            target_power=target_power,
            min_expression=min_expression,
        )
        if not curve.empty:
            ax.plot(
                curve["cells_per_guide"],
                curve["fraction_powered"],
                marker="o",
                label=f"{effect:g}%",
            )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cells per guide")
    ax.set_ylabel(f"Fraction with ≥{target_power:.0%} power")
    ax.set_title(f"At {100 * depth_multiplier:g}% relative sequencing depth")
    ax.legend(title="Effect")
    st.pyplot(fig, clear_figure=True)
    st.caption("Fraction of all evaluated pairs meeting the empirical-power target as cells per guide change; curves compare effect sizes.")

with curve_cols[1]:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for effect in effects:
        curve = fraction_powered_vs_depth(
            power_df,
            dataset=dataset,
            mode=mode,
            effect_percent=effect,
            cells_per_guide=cells_per_guide,
            target_power=target_power,
            min_expression=min_expression,
        )
        if not curve.empty:
            ax.plot(
                100 * curve["depth_multiplier"],
                curve["fraction_powered"],
                marker="o",
                label=f"{effect:g}%",
            )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Relative sequencing depth (%)")
    ax.set_ylabel(f"Fraction with ≥{target_power:.0%} power")
    ax.set_title(f"At {cells_per_guide:,} cells/guide")
    ax.legend(title="Effect")
    st.pyplot(fig, clear_figure=True)
    st.caption("Fraction of all evaluated pairs meeting the empirical-power target as sequencing depth changes; curves compare effect sizes.")

st.caption(
    "Missing points in design-comparison plots indicate conditions with no completed simulations yet; "
    "they are not treated as zero power."
)

expr_summary = pd.DataFrame()
if progress["completed"] > 0:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Distribution of per-pair empirical power")
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.hist(current["empirical_power"].dropna(), bins=np.linspace(0, 1, 21), edgecolor="white")
        ax.axvline(target_power, linestyle="--", linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Empirical power")
        ax.set_ylabel("Number of pairs")
        ax.set_title(
            f"{dataset} · {_mode_label(dataset, mode)} · {effect_percent:g}% effect"
        )
        st.pyplot(fig, clear_figure=True)
        st.caption("Distribution of empirical power across perturbation–gene pairs at the selected design. The dashed line is the target power.")

    with right:
        st.subheader("Power by baseline-expression bin")
        expr_summary = summarize_power_by_expression_decile(
            current,
            target_power,
            n_bins=10,
        )
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot(
            expr_summary["expression_bin"].astype(int),
            expr_summary["fraction_powered"],
            marker="o",
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Baseline-expression quantile (low → high)")
        ax.set_ylabel(f"Fraction with ≥{target_power:.0%} power")
        ax.set_xticks(expr_summary["expression_bin"].astype(int), expr_summary["expression_label"], rotation=35, ha="right")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption("Fraction of pairs meeting the power target in each pd.qcut expression interval. Identical values stay together, so fewer than ten bins may be shown.")


if progress["completed"] > 0:
    with st.expander("Expression-bin details", expanded=False):
        display = expr_summary.copy()
        display["fraction_powered"] = display["fraction_powered"].map(lambda x: f"{x:.1%}")
        display["median_empirical_power"] = display["median_empirical_power"].map(lambda x: f"{x:.1%}")
        st.dataframe(display, width="stretch", hide_index=True)

    with st.expander("Pair-level empirical power", expanded=False):
        cols = [
            "pair_id",
            "element",
            "gene",
            "gene_symbol",
            "baseline_expression",
            "n_detected",
            "n_replicates",
            "empirical_power",
        ]
        st.dataframe(current[cols], width="stretch", hide_index=True)
        st.download_button(
            "Download current pair-power table (CSV)",
            data=current.to_csv(index=False).encode("utf-8"),
            file_name=f"analysis3_{dataset}_{mode}_{effect_percent:g}pct.csv".replace(" ", "_"),
            mime="text/csv",
        )

with st.expander('Provisional calibration diagnostics (10 replicates)'):
    pilot_summary = pd.read_csv(CALIBRATION_SUMMARY)
    calibrated = pilot_summary.loc[
        pilot_summary["method"].eq("Calibrated cross-validated")
    ].copy()
    calibrated["Raw MAE"] = (
        calibrated.apply(
            lambda row: pilot_summary.loc[
                pilot_summary["dataset"].eq(row["dataset"])
                & pilot_summary["setting"].eq(row["setting"])
                & pilot_summary["method"].eq("Raw analytical"),
                "mae",
            ].iloc[0],
            axis=1,
        )
    )
    pilot_display = calibrated[
        ["dataset", "setting", "Raw MAE", "mae", "correlation"]
    ].rename(
        columns={
            "dataset": "Dataset",
            "setting": "Calibration scope",
            "mae": "Calibrated MAE",
            "correlation": "Correlation",
        }
    )
    st.dataframe(
        pilot_display.style.format(
            {
                "Raw MAE": "{:.1%}",
                "Calibrated MAE": "{:.1%}",
                "Correlation": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "Presentation result only: these models use 10 simulation replicates per design. "
        "Analyses 1 and 2 apply them only to their matching dataset, pair type, and direction. "
        "They will be replaced after the 30-replicate run."
    )

st.subheader("Inspect a specific perturbation–gene pair")
pairs = available_pairs(power_df, dataset, mode)
selected_element = st.selectbox("Enhancer / perturbation", pairs["element"].astype(str).drop_duplicates().tolist(), help="Choose the perturbation whose individual target-gene pairs you want to inspect.")
pair_ids = pairs.loc[pairs["element"].astype(str).eq(selected_element), "pair_id"].astype(str).tolist()
pair_info = pairs.set_index(pairs["pair_id"].astype(str))
chosen_id = st.selectbox(
    "Target gene / pair (search or scroll)",
    pair_ids,
    format_func=lambda key: f"{pair_info.loc[key, 'element']} → {pair_info.loc[key, 'gene_symbol']} ({key})",
    help="Click to search by enhancer/perturbation name, target gene, or pair ID.",
)
context = (str(artifact_path.resolve()), source_signature(artifact_path), dataset, mode)
if st.button('Load selected pair', type='primary'):
    st.session_state['screen_loaded_pair'] = (context, chosen_id)
loaded = st.session_state.get('screen_loaded_pair')
if not loaded or loaded[0] != context or loaded[1] not in pair_info.index:
    st.info('Select a perturbation and target gene, then click Load selected pair to view its details.')
    st.stop()
if loaded[1] != chosen_id:
    st.info('The selection has changed. Click Load selected pair to replace the details below.')
chosen_id = loaded[1]
one_pair = pair_info.loc[chosen_id]
st.markdown(f"**Loaded pair: {one_pair['element']} → {one_pair['gene_symbol']} ({chosen_id})**")
st.caption(f"Current simulation design: {cells_per_guide:,} cells/guide · {depth_multiplier:g}× depth · {effect_percent:g}% decrease. Element selection keeps the sidebar cell count.")
st.caption(f"Reference expression: {float(one_pair['baseline_expression']):.3g} expected counts/cell at reference depth. Effect size: {effect_percent:g}% decrease.")
selected_row = current.loc[current["pair_id"].astype(str).eq(chosen_id)]
if len(selected_row):
    row = selected_row.iloc[0]
    detail_cols = st.columns(3)
    detail_cols[0].metric("Detected / completed simulations", f"{int(row['n_detected'])} / {int(row['n_replicates'])}")
    detail_cols[1].metric("Empirical power", "—" if pd.isna(row["empirical_power"]) else f"{row['empirical_power']:.1%}")
    detail_cols[2].metric("Expected replicates", int(row.get("expected_replicates", row["n_replicates"])))
    if row["n_replicates"] < row.get("expected_replicates", row["n_replicates"]):
        st.warning("This pair's selected design has an unfinished simulation count.")
else:
    st.info("This pair is below the current expression filter or has no row at the selected design.")
pair_cells, pair_depths = pair_design_curves(
    power_df, chosen_id, dataset, mode, effect_percent, depth_multiplier, cells_per_guide
)
pair_cols = st.columns(2, gap="large")
for panel, curve, x, xlabel in [
    (pair_cols[0], pair_cells, "cells_per_guide", "Cells per guide"),
    (pair_cols[1], pair_depths, "depth_multiplier", "Relative sequencing depth (%)"),
]:
    with panel:
        fig, ax = plt.subplots(figsize=(6, 3.7))
        if not curve.empty:
            x_values = 100 * curve[x] if x == "depth_multiplier" else curve[x]
            ax.plot(x_values, curve["empirical_power"], marker="o")
        ax.axhline(target_power, linestyle="--", linewidth=1.2)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Empirical power")
        st.pyplot(fig, clear_figure=True)
        varied = "cells per guide" if x == "cells_per_guide" else "relative sequencing depth"
        st.caption(f"Empirical power for the selected perturbation–gene pair as {varied} changes; the dashed line is the target power.")
if pair_cells.empty and pair_depths.empty:
    st.info("No simulated designs for this pair at the selected effect size, depth, or cell count.")
with st.expander("Selected pair: simulated designs"):
    st.dataframe(power_df.loc[
        power_df["dataset"].astype(str).eq(dataset)
        & power_df["mode"].astype(str).eq(mode)
        & power_df["pair_id"].astype(str).eq(chosen_id)
    ], hide_index=True, width="stretch")

st.caption(
    "The plots summarize precomputed simulations only; changing a control selects a precomputed "
    "design and does not run Perturbo on the web server."
)
