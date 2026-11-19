# Bundled data

Everything the app needs is included, so it runs offline right after installation.
Raw single-cell datasets are **not** included.

## `baselines/`: fitted reference baselines

PerTurbo control-only fits (negative-binomial baseline mean `beta_0` and dispersion `theta` per gene).

| File | Screen | Genes |
|---|---|---|
| `gasperini_baseline_params.npz` | Gasperini *et al.* 2019, K562 enhancer screen, high MOI (GEO GSE120861, at-scale) | 13,135 |
| `replogle_baseline_params.npz` | Replogle *et al.* 2022, K562 Perturb-seq, low MOI (non-targeting control cells) | 8,248 |

`exp(beta_0)` is the model-expected count per cell at the baseline's reference depth. It is not
TPM and not an observed wild-type mean.

## `artifacts/`: reference tables, simulation results and calibrations

| File | Used by | Content |
|---|---|---|
| `GSE120861_gene_gRNAgroup_pair_table.at_scale.txt.gz` | Analysis 1 | Gasperini enhancer (gRNA group)–gene pairs with coordinates, from GEO GSE120861 |
| `gene_detection_summary.parquet` | Analysis 1 | Per gene: number and fraction of Gasperini cells with ≥ 1 count |
| `analysis3_pair_power.parquet` | Analysis 3 | Empirical power from repeated PerTurbo simulations of the Gasperini screen, cis mode: 7,056 enhancer–gene pairs × effect 10/25/50 % × 100–2,000 cells per guide × depth 0.5/1/2×, 30 replicates per design, detection at FDR ≤ 10 % |
| `gasperini_cis_power_calibration.json` | Analysis 1 | Effective z-threshold calibrating analytical power to simulations (Gasperini, cis, decrease) |
| `gasperini_cis_increase_power_calibration_10rep.json` | Analysis 1 | Same for cis increases (provisional, 10 replicates) |
| `replogle_trans_decrease_power_calibration_10rep.json` | Analysis 2 | Same for Replogle trans decreases (provisional, 10 replicates) |
| `calibration_pilot_summary_10rep.csv` | Analysis 3 | Held-out validation metrics of the provisional calibrations |

The calibration files can be regenerated from simulation output with
`scripts/validate_power_calibration.py`.

## `data/`: target programs for Analysis 2

| File | Content |
|---|---|
| `K562_M2Kb_replogle_programs.parquet` | K562 TF → target programs. Targets are genes with a TF ChIP-seq peak (ReMap 2022) within 2 kb of the promoter that are expressed in K562 (ENCODE RNA-seq), restricted to TFs perturbed in Replogle *et al.* 2022. Includes the number of supporting peaks. |
| `collectri_programs.tsv` | Human CollecTRI regulons (Müller-Dott *et al.* 2023) from OmniPath: 62,404 unique TF–target pairs across 1,201 regulators. Self-edges and blank IDs removed, complexes kept intact, no sign or evidence filter. |
| `collectri_provenance.json` | Download URL, date (2026-10-28) and SHA-256 checksum of the raw OmniPath export used to build `collectri_programs.tsv` |
